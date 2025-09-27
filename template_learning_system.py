#!/usr/bin/env python3
"""
TEMPLATE LEARNING SYSTEM
Learn from manual ad placement to auto-detect similar ads in future uploads
"""
import cv2
import numpy as np
import fitz
import os
import json
import base64
from datetime import datetime
from app import db, AdTemplate

class TemplateExtractor:
    """Extract visual and text features from manually placed ads"""

    @staticmethod
    def extract_template_from_region(pdf_path, page_number, x, y, width, height, business_name):
        """
        Extract template features from a manually drawn ad region

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-indexed)
            x, y, width, height: Ad region coordinates
            business_name: Name of the business

        Returns:
            dict: Template features
        """
        try:
            print(f"Extracting template for {business_name} from page {page_number}")

            # Open PDF and get page
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_number - 1)

            # Extract region as image
            rect = fitz.Rect(x, y, x + width, y + height)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=rect)
            img_data = pix.tobytes("png")

            # Convert to OpenCV format
            nparr = np.frombuffer(img_data, np.uint8)
            template_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if template_image is None:
                return None

            # Extract visual features
            visual_features = TemplateExtractor._extract_visual_features(template_image)

            # Extract text from region
            text_content = page.get_textbox(rect)
            text_patterns = TemplateExtractor._extract_text_patterns(text_content)

            # Extract logo features (if present)
            logo_features = TemplateExtractor._extract_logo_features(template_image)

            doc.close()

            # Encode image for storage
            _, buffer = cv2.imencode('.png', template_image)
            template_image_bytes = buffer.tobytes()

            template_data = {
                'business_name': business_name,
                'template_image': template_image_bytes,
                'visual_features': json.dumps(visual_features),
                'text_pattern': json.dumps(text_patterns),
                'logo_features': json.dumps(logo_features),
                'typical_width': width,
                'typical_height': height,
                'aspect_ratio': width / height if height > 0 else 1.0,
                'template_name': f"{business_name} - Standard Ad"
            }

            print(f"SUCCESS: Template extracted successfully for {business_name}")
            return template_data

        except Exception as e:
            print(f"ERROR: Error extracting template: {e}")
            return None

    @staticmethod
    def _extract_visual_features(image):
        """Extract visual signature from ad image"""
        features = {}

        # Color histogram
        hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])

        features['color_histogram'] = {
            'blue': hist_b.flatten().tolist(),
            'green': hist_g.flatten().tolist(),
            'red': hist_r.flatten().tolist()
        }

        # Edge features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features['edge_density'] = float(edge_density)

        # Dominant colors
        pixels = image.reshape(-1, 3)
        dominant_colors = TemplateExtractor._get_dominant_colors(pixels, k=3)
        features['dominant_colors'] = dominant_colors

        # Texture features
        features['texture_variance'] = float(np.var(gray))
        features['brightness_mean'] = float(np.mean(gray))

        return features

    @staticmethod
    def _get_dominant_colors(pixels, k=3):
        """Get dominant colors using K-means clustering"""
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)

            colors = []
            for center in kmeans.cluster_centers_:
                colors.append([int(c) for c in center])
            return colors
        except:
            # Fallback if sklearn not available
            return [[0, 0, 0], [128, 128, 128], [255, 255, 255]]

    @staticmethod
    def _extract_text_patterns(text_content):
        """Extract key text patterns from ad"""
        import re

        patterns = {
            'phone_numbers': re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text_content),
            'key_phrases': [],
            'business_terms': [],
            'text_length': len(text_content.strip())
        }

        # Extract key phrases (quoted text, slogans)
        quoted_text = re.findall(r'"([^"]*)"', text_content)
        patterns['key_phrases'].extend(quoted_text)

        # Extract business-related terms
        business_terms = ['clinic', 'service', 'care', 'professional', 'quality', 'family', 'trusted']
        for term in business_terms:
            if term.lower() in text_content.lower():
                patterns['business_terms'].append(term)

        return patterns

    @staticmethod
    def _extract_logo_features(image):
        """Extract logo/visual elements from ad"""
        features = {}

        # Convert to grayscale for feature detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect corners/keypoints
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=20, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            features['corner_count'] = len(corners)
            features['corner_positions'] = corners.flatten().tolist()
        else:
            features['corner_count'] = 0
            features['corner_positions'] = []

        # Contour analysis for shapes/logos
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features['contour_count'] = len(contours)

        # Shape analysis
        significant_contours = [c for c in contours if cv2.contourArea(c) > 100]
        features['significant_shapes'] = len(significant_contours)

        return features

class TemplateMatcher:
    """Match templates against new ad regions for automatic detection"""

    @staticmethod
    def find_template_matches(pdf_path, page_number, templates, confidence_threshold=0.75):
        """
        Find matches for all active templates on a page

        Args:
            pdf_path: Path to PDF file
            page_number: Page number to search
            templates: List of AdTemplate objects
            confidence_threshold: Minimum confidence for matches

        Returns:
            list: Detected matches with positions and confidence
        """
        matches = []

        try:
            # Convert PDF page to image
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_number - 1)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_data = pix.tobytes("png")

            nparr = np.frombuffer(img_data, np.uint8)
            page_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            doc.close()

            if page_image is None:
                return matches

            # Search for each template
            for template in templates:
                if not template.is_active:
                    continue

                template_matches = TemplateMatcher._match_single_template(
                    page_image, template, confidence_threshold
                )
                matches.extend(template_matches)

            return matches

        except Exception as e:
            print(f"Error finding template matches: {e}")
            return matches

    @staticmethod
    def _match_single_template(page_image, template, confidence_threshold):
        """Match a single template against the page"""
        matches = []

        try:
            # Decode template image
            template_image = cv2.imdecode(
                np.frombuffer(template.template_image, np.uint8),
                cv2.IMREAD_COLOR
            )

            if template_image is None:
                return matches

            # Template matching
            result = cv2.matchTemplate(page_image, template_image, cv2.TM_CCOEFF_NORMED)
            locations = np.where(result >= confidence_threshold)

            # Get matches
            for pt in zip(*locations[::-1]):
                confidence = float(result[pt[1], pt[0]])

                match = {
                    'template_id': template.id,
                    'business_name': template.business_name,
                    'x': float(pt[0]),
                    'y': float(pt[1]),
                    'width': template.typical_width,
                    'height': template.typical_height,
                    'confidence': confidence,
                    'template_name': template.template_name
                }
                matches.append(match)

            # Remove overlapping matches (keep highest confidence)
            matches = TemplateMatcher._remove_overlapping_matches(matches)

        except Exception as e:
            print(f"Error matching template {template.business_name}: {e}")

        return matches

    @staticmethod
    def _remove_overlapping_matches(matches, overlap_threshold=0.5):
        """Remove overlapping matches, keeping the highest confidence ones"""
        if not matches:
            return matches

        # Sort by confidence (highest first)
        sorted_matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)
        filtered_matches = []

        for match in sorted_matches:
            is_overlapping = False

            for existing in filtered_matches:
                overlap = TemplateMatcher._calculate_overlap(match, existing)
                if overlap > overlap_threshold:
                    is_overlapping = True
                    break

            if not is_overlapping:
                filtered_matches.append(match)

        return filtered_matches

    @staticmethod
    def _calculate_overlap(match1, match2):
        """Calculate overlap ratio between two matches"""
        x1, y1, w1, h1 = match1['x'], match1['y'], match1['width'], match1['height']
        x2, y2, w2, h2 = match2['x'], match2['y'], match2['width'], match2['height']

        # Calculate intersection
        inter_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        inter_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        inter_area = inter_x * inter_y

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

if __name__ == "__main__":
    print("Template Learning System initialized")
    print("Ready to learn from manual ad placement")