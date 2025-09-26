#!/usr/bin/env python3
"""
Enhanced PDF Structure Analysis for Newspaper Ad Detection
Replaces the broken sliding window AI detection with precise PDF metadata analysis
"""

import fitz  # PyMuPDF
import os
import re
from collections import defaultdict
from typing import List, Dict, Optional, Tuple


class PDFStructureAdDetector:
    """
    Advanced PDF structure analyzer that extracts exact ad boundaries from PDF structure
    Uses PyMuPDF to extract PDF object metadata instead of visual analysis
    """

    # Business/commercial patterns for ad classification
    COMMERCIAL_PATTERNS = {
        'phone_numbers': [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US phone numbers
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}\b',  # Phone with area code
        ],
        'prices': [
            r'\$\s*\d+(?:[.,]\d{2})?',  # Dollar amounts
            r'\b\d+\s*cents?\b',  # Cents
            r'\bfree\b',  # Free offers
        ],
        'business_info': [
            r'\bwww\.\w+\.\w+\b',  # Websites
            r'\b\w+@\w+\.\w+\b',  # Email addresses
            r'\bhours?\s*:',  # Business hours
            r'\bopen\s+\d',  # Opening times
            r'\bvisit\s+us\b',  # Visit us
            r'\bcontact\s+us\b',  # Contact us
        ],
        'business_keywords': [
            r'\bsale\b', r'\bcall\s+now\b', r'\bspecial\b', r'\boffer\b',
            r'\bdeal\b', r'\bdiscount\b', r'\bservice\b', r'\bstore\b',
            r'\bshop\b', r'\bbusiness\b', r'\bcompany\b'
        ]
    }

    # Editorial patterns (to avoid false positives)
    EDITORIAL_PATTERNS = [
        r'\bby\s+\w+\s+\w+\b',  # "By Author Name"
        r'\bstaff\s+writer\b',  # Staff writer
        r'\bcontinued\s+on\b',  # Continued on page X
        r'\bsee\s+page\s+\d+\b',  # See page X
        r'\byesterday\b', r'\btoday\b', r'\btomorrow\b',  # News time references
    ]

    # Standard newspaper ad sizes (width x height in inches)
    STANDARD_AD_SIZES = {
        # Display ads
        'full_page': (10.0, 13.0),
        'half_page_horizontal': (10.0, 6.5),
        'half_page_vertical': (4.9, 13.0),
        'quarter_page': (4.9, 6.5),
        'eighth_page': (4.9, 3.2),
        'business_card': (3.5, 2.0),

        # Classified sizes
        'classified_1inch': (2.0, 1.0),
        'classified_2inch': (2.0, 2.0),
        'classified_3inch': (2.0, 3.0),

        # Banner sizes
        'banner_full': (10.0, 2.0),
        'banner_half': (4.9, 2.0),

        # Special sizes
        'square_2x2': (2.0, 2.0),
        'square_3x3': (3.0, 3.0),
        'square_4x4': (4.0, 4.0),
    }

    @classmethod
    def detect_ads_from_pdf_structure(cls, pdf_path: str, page_number: int,
                                    publication_type: str = 'broadsheet') -> List[Dict]:
        """
        Main entry point - replaces the broken sliding window detection

        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-based)
            publication_type: 'broadsheet' or 'tabloid'

        Returns:
            List of detected ad regions with precise bounding boxes
        """
        try:
            if not os.path.exists(pdf_path):
                print(f"PDF file not found: {pdf_path}")
                return []

            doc = fitz.open(pdf_path)
            if page_number < 1 or page_number > len(doc):
                print(f"Invalid page number {page_number} for PDF with {len(doc)} pages")
                return []

            page = doc[page_number - 1]  # PyMuPDF uses 0-based indexing

            print(f"Analyzing PDF structure on page {page_number}")

            # Extract all structural elements
            structure_data = cls._extract_page_structure(page)

            # Analyze elements for ad classification
            potential_ads = cls._classify_elements_as_ads(structure_data, page.rect, publication_type)

            # Apply business logic and filtering
            final_ads = cls._filter_and_merge_detections(potential_ads, publication_type)

            doc.close()

            print(f"PDF structure analysis complete: {len(final_ads)} ads detected")
            return final_ads

        except Exception as e:
            print(f"Error in PDF structure analysis: {e}")
            import traceback
            traceback.print_exc()
            return []

    @classmethod
    def _extract_page_structure(cls, page) -> Dict:
        """
        Extract all structural elements from the PDF page
        Returns detailed metadata about text blocks, images, and vector graphics
        """
        structure = {
            'text_blocks': [],
            'images': [],
            'drawings': [],
            'page_rect': page.rect
        }

        # 1. Extract text blocks with detailed typography analysis
        print("Extracting text block structure...")
        text_dict = page.get_text("dict")
        text_blocks = text_dict.get("blocks", [])

        for i, block in enumerate(text_blocks):
            if "lines" in block:  # Text block (not image block)
                text_info = cls._analyze_text_block_structure(block, i)
                if text_info:
                    structure['text_blocks'].append(text_info)

        # 2. Extract images with placement metadata
        print("Extracting image structure...")
        images = page.get_images()

        for i, img in enumerate(images):
            try:
                # Get image metadata and placement
                xref = img[0]
                img_dict = page.parent.extract_image(xref)
                img_rects = page.get_image_rects(img)

                for j, rect in enumerate(img_rects):
                    image_info = {
                        'id': f"img_{i}_{j}",
                        'bounds': [float(rect.x0), float(rect.y0),
                                  float(rect.x1), float(rect.y1)],
                        'width': float(rect.width),
                        'height': float(rect.height),
                        'area': float(rect.width * rect.height),
                        'aspect_ratio': float(rect.width / rect.height) if rect.height > 0 else 0,
                        'image_width_px': img_dict.get('width', 0),
                        'image_height_px': img_dict.get('height', 0),
                        'colorspace': img_dict.get('colorspace', 'unknown'),
                        'file_size': len(img_dict.get('image', b'')),
                        'position': cls._analyze_position_context(rect, page.rect)
                    }
                    structure['images'].append(image_info)

            except Exception as e:
                print(f"   Error processing image {i}: {e}")

        # 3. Extract vector graphics with path analysis
        print("Extracting vector graphics structure...")
        drawings = page.get_drawings()

        for i, drawing in enumerate(drawings):
            try:
                drawing_info = {
                    'id': f"drawing_{i}",
                    'bounds': [float(drawing['rect'].x0), float(drawing['rect'].y0),
                              float(drawing['rect'].x1), float(drawing['rect'].y1)],
                    'width': float(drawing['rect'].width),
                    'height': float(drawing['rect'].height),
                    'area': float(drawing['rect'].width * drawing['rect'].height),
                    'items': drawing.get('items', []),
                    'item_count': len(drawing.get('items', [])),
                    'is_rectangle': cls._is_rectangular_border(drawing),
                    'is_complex': len(drawing.get('items', [])) > 5,
                    'position': cls._analyze_position_context(drawing['rect'], page.rect)
                }
                structure['drawings'].append(drawing_info)

            except Exception as e:
                print(f"   Error processing drawing {i}: {e}")

        print(f"   Found {len(structure['text_blocks'])} text blocks, "
              f"{len(structure['images'])} images, {len(structure['drawings'])} drawings")

        return structure

    @classmethod
    def _analyze_text_block_structure(cls, block, block_id) -> Optional[Dict]:
        """
        Detailed analysis of text block typography and content patterns
        """
        bbox = block["bbox"]
        fonts = set()
        font_sizes = set()
        font_weights = set()
        total_chars = 0
        text_content = ""

        # Analyze typography across all spans
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_name = span.get("font", "unknown")
                fonts.add(font_name)
                font_sizes.add(span.get("size", 0))

                # Extract weight from font name
                if "bold" in font_name.lower():
                    font_weights.add("bold")
                elif "light" in font_name.lower():
                    font_weights.add("light")
                else:
                    font_weights.add("regular")

                span_text = span.get("text", "")
                text_content += span_text + " "
                total_chars += len(span_text)

        text_content = text_content.strip()

        if not text_content or total_chars < 5:  # Skip tiny text blocks
            return None

        return {
            'id': f"text_{block_id}",
            'bounds': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
            'width': float(bbox[2] - bbox[0]),
            'height': float(bbox[3] - bbox[1]),
            'area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
            'font_count': len(fonts),
            'fonts': list(fonts),
            'font_sizes': sorted(list(font_sizes)),
            'font_weights': list(font_weights),
            'char_count': total_chars,
            'text_content': text_content,
            'text_preview': text_content[:100].replace('\n', ' ').strip(),
            'has_mixed_fonts': len(fonts) > 1,
            'has_mixed_sizes': len(font_sizes) > 1,
            'business_score': cls._calculate_business_score(text_content),
            'editorial_score': cls._calculate_editorial_score(text_content),
            'position': cls._analyze_position_context([bbox[0], bbox[1], bbox[2], bbox[3]], None)
        }

    @classmethod
    def _calculate_business_score(cls, text: str) -> float:
        """
        Calculate how likely text is commercial/business content
        """
        text_lower = text.lower()
        score = 0.0

        # Check for business patterns
        for pattern_type, patterns in cls.COMMERCIAL_PATTERNS.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                if pattern_type == 'phone_numbers':
                    score += matches * 20  # Phone numbers are strong indicators
                elif pattern_type == 'prices':
                    score += matches * 15  # Prices are strong indicators
                elif pattern_type == 'business_info':
                    score += matches * 10  # Contact info is strong
                else:
                    score += matches * 5   # General business keywords

        # Length normalization - longer text needs higher absolute score
        if len(text) > 100:
            score = score * 0.8  # Reduce score for very long text (likely editorial)

        return min(score, 100.0)  # Cap at 100

    @classmethod
    def _calculate_editorial_score(cls, text: str) -> float:
        """
        Calculate how likely text is editorial content
        """
        text_lower = text.lower()
        score = 0.0

        # Check for editorial patterns
        for pattern in cls.EDITORIAL_PATTERNS:
            matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
            score += matches * 10

        # Length bonus for editorial content
        if len(text) > 200:
            score += 20  # Long text blocks are more likely editorial

        return min(score, 100.0)  # Cap at 100

    @classmethod
    def _analyze_position_context(cls, bounds, page_rect) -> Dict:
        """
        Analyze the position context of an element on the page
        bounds can be [x0, y0, x1, y1] list or rect object
        """
        # Handle different input types
        if isinstance(bounds, list):
            x0, y0, x1, y1 = bounds
            width = x1 - x0
            height = y1 - y0
        else:
            x0, y0 = bounds.x0, bounds.y0
            width, height = bounds.width, bounds.height

        # Default values if page_rect is not available
        if page_rect is None:
            return {
                'relative_x': 0.5,
                'relative_y': 0.5,
                'relative_center_x': 0.5,
                'relative_center_y': 0.5,
                'zones': ['center', 'middle'],
                'is_edge_element': False
            }

        # Calculate relative positions
        rel_x = x0 / page_rect.width if hasattr(page_rect, 'width') else x0 / 600
        rel_y = y0 / page_rect.height if hasattr(page_rect, 'height') else y0 / 800
        rel_center_x = (x0 + width / 2) / (page_rect.width if hasattr(page_rect, 'width') else 600)
        rel_center_y = (y0 + height / 2) / (page_rect.height if hasattr(page_rect, 'height') else 800)

        # Determine zones
        zones = []
        if rel_x < 0.1:
            zones.append('left_edge')
        elif rel_x > 0.9:
            zones.append('right_edge')
        else:
            zones.append('center')

        if rel_y < 0.1:
            zones.append('top_edge')
        elif rel_y > 0.9:
            zones.append('bottom_edge')
        else:
            zones.append('middle')

        return {
            'relative_x': rel_x,
            'relative_y': rel_y,
            'relative_center_x': rel_center_x,
            'relative_center_y': rel_center_y,
            'zones': zones,
            'is_edge_element': 'edge' in ' '.join(zones)
        }

    @classmethod
    def _is_rectangular_border(cls, drawing) -> bool:
        """
        Analyze if a drawing represents a rectangular border/frame
        """
        items = drawing.get('items', [])

        # Simple heuristic: rectangular borders have few items (1-4 typically)
        # and create a closed rectangular path
        if len(items) == 0:
            return False

        if len(items) <= 4:
            # Check if it forms a rectangle by analyzing the bounding box
            rect = drawing.get('rect')
            if rect and rect.width > 50 and rect.height > 30:  # Minimum size for ad border
                return True

        return False

    @classmethod
    def _classify_elements_as_ads(cls, structure_data: Dict, page_rect, publication_type: str) -> List[Dict]:
        """
        Classify extracted elements as potential advertisements based on structure
        """
        potential_ads = []

        # 1. Analyze images for ad characteristics
        for img in structure_data['images']:
            img_score = cls._score_image_as_ad(img, page_rect)
            if img_score > 20:  # LOWERED threshold for image ads (was 30)
                potential_ads.append({
                    'x': img['bounds'][0],
                    'y': img['bounds'][1],
                    'width': img['bounds'][2] - img['bounds'][0],
                    'height': img['bounds'][3] - img['bounds'][1],
                    'confidence': img_score / 100.0,
                    'type': 'image_ad',
                    'source_elements': [img['id']],
                    'classification': 'business_image'
                })

        # 2. Analyze text blocks for ad characteristics
        for text in structure_data['text_blocks']:
            text_score = cls._score_text_as_ad(text, page_rect)
            if text_score > 25:  # LOWERED threshold for text ads (was 40)
                potential_ads.append({
                    'x': text['bounds'][0],
                    'y': text['bounds'][1],
                    'width': text['width'],
                    'height': text['height'],
                    'confidence': text_score / 100.0,
                    'type': 'text_ad',
                    'source_elements': [text['id']],
                    'classification': 'business_text'
                })

        # 3. Find grouped elements (image + text combinations)
        grouped_ads = cls._find_grouped_ad_elements(structure_data, page_rect)
        potential_ads.extend(grouped_ads)

        # 4. Analyze bordered regions
        bordered_ads = cls._find_bordered_ad_regions(structure_data, page_rect)
        potential_ads.extend(bordered_ads)

        return potential_ads

    @classmethod
    def _score_image_as_ad(cls, img: Dict, page_rect) -> float:
        """
        Score an image based on ad likelihood using structural patterns
        """
        score = 0.0

        # Size analysis
        area = img['area']
        if 5000 <= area <= 500000:  # Reasonable ad image sizes
            score += 25
        elif area > 500000:
            score += 10  # Large images could be ads or editorial photos

        # Aspect ratio analysis
        aspect_ratio = img['aspect_ratio']
        if 0.2 <= aspect_ratio <= 5.0:  # Reasonable ad proportions
            score += 20

        # Position analysis - ads often in specific zones
        position = img['position']
        if not position['is_edge_element']:
            score += 15  # Not at page edges

        # File size analysis - ads tend to be substantial images
        if img['file_size'] > 20000:  # > 20KB suggests quality image
            score += 15

        # Quality analysis
        if img['image_width_px'] >= 150 and img['image_height_px'] >= 100:
            score += 10  # Reasonable resolution

        return score

    @classmethod
    def _score_text_as_ad(cls, text: Dict, page_rect) -> float:
        """
        Score text block based on ad likelihood using typography and content
        """
        base_score = text['business_score'] - text['editorial_score']

        # ENHANCED: More generous typography bonuses for clear ad indicators
        if text['has_mixed_fonts']:
            base_score += 30  # Mixed fonts are strong ad indicators

        if text['has_mixed_sizes']:
            base_score += 25  # Mixed sizes suggest hierarchical ad design

        if 'bold' in text['font_weights']:
            base_score += 15  # Bold text very common in ads

        # Additional bonus for designed layouts (multiple typography features)
        if text['has_mixed_fonts'] and text['has_mixed_sizes']:
            base_score += 15  # Combo bonus for clearly designed content

        # Size analysis - ads have typical size ranges
        area = text['area']
        if 5000 <= area <= 200000:  # Typical ad text area
            base_score += 15

        # Position bonuses
        position = text['position']
        if 0.1 <= position['relative_x'] <= 0.9:  # Not at extreme edges
            base_score += 10

        return max(0, base_score)  # Don't return negative scores

    @classmethod
    def _find_grouped_ad_elements(cls, structure_data: Dict, page_rect) -> List[Dict]:
        """
        Find image+text combinations that form cohesive ad units
        """
        grouped_ads = []
        proximity_threshold = 100  # pixels

        for img in structure_data['images']:
            for text in structure_data['text_blocks']:
                if cls._are_elements_grouped(img['bounds'], text['bounds'], proximity_threshold):
                    # Calculate combined confidence
                    img_score = cls._score_image_as_ad(img, page_rect)
                    text_score = cls._score_text_as_ad(text, page_rect)

                    if img_score > 15 and text_score > 15:  # LOWERED thresholds for grouped elements
                        combined_bounds = cls._combine_bounds(img['bounds'], text['bounds'])
                        combined_confidence = (img_score + text_score + 30) / 100.0  # Grouping bonus

                        grouped_ads.append({
                            'x': combined_bounds[0],
                            'y': combined_bounds[1],
                            'width': combined_bounds[2] - combined_bounds[0],
                            'height': combined_bounds[3] - combined_bounds[1],
                            'confidence': min(combined_confidence, 0.95),
                            'type': 'mixed_ad',
                            'source_elements': [img['id'], text['id']],
                            'classification': 'image_text_combo'
                        })

        return grouped_ads

    @classmethod
    def _find_bordered_ad_regions(cls, structure_data: Dict, page_rect) -> List[Dict]:
        """
        Find rectangular borders that likely contain advertisements
        """
        bordered_ads = []

        for drawing in structure_data['drawings']:
            if drawing['is_rectangle'] and not drawing['is_complex']:
                # Check if border size suggests ad content
                width = drawing['width']
                height = drawing['height']
                area = drawing['area']

                if 10000 <= area <= 600000:  # Reasonable ad container sizes
                    # Look for content inside this border
                    has_content = cls._has_content_inside_border(
                        drawing['bounds'], structure_data['text_blocks'], structure_data['images']
                    )

                    if has_content:
                        bordered_ads.append({
                            'x': drawing['bounds'][0],
                            'y': drawing['bounds'][1],
                            'width': width,
                            'height': height,
                            'confidence': 0.85,  # High confidence for bordered content
                            'type': 'bordered_ad',
                            'source_elements': [drawing['id']],
                            'classification': 'bordered_content'
                        })

        return bordered_ads

    @classmethod
    def _are_elements_grouped(cls, bounds1: List[float], bounds2: List[float], threshold: float) -> bool:
        """
        Check if two elements are close enough to be part of the same ad
        """
        # Calculate distance between centers
        center1_x = (bounds1[0] + bounds1[2]) / 2
        center1_y = (bounds1[1] + bounds1[3]) / 2
        center2_x = (bounds2[0] + bounds2[2]) / 2
        center2_y = (bounds2[1] + bounds2[3]) / 2

        distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
        return distance <= threshold

    @classmethod
    def _combine_bounds(cls, bounds1: List[float], bounds2: List[float]) -> List[float]:
        """
        Combine two bounding boxes into a single encompassing box
        """
        return [
            min(bounds1[0], bounds2[0]),  # min x
            min(bounds1[1], bounds2[1]),  # min y
            max(bounds1[2], bounds2[2]),  # max x
            max(bounds1[3], bounds2[3])   # max y
        ]

    @classmethod
    def _has_content_inside_border(cls, border_bounds: List[float], text_blocks: List, images: List) -> bool:
        """
        Check if a border rectangle contains text or image content
        """
        border_x1, border_y1, border_x2, border_y2 = border_bounds

        # Check for text content inside border
        for text in text_blocks:
            text_x1, text_y1, text_x2, text_y2 = text['bounds']
            if (border_x1 <= text_x1 and text_x2 <= border_x2 and
                border_y1 <= text_y1 and text_y2 <= border_y2):
                return True

        # Check for image content inside border
        for img in images:
            img_x1, img_y1, img_x2, img_y2 = img['bounds']
            if (border_x1 <= img_x1 and img_x2 <= border_x2 and
                border_y1 <= img_y1 and img_y2 <= border_y2):
                return True

        return False

    @classmethod
    def _filter_and_merge_detections(cls, detections: List[Dict], publication_type: str) -> List[Dict]:
        """
        Apply business logic to filter and merge detected ads
        """
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        # Remove overlapping detections
        filtered = []
        overlap_threshold = 0.3

        for detection in detections:
            overlaps = False
            for existing in filtered:
                if cls._calculate_overlap_ratio(detection, existing) > overlap_threshold:
                    overlaps = True
                    break

            if not overlaps:
                filtered.append(detection)

        # FIXED: More flexible size validation - don't reject ads based on arbitrary size rules
        final_ads = []

        # Accept ads based on confidence and structural analysis, not rigid size constraints
        for ad in filtered:
            # Accept all high-confidence detections (structural analysis is reliable)
            if ad['confidence'] > 0.5:  # Lowered threshold from 0.8
                final_ads.append(ad)
                print(f"Detected {ad['type']}: {ad['width']:.0f}x{ad['height']:.0f}px - confidence: {ad['confidence']:.2f}")
            else:
                print(f"Rejected {ad['type']}: confidence {ad['confidence']:.2f} too low")

        return final_ads

    @classmethod
    def _calculate_overlap_ratio(cls, ad1: Dict, ad2: Dict) -> float:
        """
        Calculate overlap ratio between two ad detections
        """
        x1 = max(ad1['x'], ad2['x'])
        y1 = max(ad1['y'], ad2['y'])
        x2 = min(ad1['x'] + ad1['width'], ad2['x'] + ad2['width'])
        y2 = min(ad1['y'] + ad1['height'], ad2['y'] + ad2['height'])

        if x2 <= x1 or y2 <= y1:
            return 0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = ad1['width'] * ad1['height']
        area2 = ad2['width'] * ad2['height']
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    @classmethod
    def _matches_standard_ad_size(cls, width_inches: float, height_inches: float,
                                  tolerance: float = 0.5) -> bool:
        """
        Check if dimensions match standard newspaper ad sizes within tolerance
        """
        for size_name, (std_width, std_height) in cls.STANDARD_AD_SIZES.items():
            # Check both orientations
            if (abs(width_inches - std_width) <= tolerance and
                abs(height_inches - std_height) <= tolerance):
                return True
            if (abs(width_inches - std_height) <= tolerance and
                abs(height_inches - std_width) <= tolerance):
                return True

        return False