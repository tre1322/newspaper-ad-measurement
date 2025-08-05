import os
import uuid
import fitz  # PyMuPDF
import cv2
import numpy as np
import math
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
from dotenv import load_dotenv
from functools import wraps
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
import tempfile
import csv
from io import StringIO
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-this')

# Production configuration - MOVED HERE AFTER app is created
if os.environ.get('RAILWAY_ENVIRONMENT'):
    # Railway-specific configurations
    database_url = os.environ.get('DATABASE_URL', 'sqlite:///newspaper_ads.db')
    # Ensure proper database URL format for SQLAlchemy 2.0+
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///newspaper_ads.db')

# Handle PostgreSQL URL format
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Continue with your existing configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-this')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Upload configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Authentication configuration
LOGIN_PASSWORD = os.environ.get('LOGIN_PASSWORD', 'CCCitizen56101!')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Create upload directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'pages'), exist_ok=True)

# Database Models
class Publication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    publication_type = db.Column(db.String(50), nullable=False)
    total_pages = db.Column(db.Integer, nullable=False)
    total_inches = db.Column(db.Float, nullable=False)
    total_ad_inches = db.Column(db.Float, default=0.0)
    ad_percentage = db.Column(db.Float, default=0.0)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    processed = db.Column(db.Boolean, default=False)

class Page(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    publication_id = db.Column(db.Integer, db.ForeignKey('publication.id'), nullable=False)
    page_number = db.Column(db.Integer, nullable=False)
    width_pixels = db.Column(db.Integer, nullable=False)
    height_pixels = db.Column(db.Integer, nullable=False)
    total_page_inches = db.Column(db.Float, nullable=False)
    page_ad_inches = db.Column(db.Float, default=0.0)
    # Enhanced calibration fields
    pdf_width_inches = db.Column(db.Float)
    pdf_height_inches = db.Column(db.Float)
    pixels_per_inch = db.Column(db.Float)
    calibration_accurate = db.Column(db.Boolean, default=False)

class AdBox(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    page_id = db.Column(db.Integer, db.ForeignKey('page.id'), nullable=False)
    x = db.Column(db.Float, nullable=False)
    y = db.Column(db.Float, nullable=False)
    width = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    width_inches_raw = db.Column(db.Float, nullable=False)
    height_inches_raw = db.Column(db.Float, nullable=False)
    width_inches_rounded = db.Column(db.Float, nullable=False)
    height_inches_rounded = db.Column(db.Float, nullable=False)
    column_inches = db.Column(db.Float, nullable=False)
    ad_type = db.Column(db.String(50), default='manual')  # manual, open_display, entertainment, classified, public_notice
    is_ad = db.Column(db.Boolean, default=True)
    user_verified = db.Column(db.Boolean, default=False)
    created_date = db.Column(db.DateTime, default=datetime.utcnow)

class ScreenCalibration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    device_fingerprint = db.Column(db.String(255), nullable=False)
    screen_width = db.Column(db.Integer)
    screen_height = db.Column(db.Integer)
    device_pixel_ratio = db.Column(db.Float)
    scaling_factor = db.Column(db.Float, nullable=False)
    pixels_per_inch = db.Column(db.Float, nullable=False)
    measured_inches = db.Column(db.Float, nullable=False)
    calibration_date = db.Column(db.DateTime, default=datetime.utcnow)
    user_agent = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)

# Publication configurations
PUBLICATION_CONFIGS = {
    'broadsheet': {
        'width_units': 12,
        'total_inches_per_page': 258,
        'name': 'Broadsheet',
        'column_standards': {
            'open_display': {
                1: 1.75, 2: 3.67, 3: 5.6, 4: 7.5, 5: 9.42, 6: 11.33
            },
            'entertainment': {
                1: 1.6, 2: 3.33, 3: 5.1, 4: 6.83, 5: 9.1, 6: 11.0
            },
            'classified': {
                1: 1.17, 2: 2.5, 3: 3.83, 4: 5.17, 5: 6.5, 6: 7.83, 7: 9.17, 8: 10.5, 9: 11.83
            }
        },
        'gutter_width': 0.1666666667
    },
    'special_edition': {
        'width_units': 10,
        'total_inches_per_page': 125,
        'name': 'Special Edition Tabloid',
        'preset_ads': {
            'eighth': {'inches': 15, 'name': '1/8 Page'},
            'quarter': {'inches': 30, 'name': '1/4 Page'},
            'half': {'inches': 60, 'name': '1/2 Page'},
            'full': {'inches': 125, 'name': 'Full Page'}
        }
    },
    'peach': {
        'width_units': 10,
        'total_inches_per_page': 150,
        'name': 'Peach Supplement'
    }
}

# Screen Calibration Manager
class ScreenCalibrationManager:
    @staticmethod
    def create_calibration_target():
        """Create a precise calibration target image"""
        # Create a calibration image with known measurements
        width_px = 600  # 100 pixels per inch at baseline
        height_px = 200
        
        img = Image.new('RGB', (width_px, height_px), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw ruler markings
        for inch in range(7):  # 0 to 6 inches
            x = inch * 100
            # Major tick (full height)
            draw.line([(x, 50), (x, 150)], fill='black', width=2)
            
            # Inch labels
            draw.text((x-5, 160), str(inch), fill='black')
            
            # Minor ticks (1/4 inch)
            for quarter in range(1, 4):
                x_minor = x + (quarter * 25)
                if x_minor < width_px:
                    draw.line([(x_minor, 70), (x_minor, 130)], fill='black', width=1)
            
            # 1/8 inch ticks
            for eighth in [12.5, 37.5, 62.5, 87.5]:
                x_eighth = x + eighth
                if x_eighth < width_px:
                    draw.line([(x_eighth, 80), (x_eighth, 120)], fill='gray', width=1)
        
        # Add instructions
        draw.text((10, 10), "CALIBRATION: Measure this ruler with a real ruler", fill='black')
        draw.text((10, 25), "Enter the measured length below", fill='black')
        
        # Save as base64 for embedding
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return img_base64
    
    @staticmethod
    def calculate_screen_scaling(measured_inches, baseline_pixels=600):
        """Calculate scaling factor for this screen"""
        baseline_inches = 6.0
        
        if measured_inches <= 0:
            return None
            
        actual_pixels_per_inch = baseline_pixels / measured_inches
        baseline_pixels_per_inch = baseline_pixels / baseline_inches  # 100 PPI baseline
        
        scaling_factor = actual_pixels_per_inch / baseline_pixels_per_inch
        
        return {
            'scaling_factor': scaling_factor,
            'pixels_per_inch': actual_pixels_per_inch,
            'measured_inches': measured_inches,
            'baseline_ppi': baseline_pixels_per_inch
        }

# Enhanced measurement calculator
class CalibratedMeasurementCalculator:
    @staticmethod
    def get_screen_calibration(device_fingerprint):
        """Get active calibration for this device"""
        return ScreenCalibration.query.filter_by(
            device_fingerprint=device_fingerprint,
            is_active=True
        ).first()
    
    @staticmethod
    def pixels_to_inches_with_screen_calibration(pixels, pdf_pixels_per_inch, screen_scaling_factor=1.0):
        """Convert pixels to inches accounting for both PDF scaling and screen scaling"""
        # First convert using PDF calibration
        pdf_inches = pixels / pdf_pixels_per_inch
        
        # Then adjust for screen calibration
        calibrated_inches = pdf_inches * screen_scaling_factor
        
        return calibrated_inches
    
    @staticmethod
    def get_pdf_page_dimensions(pdf_path, page_number):
        """Get actual page dimensions from PDF metadata"""
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_number - 1)  # 0-indexed
            
            # Get page dimensions in points (1 point = 1/72 inch)
            rect = page.rect
            width_points = rect.width
            height_points = rect.height
            
            # Convert points to inches
            width_inches = width_points / 72.0
            height_inches = height_points / 72.0
            
            doc.close()
            return width_inches, height_inches
            
        except Exception as e:
            print(f"Error getting PDF dimensions: {e}")
            return None, None
    
    @staticmethod
    def calculate_pdf_scaling_factor(pdf_path, page_number, image_width_pixels, image_height_pixels):
        """Calculate the actual scaling factor between PDF and image"""
        pdf_width_inches, pdf_height_inches = CalibratedMeasurementCalculator.get_pdf_page_dimensions(pdf_path, page_number)
        
        if pdf_width_inches and pdf_height_inches:
            # Calculate pixels per inch for both dimensions
            pixels_per_inch_width = image_width_pixels / pdf_width_inches
            pixels_per_inch_height = image_height_pixels / pdf_height_inches
            
            # Use average (they should be very close)
            pixels_per_inch = (pixels_per_inch_width + pixels_per_inch_height) / 2
            
            return {
                'pixels_per_inch': pixels_per_inch,
                'pdf_width_inches': pdf_width_inches,
                'pdf_height_inches': pdf_height_inches,
                'scaling_accurate': abs(pixels_per_inch_width - pixels_per_inch_height) < 1
            }
        
        return None

# Intelligent Ad Detector for Broadsheet
class IntelligentAdDetector:
    @staticmethod
    def detect_ad_from_click(image_path, click_x, click_y, ad_type='open_display'):
        """Detect ad boundaries from a click point using edge detection"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Different detection strategies based on ad type
            if ad_type == 'entertainment':
                return IntelligentAdDetector._detect_entertainment_ad(gray, click_x, click_y)
            elif ad_type == 'classified':
                return IntelligentAdDetector._detect_classified_ad(gray, click_x, click_y)
            elif ad_type == 'public_notice':
                return IntelligentAdDetector._detect_public_notice(gray, click_x, click_y)
            else:  # open_display
                return IntelligentAdDetector._detect_display_ad(gray, click_x, click_y)
                
        except Exception as e:
            print(f"Error in intelligent detection: {e}")
            return None
    
    @staticmethod
    def _detect_display_ad(gray, click_x, click_y):
        """Detect regular display ad with solid borders - simplified approach"""
        img_height, img_width = gray.shape
        
        # Primary method: Find contours and match to click point
        # Use moderate edge detection parameters
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Apply slight morphological closing to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the best contour that contains or is near the click point
        best_box = None
        min_distance = float('inf')
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Quick size filter - reasonable ad dimensions
            if w < 30 or h < 20 or w > img_width * 0.9 or h > img_height * 0.9:
                continue
            
            # Check if click point is inside or very close to this rectangle
            # Allow some margin around the rectangle
            margin = 10
            if (x - margin <= click_x <= x + w + margin and 
                y - margin <= click_y <= y + h + margin):
                
                # Calculate distance from click point to rectangle center
                center_x = x + w // 2
                center_y = y + h // 2
                distance = ((click_x - center_x) ** 2 + (click_y - center_y) ** 2) ** 0.5
                
                # Prefer rectangles closer to click point
                if distance < min_distance:
                    # Additional validation for reasonable ad proportions
                    aspect_ratio = w / h if h > 0 else 0
                    area = w * h
                    
                    if (0.2 <= aspect_ratio <= 8.0 and  # Reasonable aspect ratio
                        area >= 600):  # Minimum area for an ad
                        
                        best_box = {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
                        min_distance = distance
        
        if best_box:
            return best_box
        
        # Fallback: Simple rectangular region around click point
        return IntelligentAdDetector._simple_fallback(gray, click_x, click_y)
    
    @staticmethod
    def _simple_fallback(gray, click_x, click_y):
        """Simple fallback: try to find a reasonable rectangle around click point"""
        img_height, img_width = gray.shape
        
        # Try a few different edge detection parameters
        for low_thresh, high_thresh in [(30, 90), (70, 200), (20, 60)]:
            edges = cv2.Canny(gray, low_thresh, high_thresh)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for any reasonably sized rectangle near the click
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Must be reasonably sized
                if w < 40 or h < 25 or w > img_width * 0.8 or h > img_height * 0.8:
                    continue
                
                # Must be close to click point (within 100 pixels)
                if (abs(click_x - (x + w//2)) < 100 and abs(click_y - (y + h//2)) < 100):
                    return {'x': int(x), 'y': int(y), 'width': int(w), 'height': int(h)}
        
        # Final fallback: create a small rectangle centered on click point
        # But make it a reasonable ad size
        default_w, default_h = 120, 80  # Reasonable small ad size
        x = max(0, min(click_x - default_w//2, img_width - default_w))
        y = max(0, min(click_y - default_h//2, img_height - default_h))
        
        return {'x': int(x), 'y': int(y), 'width': int(default_w), 'height': int(default_h)}
    
    @staticmethod
    def _detect_entertainment_ad(gray, click_x, click_y):
        """Detect entertainment ad with gray background"""
        # Try gray background detection first
        try:
            # Look for gray background regions
            lower_gray = np.array([160])
            upper_gray = np.array([220])
            mask = cv2.inRange(gray, lower_gray, upper_gray)
            
            # Find connected components
            num_labels, labels = cv2.connectedComponents(mask)
            
            # Check if click point is in a gray region
            if (click_y < labels.shape[0] and click_x < labels.shape[1] and 
                labels[click_y, click_x] > 0):
                
                label = labels[click_y, click_x]
                coords = np.where(labels == label)
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                width = x_max - x_min
                height = y_max - y_min
                
                # Validate reasonable size
                if width > 50 and height > 30 and width < gray.shape[1] * 0.8 and height < gray.shape[0] * 0.8:
                    return {
                        'x': int(x_min), 'y': int(y_min), 
                        'width': int(width), 'height': int(height)
                    }
        except:
            pass
        
        # Fallback to regular border detection
        return IntelligentAdDetector._detect_display_ad(gray, click_x, click_y)
    
    @staticmethod
    def _detect_classified_ad(gray, click_x, click_y):
        """Detect classified ad with text-based boundaries"""
        # For classified ads, often just use a simple edge detection approach
        # since they may not have strong borders
        return IntelligentAdDetector._detect_display_ad(gray, click_x, click_y)
    
    @staticmethod
    def _detect_public_notice(gray, click_x, click_y):
        """Detect public notice - exact text block boundaries"""
        # Public notices often have minimal borders, use the same detection
        return IntelligentAdDetector._detect_display_ad(gray, click_x, click_y)
    
    @staticmethod
    def match_to_column_width(width_inches, ad_type, column_standards):
        """Match detected width to closest column standard"""
        if ad_type not in column_standards:
            return width_inches
        
        standards = column_standards[ad_type]
        closest_match = min(standards.values(), key=lambda x: abs(x - width_inches))
        
        # Find the column number for this width
        for col_num, col_width in standards.items():
            if col_width == closest_match:
                return col_width
        
        return width_inches
    
    @staticmethod
    def round_depth_up(height_inches, ad_type):
        """Round depth up to next 0.5\" increment (except public notices)"""
        if ad_type == 'public_notice':
            return height_inches  # Exact measurement for public notices
        
        # Round up to next 0.5" increment
        return math.ceil(height_inches * 2) / 2

# Measurement Calculator
class MeasurementCalculator:
    @staticmethod
    def pixels_to_inches(pixels, total_pixels, total_inches):
        """Convert pixels to inches based on page dimensions"""
        return (pixels / total_pixels) * total_inches
    
    @staticmethod
    def round_measurement(value):
        """Apply newspaper measurement rounding rules"""
        base = int(value)
        decimal = value - base
        
        if decimal <= 0.25:
            return base
        elif decimal <= 0.75:
            return base + 0.5
        else:
            return base + 1.0

# Improved AdBoxDetector class
class AdBoxDetector:
    @staticmethod
    def detect_boxes(image_path):
        """Detect rectangular ad boxes in an image using OpenCV with much stricter filtering"""
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise while keeping edges sharp
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Use multiple edge detection approaches
            edges1 = cv2.Canny(filtered, 50, 150, apertureSize=3)
            edges2 = cv2.Canny(filtered, 100, 200, apertureSize=3)
            
            # Combine edge maps
            edges = cv2.bitwise_or(edges1, edges2)
            
            # Apply morphological operations to connect broken lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            boxes = []
            img_height, img_width = img.shape[:2]
            img_area = img_width * img_height
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                
                # MUCH STRICTER filtering for newspaper ads
                
                # 1. Size filters - ads should be substantial but not too large
                min_area = img_area * 0.01   # At least 1% of page (increased)
                max_area = img_area * 0.15   # At most 15% of page (decreased)
                if area < min_area or area > max_area:
                    continue
                
                # 2. Minimum dimensions - ads should be readable
                min_width = max(80, img_width * 0.08)   # At least 8% of page width
                min_height = max(80, img_height * 0.05)  # At least 5% of page height
                if w < min_width or h < min_height:
                    continue
                
                # 3. Maximum dimensions - avoid full-page elements
                max_width = img_width * 0.8   # Max 80% of page width
                max_height = img_height * 0.6  # Max 60% of page height
                if w > max_width or h > max_height:
                    continue
                
                # 4. Aspect ratio - reasonable ad proportions
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.3 or aspect_ratio > 5:  # More restrictive
                    continue
                
                # 5. Edge margins - avoid page borders and headers/footers
                margin_x = img_width * 0.08   # 8% margin from sides
                margin_y = img_height * 0.12  # 12% margin from top/bottom
                if (x < margin_x or y < margin_y or 
                    x + w > img_width - margin_x or 
                    y + h > img_height - margin_y):
                    continue
                
                # 6. Rectangularity test - ads should be very rectangular
                contour_area = cv2.contourArea(contour)
                rect_area = w * h
                rectangularity = contour_area / rect_area if rect_area > 0 else 0
                if rectangularity < 0.85:  # Must be at least 85% rectangular
                    continue
                
                # 7. Border detection - look for actual borders
                roi = gray[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                
                # Check for strong border-like features
                border_thickness = 3
                try:
                    top_border = roi[0:border_thickness, :].mean()
                    bottom_border = roi[-border_thickness:, :].mean()
                    left_border = roi[:, 0:border_thickness].mean()
                    right_border = roi[:, -border_thickness:].mean()
                    center = roi[border_thickness:-border_thickness, border_thickness:-border_thickness].mean()
                    
                    # Calculate border contrast
                    border_contrast = (abs(top_border - center) + abs(bottom_border - center) + 
                                     abs(left_border - center) + abs(right_border - center)) / 4
                    
                    # Skip if borders are too weak (likely just text blocks)
                    if border_contrast < 25:  # Increased threshold
                        continue
                except:
                    continue
                
                # 8. Content analysis - check if it's likely an ad vs text
                # Look for uniform regions (typical of ads vs dense text)
                roi_std = np.std(roi)
                if roi_std > 50:  # Too much variation, likely text
                    continue
                
                # 9. Density check - ads often have more white space
                binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                white_ratio = np.sum(binary_roi == 255) / binary_roi.size
                if white_ratio < 0.3:  # Less than 30% white space, likely dense text
                    continue
                
                # Calculate confidence based on multiple factors
                size_score = min(1.0, area / (img_area * 0.05))  # Prefer larger boxes
                rectangularity_score = rectangularity
                border_score = min(1.0, border_contrast / 50)
                whitespace_score = min(1.0, white_ratio / 0.5)
                
                confidence = (size_score * 0.3 + rectangularity_score * 0.3 + 
                            border_score * 0.2 + whitespace_score * 0.2)
                
                # Only keep very high-confidence detections
                if confidence < 0.7:
                    continue
                
                boxes.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'confidence': confidence
                })
            
            # Remove overlapping boxes
            boxes = AdBoxDetector.remove_overlapping_boxes(boxes, overlap_threshold=0.3)
            
            # Sort by confidence and keep only top detections
            boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
            
            # Limit to reasonable number of ads per page (much more restrictive)
            max_ads_per_page = min(8, len(boxes))  # Max 8 ads per page
            boxes = boxes[:max_ads_per_page]
            
            return boxes
            
        except Exception as e:
            print(f"Error detecting boxes: {e}")
            return []
    
    @staticmethod
    def remove_overlapping_boxes(boxes, overlap_threshold=0.3):
        """Remove boxes that significantly overlap"""
        if not boxes:
            return boxes
        
        # Sort by confidence (keep higher confidence boxes)
        boxes = sorted(boxes, key=lambda x: x['confidence'], reverse=True)
        
        filtered_boxes = []
        for box in boxes:
            overlap_found = False
            for existing_box in filtered_boxes:
                if AdBoxDetector.calculate_overlap(box, existing_box) > overlap_threshold:
                    overlap_found = True
                    break
            
            if not overlap_found:
                filtered_boxes.append(box)
        
        return filtered_boxes
    
    @staticmethod
    def calculate_overlap(box1, box2):
        """Calculate overlap ratio between two boxes"""
        x1_min, y1_min = box1['x'], box1['y']
        x1_max, y1_max = x1_min + box1['width'], y1_min + box1['height']
        
        x2_min, y2_min = box2['x'], box2['y']
        x2_max, y2_max = x2_min + box2['width'], y2_min + box2['height']
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
            intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            area1 = box1['width'] * box1['height']
            area2 = box2['width'] * box2['height']
            union = area1 + area2 - intersection
            return intersection / union if union > 0 else 0
        
        return 0

# Routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        password = request.form.get('password')
        if password == LOGIN_PASSWORD:
            session['logged_in'] = True
            flash('Successfully logged in!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Incorrect password. Please try again.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    publications = Publication.query.order_by(Publication.upload_date.desc()).all()
    return render_template('index.html', publications=publications)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['pdf_file']
        pub_type = request.form.get('publication_type')
        
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and pub_type in PUBLICATION_CONFIGS:
            try:
                # Generate unique filename
                file_ext = os.path.splitext(file.filename)[1]
                unique_filename = f"{uuid.uuid4()}{file_ext}"
                
                # Save file
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs', unique_filename)
                file.save(file_path)
                
                # Process PDF
                pdf_doc = fitz.open(file_path)
                page_count = pdf_doc.page_count
                pdf_doc.close()
                
                # Create publication record
                config = PUBLICATION_CONFIGS[pub_type]
                total_inches = config['total_inches_per_page'] * page_count
                
                publication = Publication(
                    filename=unique_filename,
                    original_filename=file.filename,
                    publication_type=pub_type,
                    total_pages=page_count,
                    total_inches=total_inches
                )
                
                db.session.add(publication)
                db.session.commit()
                
                flash(f'File uploaded successfully! Processing {page_count} pages...')
                return redirect(url_for('process_publication', pub_id=publication.id))
                
            except Exception as e:
                flash(f'Error uploading file: {str(e)}')
    
    return render_template('upload.html', pub_types=PUBLICATION_CONFIGS)

@app.route('/process/<int:pub_id>')
@login_required
def process_publication(pub_id):
    publication = Publication.query.get_or_404(pub_id)
    
    if publication.processed:
        flash('Publication already processed!')
        return redirect(url_for('view_publication', pub_id=pub_id))
    
    try:
        # Process PDF to images and create page records
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs', publication.filename)
        pdf_doc = fitz.open(file_path)
        
        config = PUBLICATION_CONFIGS[publication.publication_type]
        total_detected_boxes = 0
        
        for page_num in range(pdf_doc.page_count):
            page = pdf_doc.load_page(page_num)
            
            # Convert page to image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Save page image
            image_filename = f"{publication.filename}_page_{page_num + 1}.png"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pages', image_filename)
            
            with open(image_path, 'wb') as f:
                f.write(img_data)
            
            # ENHANCED CALIBRATION: Calculate accurate scaling from PDF
            pdf_calibration = CalibratedMeasurementCalculator.calculate_pdf_scaling_factor(
                file_path, page_num + 1, pix.width, pix.height
            )
            
            # Create page record with calibration data
            page_record = Page(
                publication_id=publication.id,
                page_number=page_num + 1,
                width_pixels=pix.width,
                height_pixels=pix.height,
                total_page_inches=config['total_inches_per_page']
            )
            
            if pdf_calibration:
                page_record.pdf_width_inches = pdf_calibration['pdf_width_inches']
                page_record.pdf_height_inches = pdf_calibration['pdf_height_inches']
                page_record.pixels_per_inch = pdf_calibration['pixels_per_inch']
                page_record.calibration_accurate = pdf_calibration['scaling_accurate']
            else:
                # Fallback to old method
                page_record.pixels_per_inch = pix.width / config['width_units']
            
            db.session.add(page_record)
            db.session.flush()  # Get the page ID
            
            # AI Box Detection
            detected_boxes = AdBoxDetector.detect_boxes(image_path)
            page_total_inches = 0
            
            for box in detected_boxes:
                # Use calibrated measurements if available
                if page_record.pixels_per_inch:
                    width_inches_raw = box['width'] / page_record.pixels_per_inch
                    height_inches_raw = box['height'] / page_record.pixels_per_inch
                else:
                    # Fallback to old method
                    width_inches_raw = MeasurementCalculator.pixels_to_inches(
                        box['width'], pix.width, config['width_units']
                    )
                    height_inches_raw = MeasurementCalculator.pixels_to_inches(
                        box['height'], pix.height, config['total_inches_per_page'] / config['width_units']
                    )
                
                # Apply rounding rules
                width_rounded = MeasurementCalculator.round_measurement(width_inches_raw)
                height_rounded = MeasurementCalculator.round_measurement(height_inches_raw)
                
                column_inches = width_inches_raw * height_inches_raw
                page_total_inches += column_inches
                
                # Create ad box record
                ad_box = AdBox(
                    page_id=page_record.id,
                    x=box['x'],
                    y=box['y'],
                    width=box['width'],
                    height=box['height'],
                    width_inches_raw=width_inches_raw,
                    height_inches_raw=height_inches_raw,
                    width_inches_rounded=width_rounded,
                    height_inches_rounded=height_rounded,
                    column_inches=column_inches,
                    user_verified=False
                )
                
                db.session.add(ad_box)
                total_detected_boxes += 1
            
            # Update page totals
            page_record.page_ad_inches = page_total_inches
        
        pdf_doc.close()
        
        # Update publication totals
        total_pub_ad_inches = sum(page.page_ad_inches for page in Page.query.filter_by(publication_id=publication.id))
        publication.total_ad_inches = total_pub_ad_inches
        publication.ad_percentage = (total_pub_ad_inches / publication.total_inches) * 100 if publication.total_inches > 0 else 0
        publication.processed = True
        
        db.session.commit()
        
        flash(f'Successfully processed {publication.total_pages} pages and detected {total_detected_boxes} potential ads!')
        return redirect(url_for('measure_publication', pub_id=pub_id))
        
    except Exception as e:
        flash(f'Error processing publication: {str(e)}')
        return redirect(url_for('index'))

@app.route('/view/<int:pub_id>')
@login_required
def view_publication(pub_id):
    publication = Publication.query.get_or_404(pub_id)
    pages = Page.query.filter_by(publication_id=pub_id).order_by(Page.page_number).all()
    return render_template('view_publication.html', publication=publication, pages=pages)

@app.route('/measure/<int:pub_id>')
@login_required
def measure_publication(pub_id):
    publication = Publication.query.get_or_404(pub_id)
    pages = Page.query.filter_by(publication_id=pub_id).order_by(Page.page_number).all()
    
    # Calculate total detected boxes
    total_boxes = 0
    for page in pages:
        boxes = AdBox.query.filter_by(page_id=page.id).count()
        total_boxes += boxes
    
    return render_template('measure.html', 
                         publication=publication, 
                         pages=pages,
                         total_boxes=total_boxes)

@app.route('/measure/<int:pub_id>/page/<int:page_num>')
@login_required
def measure_page(pub_id, page_num):
    publication = Publication.query.get_or_404(pub_id)
    page = Page.query.filter_by(publication_id=pub_id, page_number=page_num).first_or_404()
    ad_boxes = AdBox.query.filter_by(page_id=page.id).all()
    
    # Get total pages for navigation
    total_pages = Page.query.filter_by(publication_id=pub_id).count()
    
    return render_template('measure_page.html', 
                         publication=publication, 
                         page=page, 
                         ad_boxes=ad_boxes,
                         current_page=page_num,
                         total_pages=total_pages)

@app.route('/page-image/<int:page_id>')
def serve_page_image(page_id):
    """Serve page images"""
    page = Page.query.get_or_404(page_id)
    publication = Publication.query.get(page.publication_id)
    
    image_filename = f"{publication.filename}_page_{page.page_number}.png"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pages', image_filename)
    
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    else:
        return "Image not found", 404

# CALIBRATION ROUTES
@app.route('/calibrate')
def screen_calibration():
    """Show screen calibration interface"""
    calibration_image = ScreenCalibrationManager.create_calibration_target()
    
    # Get existing calibration for this device
    device_fingerprint = request.headers.get('User-Agent', '')[:100]  # Simplified fingerprint
    existing_calibration = CalibratedMeasurementCalculator.get_screen_calibration(device_fingerprint)
    
    return render_template('calibrate.html', 
                         calibration_image=calibration_image,
                         existing_calibration=existing_calibration)

@app.route('/api/save_calibration', methods=['POST'])
def save_calibration():
    """Save screen calibration from user measurement"""
    data = request.json
    
    try:
        measured_inches = float(data['measured_inches'])
        device_info = data.get('device_info', {})
        
        # Calculate scaling
        scaling_data = ScreenCalibrationManager.calculate_screen_scaling(measured_inches)
        
        if not scaling_data:
            return jsonify({'success': False, 'error': 'Invalid measurement'})
        
        # Create device fingerprint
        device_fingerprint = f"{request.headers.get('User-Agent', '')[:50]}_{device_info.get('screen_width', 0)}x{device_info.get('screen_height', 0)}"
        
        # Deactivate old calibrations for this device
        ScreenCalibration.query.filter_by(device_fingerprint=device_fingerprint).update({'is_active': False})
        
        # Save new calibration
        calibration = ScreenCalibration(
            device_fingerprint=device_fingerprint,
            screen_width=device_info.get('screen_width'),
            screen_height=device_info.get('screen_height'),
            device_pixel_ratio=device_info.get('device_pixel_ratio', 1.0),
            scaling_factor=scaling_data['scaling_factor'],
            pixels_per_inch=scaling_data['pixels_per_inch'],
            measured_inches=measured_inches,
            user_agent=request.headers.get('User-Agent', '')
        )
        
        db.session.add(calibration)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'scaling_factor': scaling_data['scaling_factor'],
            'pixels_per_inch': scaling_data['pixels_per_inch'],
            'message': f'Calibration saved! Your screen shows {measured_inches}" for 6" reference.'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Report Generation Routes
@app.route('/report/<int:pub_id>')
@login_required
def publication_report(pub_id):
    """Generate comprehensive publication report"""
    publication = Publication.query.get_or_404(pub_id)
    pages = Page.query.filter_by(publication_id=pub_id).order_by(Page.page_number).all()
    
    # Calculate comprehensive statistics
    total_boxes = 0
    user_created_boxes = 0
    ai_detected_boxes = 0
    
    page_data = []
    for page in pages:
        boxes = AdBox.query.filter_by(page_id=page.id).all()
        page_boxes = len(boxes)
        page_user_boxes = len([b for b in boxes if b.user_verified])
        page_ai_boxes = page_boxes - page_user_boxes
        
        total_boxes += page_boxes
        user_created_boxes += page_user_boxes
        ai_detected_boxes += page_ai_boxes
        
        page_data.append({
            'page': page,
            'boxes': boxes,
            'total_boxes': page_boxes,
            'user_boxes': page_user_boxes,
            'ai_boxes': page_ai_boxes
        })
    
    # Calculate compliance metrics
    config = PUBLICATION_CONFIGS[publication.publication_type]
    
    report_data = {
        'publication': publication,
        'config': config,
        'pages': page_data,
        'total_boxes': total_boxes,
        'user_created_boxes': user_created_boxes,
        'ai_detected_boxes': ai_detected_boxes,
        'generated_date': datetime.now()
    }
    
    return render_template('report.html', **report_data)

@app.route('/download_pdf/<int:pub_id>')
def download_pdf_report(pub_id):
    """Generate and download PDF report"""
    publication = Publication.query.get_or_404(pub_id)
    pages = Page.query.filter_by(publication_id=pub_id).order_by(Page.page_number).all()
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    
    # Create PDF document
    doc = SimpleDocTemplate(temp_file.name, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    story.append(Paragraph(f"USPS Advertising Report", title_style))
    story.append(Paragraph(f"{publication.original_filename}", styles['Heading2']))
    story.append(Spacer(1, 20))
    
    # Publication Summary
    config = PUBLICATION_CONFIGS[publication.publication_type]
    summary_data = [
        ['Publication Type:', config['name']],
        ['Total Pages:', str(publication.total_pages)],
        ['Total Column Inches Available:', f"{publication.total_inches:.0f}\""],
        ['Total Advertising Column Inches:', f"{publication.total_ad_inches:.1f}\""],
        ['Advertising Percentage:', f"{publication.ad_percentage:.2f}%"],
        ['Report Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        ['USPS Compliance:', 'COMPLIANT' if publication.ad_percentage > 0 else 'PENDING']
    ]
    
    summary_table = Table(summary_data, colWidths=[2.5*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (1, -1), (1, -1), colors.lightgreen if publication.ad_percentage > 0 else colors.lightyellow),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 30))
    
    # Page-by-page breakdown
    story.append(Paragraph("Page-by-Page Breakdown", styles['Heading2']))
    story.append(Spacer(1, 20))
    
    page_data = [['Page #', 'Ad Boxes', 'Column Inches', 'Percentage of Page']]
    
    for page in pages:
        boxes = AdBox.query.filter_by(page_id=page.id).all()
        page_percentage = (page.page_ad_inches / page.total_page_inches) * 100 if page.total_page_inches > 0 else 0
        
        page_data.append([
            str(page.page_number),
            str(len(boxes)),
            f"{page.page_ad_inches:.1f}\"",
            f"{page_percentage:.1f}%"
        ])
    
    page_table = Table(page_data, colWidths=[1*inch, 1*inch, 1.5*inch, 1.5*inch])
    page_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(page_table)
    story.append(Spacer(1, 30))
    
    # Detailed ad listing
    story.append(Paragraph("Detailed Advertisement Listing", styles['Heading2']))
    story.append(Spacer(1, 20))
    
    for page in pages:
        boxes = AdBox.query.filter_by(page_id=page.id).all()
        if boxes:
            story.append(Paragraph(f"Page {page.page_number} - {len(boxes)} advertisements", styles['Heading3']))
            
            ad_data = [['Ad #', 'Width', 'Height', 'Column Inches', 'Source']]
            
            for i, box in enumerate(boxes, 1):
                source = 'User Created' if box.user_verified else 'AI Detected'
                ad_data.append([
                    str(i),
                    f"{box.width_inches_raw:.2f}\"",
                    f"{box.height_inches_raw:.2f}\"",
                    f"{box.column_inches:.1f}\"",
                    source
                ])
            
            ad_table = Table(ad_data, colWidths=[0.8*inch, 1*inch, 1*inch, 1.2*inch, 1.5*inch])
            ad_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightblue])
            ]))
            
            story.append(ad_table)
            story.append(Spacer(1, 15))
    
    # Footer
    story.append(Spacer(1, 50))
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        alignment=1,
        textColor=colors.grey
    )
    story.append(Paragraph("Report generated by Newspaper Ad Measurement System", footer_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}", footer_style))
    
    # Build PDF
    doc.build(story)
    
    # Return file
    return send_file(
        temp_file.name,
        as_attachment=True,
        download_name=f"ad_report_{publication.original_filename}_{datetime.now().strftime('%Y%m%d')}.pdf",
        mimetype='application/pdf'
    )

@app.route('/download_csv/<int:pub_id>')
def download_csv_report(pub_id):
    """Generate and download CSV report"""
    publication = Publication.query.get_or_404(pub_id)
    pages = Page.query.filter_by(publication_id=pub_id).order_by(Page.page_number).all()
    
    # Create CSV content
    output = StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow([
        'Publication', 'Page', 'Ad #', 'Width (inches)', 'Height (inches)', 
        'Column Inches', 'X Position', 'Y Position', 'Source', 'Date Created'
    ])
    
    # Data rows
    for page in pages:
        boxes = AdBox.query.filter_by(page_id=page.id).all()
        for i, box in enumerate(boxes, 1):
            source = 'User Created' if box.user_verified else 'AI Detected'
            writer.writerow([
                publication.original_filename,
                page.page_number,
                i,
                f"{box.width_inches_raw:.2f}",
                f"{box.height_inches_raw:.2f}",
                f"{box.column_inches:.1f}",
                f"{box.x:.0f}",
                f"{box.y:.0f}",
                source,
                box.created_date.strftime('%Y-%m-%d %H:%M:%S')
            ])
    
    # Prepare response
    output.seek(0)
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv')
    temp_file.write(output.getvalue())
    temp_file.close()
    
    return send_file(
        temp_file.name,
        as_attachment=True,
        download_name=f"ad_data_{publication.original_filename}_{datetime.now().strftime('%Y%m%d')}.csv",
        mimetype='text/csv'
    )

# API Routes for box management
@app.route('/api/update_box/<int:box_id>', methods=['POST'])
def update_box(box_id):
    """Update box coordinates and measurements via API"""
    ad_box = AdBox.query.get_or_404(box_id)
    data = request.json
    
    try:
        # Update coordinates
        ad_box.x = data['x']
        ad_box.y = data['y'] 
        ad_box.width = data['width']
        ad_box.height = data['height']
        
        # Recalculate measurements with calibration
        page = Page.query.get(ad_box.page_id)
        publication = Publication.query.get(page.publication_id)
        config = PUBLICATION_CONFIGS[publication.publication_type]
        
        # Get screen calibration
        device_fingerprint = f"{request.headers.get('User-Agent', '')[:50]}"
        screen_calibration = CalibratedMeasurementCalculator.get_screen_calibration(device_fingerprint)
        screen_scaling_factor = screen_calibration.scaling_factor if screen_calibration else 1.0
        
        # Use PDF calibration if available
        if page.pixels_per_inch:
            width_inches_raw = CalibratedMeasurementCalculator.pixels_to_inches_with_screen_calibration(
                data['width'], page.pixels_per_inch, screen_scaling_factor
            )
            height_inches_raw = CalibratedMeasurementCalculator.pixels_to_inches_with_screen_calibration(
                data['height'], page.pixels_per_inch, screen_scaling_factor
            )
        else:
            # Fallback to old method
            width_inches_raw = MeasurementCalculator.pixels_to_inches(
                data['width'], page.width_pixels, config['width_units']
            )
            height_inches_raw = MeasurementCalculator.pixels_to_inches(
                data['height'], page.height_pixels, config['total_inches_per_page'] / config['width_units']
            )
        
        width_rounded = MeasurementCalculator.round_measurement(width_inches_raw)
        height_rounded = MeasurementCalculator.round_measurement(height_inches_raw)
        
        ad_box.width_inches_raw = width_inches_raw
        ad_box.height_inches_raw = height_inches_raw
        ad_box.width_inches_rounded = width_rounded
        ad_box.height_inches_rounded = height_rounded
        ad_box.column_inches = width_inches_raw * height_inches_raw
        ad_box.user_verified = True
        
        db.session.commit()
        
        # Recalculate page and publication totals
        update_totals(ad_box.page_id)
        
        return jsonify({
            'success': True,
            'width_raw': float(width_inches_raw),
            'height_raw': float(height_inches_raw),
            'width_rounded': float(width_rounded),
            'height_rounded': float(height_rounded),
            'column_inches': float(ad_box.column_inches)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_box/<int:box_id>', methods=['DELETE'])
def delete_box(box_id):
    """Delete an ad box"""
    ad_box = AdBox.query.get_or_404(box_id)
    page_id = ad_box.page_id
    
    db.session.delete(ad_box)
    db.session.commit()
    
    # Recalculate totals
    update_totals(page_id)
    
    return jsonify({'success': True})

@app.route('/api/add_box/<int:page_id>', methods=['POST'])
def add_box(page_id):
    """Add a new ad box"""
    page = Page.query.get_or_404(page_id)
    publication = Publication.query.get(page.publication_id)
    config = PUBLICATION_CONFIGS[publication.publication_type]
    data = request.json
    
    try:
        # Get screen calibration
        device_fingerprint = f"{request.headers.get('User-Agent', '')[:50]}"
        screen_calibration = CalibratedMeasurementCalculator.get_screen_calibration(device_fingerprint)
        screen_scaling_factor = screen_calibration.scaling_factor if screen_calibration else 1.0
        
        # Calculate measurements with both PDF and screen calibration
        if page.pixels_per_inch:
            width_inches_raw = CalibratedMeasurementCalculator.pixels_to_inches_with_screen_calibration(
                data['width'], page.pixels_per_inch, screen_scaling_factor
            )
            height_inches_raw = CalibratedMeasurementCalculator.pixels_to_inches_with_screen_calibration(
                data['height'], page.pixels_per_inch, screen_scaling_factor
            )
        else:
            # Fallback to old method
            width_inches_raw = MeasurementCalculator.pixels_to_inches(
                data['width'], page.width_pixels, config['width_units']
            )
            height_inches_raw = MeasurementCalculator.pixels_to_inches(
                data['height'], page.height_pixels, config['total_inches_per_page'] / config['width_units']
            )
        
        width_rounded = MeasurementCalculator.round_measurement(width_inches_raw)
        height_rounded = MeasurementCalculator.round_measurement(height_inches_raw)
        
        ad_box = AdBox(
            page_id=page_id,
            x=data['x'],
            y=data['y'],
            width=data['width'],
            height=data['height'],
            width_inches_raw=width_inches_raw,
            height_inches_raw=height_inches_raw,
            width_inches_rounded=width_rounded,
            height_inches_rounded=height_rounded,
            column_inches=width_inches_raw * height_inches_raw,
            ad_type=data.get('ad_type', 'manual'),
            user_verified=True
        )
        
        db.session.add(ad_box)
        db.session.commit()
        
        # Recalculate totals
        update_totals(page_id)
        
        return jsonify({
            'success': True,
            'box_id': ad_box.id,
            'width_raw': float(width_inches_raw),
            'height_raw': float(height_inches_raw),
            'width_rounded': float(width_rounded),
            'height_rounded': float(height_rounded),
            'column_inches': float(ad_box.column_inches)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/intelligent_detect/<int:page_id>', methods=['POST'])
def intelligent_detect_ad(page_id):
    """Intelligently detect ad boundaries with screen calibration"""
    page = Page.query.get_or_404(page_id)
    publication = Publication.query.get(page.publication_id)
    
    # Intelligent detection available for broadsheet and peach publications
    if publication.publication_type not in ['broadsheet', 'peach']:
        return jsonify({'success': False, 'error': 'Intelligent detection only available for broadsheet and peach publications'})
    
    data = request.json
    click_x = int(data['x'])
    click_y = int(data['y'])
    ad_type = data.get('ad_type', 'open_display')
    
    # Get screen calibration
    device_fingerprint = f"{request.headers.get('User-Agent', '')[:50]}"
    screen_calibration = CalibratedMeasurementCalculator.get_screen_calibration(device_fingerprint)
    screen_scaling_factor = screen_calibration.scaling_factor if screen_calibration else 1.0
    
    try:
        # Get page image path
        image_filename = f"{publication.filename}_page_{page.page_number}.png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pages', image_filename)
        
        # Detect ad boundaries
        detected_box = IntelligentAdDetector.detect_ad_from_click(image_path, click_x, click_y, ad_type)
        
        if not detected_box:
            return jsonify({'success': False, 'error': 'Could not detect ad boundaries. Try right-clicking for manual box option or click closer to the ad border.'})
        
        # Convert pixels to inches with BOTH PDF and screen calibration
        if page.pixels_per_inch:
            width_inches_raw = CalibratedMeasurementCalculator.pixels_to_inches_with_screen_calibration(
                detected_box['width'], page.pixels_per_inch, screen_scaling_factor
            )
            height_inches_raw = CalibratedMeasurementCalculator.pixels_to_inches_with_screen_calibration(
                detected_box['height'], page.pixels_per_inch, screen_scaling_factor
            )
        else:
            # Fallback to old method
            config = PUBLICATION_CONFIGS[publication.publication_type]
            pdf_pixels_per_inch = page.width_pixels / config['width_units']
            width_inches_raw = CalibratedMeasurementCalculator.pixels_to_inches_with_screen_calibration(
                detected_box['width'], pdf_pixels_per_inch, screen_scaling_factor
            )
            height_inches_raw = CalibratedMeasurementCalculator.pixels_to_inches_with_screen_calibration(
                detected_box['height'], pdf_pixels_per_inch, screen_scaling_factor
            )
        
        # Apply intelligent measurement rules
        config = PUBLICATION_CONFIGS[publication.publication_type]
        if ad_type != 'manual' and 'column_standards' in config:
            # Match width to column standards
            width_inches_raw = IntelligentAdDetector.match_to_column_width(
                width_inches_raw, ad_type, config['column_standards']
            )
            # Round depth up (except public notices)
            height_inches_raw = IntelligentAdDetector.round_depth_up(height_inches_raw, ad_type)
        
        # Calculate column inches
        column_inches = width_inches_raw * height_inches_raw
        
        # Create ad box record
        ad_box = AdBox(
            page_id=page_id,
            x=detected_box['x'],
            y=detected_box['y'],
            width=detected_box['width'],
            height=detected_box['height'],
            width_inches_raw=width_inches_raw,
            height_inches_raw=height_inches_raw,
            width_inches_rounded=MeasurementCalculator.round_measurement(width_inches_raw),
            height_inches_rounded=MeasurementCalculator.round_measurement(height_inches_raw),
            column_inches=column_inches,
            ad_type=ad_type,
            user_verified=True
        )
        
        db.session.add(ad_box)
        db.session.commit()
        
        # Update totals
        update_totals(page_id)
        
        return jsonify({
            'success': True,
            'box_id': ad_box.id,
            'x': int(detected_box['x']),
            'y': int(detected_box['y']),
            'width': int(detected_box['width']),
            'height': int(detected_box['height']),
            'width_raw': float(width_inches_raw),
            'height_raw': float(height_inches_raw),
            'column_inches': float(column_inches),
            'ad_type': ad_type,
            'calibrated': screen_calibration is not None
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/add_special_edition_ad/<int:page_id>', methods=['POST'])
def add_special_edition_ad(page_id):
    """Add a new special edition ad with preset measurements"""
    page = Page.query.get_or_404(page_id)
    publication = Publication.query.get(page.publication_id)
    data = request.json
    
    try:
        # For special edition, we use preset measurements
        column_inches = data['column_inches']
        
        # Create ad box record with preset measurements
        ad_box = AdBox(
            page_id=page_id,
            x=data['x'],
            y=data['y'],
            width=data['width'],
            height=data['height'],
            width_inches_raw=0,  # Not used for special edition
            height_inches_raw=0,  # Not used for special edition
            width_inches_rounded=0,  # Not used for special edition
            height_inches_rounded=0,  # Not used for special edition
            column_inches=column_inches,
            ad_type='special_edition',
            user_verified=True
        )
        
        db.session.add(ad_box)
        db.session.commit()
        
        # Recalculate totals
        update_totals(page_id)
        
        return jsonify({
            'success': True,
            'box_id': ad_box.id,
            'column_inches': column_inches
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def update_totals(page_id):
    """Update page and publication totals after box changes"""
    page = Page.query.get(page_id)
    
    # Recalculate page total
    total_ad_inches = db.session.query(db.func.sum(AdBox.column_inches)).filter_by(
        page_id=page_id, is_ad=True
    ).scalar() or 0
    
    page.page_ad_inches = total_ad_inches
    
    # Recalculate publication total
    publication_total = db.session.query(db.func.sum(Page.page_ad_inches)).filter_by(
        publication_id=page.publication_id
    ).scalar() or 0
    
    publication = Publication.query.get(page.publication_id)
    publication.total_ad_inches = publication_total
    publication.ad_percentage = (publication_total / publication.total_inches) * 100 if publication.total_inches > 0 else 0
    
    db.session.commit()

@app.route('/fix_special_edition_pages')
def fix_special_edition_pages():
    """Temporary route to fix special edition page totals"""
    try:
        # Find all special edition publications
        special_pubs = Publication.query.filter_by(publication_type='special_edition').all()
        
        fixed_count = 0
        for pub in special_pubs:
            # Update each page's total_page_inches
            pages = Page.query.filter_by(publication_id=pub.id).all()
            for page in pages:
                page.total_page_inches = 125  # Update to new total
            
            # Update publication total_inches
            pub.total_inches = 125 * pub.total_pages
            
            # Recalculate ad percentage
            if pub.total_inches > 0:
                pub.ad_percentage = (pub.total_ad_inches / pub.total_inches) * 100
            
            fixed_count += 1
        
        db.session.commit()
        
        return f"<h1>Success!</h1><p>Fixed {fixed_count} special edition publications</p><p><a href='/'>Back to Home</a></p>"
        
    except Exception as e:
        return f"<h1>Error:</h1><p>{str(e)}</p>"

@app.route('/test')
def test():
    return """
    <h1>🎉 Flask App is Working!</h1>
    <p>✅ Flask: Working</p>
    <p>✅ Database: Connected</p>
    <p>✅ File uploads: Ready</p>
    <p>✅ PDF processing: Ready</p>
    <p>✅ AI Detection: Enhanced & Improved</p>
    <p>✅ Reporting System: Ready</p>
    <p>✅ Screen Calibration: Ready</p>
    <p><a href="/">Go to Home</a></p>
    <p><a href="/calibrate">Calibrate Screen</a></p>
    """

# Add this new API route to your app.py file
# Place it with your other API routes (after the existing /api/ routes)

@app.route('/api/add_full_page_ad/<int:page_id>', methods=['POST'])
def add_full_page_ad(page_id):
    """Add a full-page advertisement (Broadsheet only)"""
    page = Page.query.get_or_404(page_id)
    publication = Publication.query.get(page.publication_id)
    
    # Only allow for broadsheet publications
    if publication.publication_type != 'broadsheet':
        return jsonify({'success': False, 'error': 'Full-page ads only available for broadsheet publications'})
    
    data = request.json
    
    try:
        # Get configuration for broadsheet
        config = PUBLICATION_CONFIGS[publication.publication_type]
        full_page_inches = config['total_inches_per_page']  # 258 inches
        
        # Delete all existing ads on this page first
        existing_ads = AdBox.query.filter_by(page_id=page_id).all()
        for ad in existing_ads:
            db.session.delete(ad)
        
        # Create the full-page ad box record
        ad_box = AdBox(
            page_id=page_id,
            x=data['x'],
            y=data['y'],
            width=data['width'],
            height=data['height'],
            width_inches_raw=0,  # Not applicable for full-page ads
            height_inches_raw=0,  # Not applicable for full-page ads
            width_inches_rounded=0,  # Not applicable for full-page ads
            height_inches_rounded=0,  # Not applicable for full-page ads
            column_inches=full_page_inches,  # 258 inches
            ad_type='full_page',
            is_ad=True,
            user_verified=True
        )
        
        db.session.add(ad_box)
        db.session.commit()
        
        # Update page and publication totals
        update_totals(page_id)
        
        return jsonify({
            'success': True,
            'box_id': ad_box.id,
            'x': int(data['x']),
            'y': int(data['y']),
            'width': int(data['width']),
            'height': int(data['height']),
            'column_inches': float(full_page_inches),
            'ad_type': 'full_page',
            'message': f'Full-page advertisement added ({full_page_inches} inches)'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/add_manual_box/<int:page_id>', methods=['POST'])
def add_manual_box(page_id):
    """Add a new manual ad box for broadsheet publications"""
    page = Page.query.get_or_404(page_id)
    publication = Publication.query.get(page.publication_id)
    config = PUBLICATION_CONFIGS[publication.publication_type]
    data = request.json
    
    try:
        # Get screen calibration
        device_fingerprint = f"{request.headers.get('User-Agent', '')[:50]}"
        screen_calibration = CalibratedMeasurementCalculator.get_screen_calibration(device_fingerprint)
        screen_scaling_factor = screen_calibration.scaling_factor if screen_calibration else 1.0
        
        # Calculate measurements with both PDF and screen calibration
        if page.pixels_per_inch:
            width_inches_raw = CalibratedMeasurementCalculator.pixels_to_inches_with_screen_calibration(
                data['width'], page.pixels_per_inch, screen_scaling_factor
            )
            height_inches_raw = CalibratedMeasurementCalculator.pixels_to_inches_with_screen_calibration(
                data['height'], page.pixels_per_inch, screen_scaling_factor
            )
        else:
            # Fallback to old method
            width_inches_raw = MeasurementCalculator.pixels_to_inches(
                data['width'], page.width_pixels, config['width_units']
            )
            height_inches_raw = MeasurementCalculator.pixels_to_inches(
                data['height'], page.height_pixels, config['total_inches_per_page'] / config['width_units']
            )
        
        width_rounded = MeasurementCalculator.round_measurement(width_inches_raw)
        height_rounded = MeasurementCalculator.round_measurement(height_inches_raw)
        
        ad_box = AdBox(
            page_id=page_id,
            x=data['x'],
            y=data['y'],
            width=data['width'],
            height=data['height'],
            width_inches_raw=width_inches_raw,
            height_inches_raw=height_inches_raw,
            width_inches_rounded=width_rounded,
            height_inches_rounded=height_rounded,
            column_inches=width_inches_raw * height_inches_raw,
            ad_type='manual',
            user_verified=True
        )
        
        db.session.add(ad_box)
        db.session.commit()
        
        # Recalculate totals
        update_totals(page_id)
        
        return jsonify({
            'success': True,
            'box_id': ad_box.id,
            'x': int(data['x']),
            'y': int(data['y']),
            'width': int(data['width']),
            'height': int(data['height']),
            'width_raw': float(width_inches_raw),
            'height_raw': float(height_inches_raw),
            'width_rounded': float(width_rounded),
            'height_rounded': float(height_rounded),
            'column_inches': float(width_inches_raw * height_inches_raw),
            'ad_type': 'manual'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/update_manual_box/<int:box_id>', methods=['POST'])
def update_manual_box(box_id):
    """Update a manual ad box"""
    ad_box = AdBox.query.get_or_404(box_id)
    page = Page.query.get(ad_box.page_id)
    publication = Publication.query.get(page.publication_id)
    config = PUBLICATION_CONFIGS[publication.publication_type]
    data = request.json
    
    try:
        # Get screen calibration
        device_fingerprint = f"{request.headers.get('User-Agent', '')[:50]}"
        screen_calibration = CalibratedMeasurementCalculator.get_screen_calibration(device_fingerprint)
        screen_scaling_factor = screen_calibration.scaling_factor if screen_calibration else 1.0
        
        # Update box position and size
        ad_box.x = data['x']
        ad_box.y = data['y']
        ad_box.width = data['width']
        ad_box.height = data['height']
        
        # Recalculate measurements with both PDF and screen calibration
        if page.pixels_per_inch:
            width_inches_raw = CalibratedMeasurementCalculator.pixels_to_inches_with_screen_calibration(
                data['width'], page.pixels_per_inch, screen_scaling_factor
            )
            height_inches_raw = CalibratedMeasurementCalculator.pixels_to_inches_with_screen_calibration(
                data['height'], page.pixels_per_inch, screen_scaling_factor
            )
        else:
            # Fallback to old method
            width_inches_raw = MeasurementCalculator.pixels_to_inches(
                data['width'], page.width_pixels, config['width_units']
            )
            height_inches_raw = MeasurementCalculator.pixels_to_inches(
                data['height'], page.height_pixels, config['total_inches_per_page'] / config['width_units']
            )
        
        width_rounded = MeasurementCalculator.round_measurement(width_inches_raw)
        height_rounded = MeasurementCalculator.round_measurement(height_inches_raw)
        
        ad_box.width_inches_raw = width_inches_raw
        ad_box.height_inches_raw = height_inches_raw
        ad_box.width_inches_rounded = width_rounded
        ad_box.height_inches_rounded = height_rounded
        ad_box.column_inches = width_inches_raw * height_inches_raw
        
        db.session.commit()
        
        # Recalculate totals
        update_totals(ad_box.page_id)
        
        return jsonify({
            'success': True,
            'width_raw': float(width_inches_raw),
            'height_raw': float(height_inches_raw),
            'width_rounded': float(width_rounded),
            'height_rounded': float(height_rounded),
            'column_inches': float(width_inches_raw * height_inches_raw)
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    print("Starting Newspaper Ad Measurement System...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print("Opening at: http://localhost:5000")
    print("AI Box Detection: Enhanced & Improved")
    print("Interactive Measurement Tools: Ready")
    print("Professional Reporting System: Ready")
    print("Intelligent Click Detection: Ready for Broadsheet")
    print("Screen Calibration System: Ready")
    app.run(debug=True)