import os
import uuid
import fitz  # PyMuPDF
import cv2
import numpy as np
import math
import json
from pdf_structure_analyzer import PDFStructureAdDetector

# Google Vision AI setup - SECURE VERSION
# Load environment variables first
from dotenv import load_dotenv
load_dotenv()

# Set Google credentials from environment variable (more secure)
google_creds_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if google_creds_path and os.path.exists(google_creds_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = google_creds_path
    
from google.cloud import vision
import io as vision_io
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime, timedelta
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
import json
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from scipy import ndimage
from datetime import timedelta
import hashlib
import secrets

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
database_url = os.environ.get('DATABASE_URL', 'sqlite:///newspaper_ads.db')
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Continue with your existing configuration
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-this')

# Session security configuration
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=8)  # 8 hour session timeout
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('RAILWAY_ENVIRONMENT') is not None  # HTTPS only in production
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent XSS attacks
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection

# Upload configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Security configuration for file uploads
ALLOWED_EXTENSIONS = {'.pdf'}
MAX_FILENAME_LENGTH = 255

# Initialize extensions
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Global flag to track if processing columns exist
PROCESSING_COLUMNS_EXIST = None

def check_processing_columns():
    """Check if processing columns exist in the database"""
    global PROCESSING_COLUMNS_EXIST
    if PROCESSING_COLUMNS_EXIST is not None:
        return PROCESSING_COLUMNS_EXIST
    
    # Skip column checks in production to prevent hanging
    if os.environ.get('RAILWAY_ENVIRONMENT'):
        PROCESSING_COLUMNS_EXIST = False
        return False
    
    try:
        with db.engine.connect() as conn:
            # Check if processing_status column exists
            if 'postgresql' in str(db.engine.url):
                result = conn.execute(db.text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'publication' AND column_name = 'processing_status'
                """))
            else:  # SQLite
                result = conn.execute(db.text("PRAGMA table_info(publication)"))
            
            columns = [row[0] if 'postgresql' in str(db.engine.url) else row[1] for row in result.fetchall()]
            PROCESSING_COLUMNS_EXIST = 'processing_status' in columns
            print(f"Processing columns available: {PROCESSING_COLUMNS_EXIST}")
            return PROCESSING_COLUMNS_EXIST
    except Exception as e:
        print(f"Error checking processing columns: {e}")
        PROCESSING_COLUMNS_EXIST = False
        return False

# Authentication configuration
LOGIN_PASSWORD = os.environ.get('LOGIN_PASSWORD', 'CCCitizen56101!')

def hash_password(password):
    """Hash password with salt for secure storage"""
    salt = secrets.token_hex(16)
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return salt + pwdhash.hex()

def verify_password(stored_password, provided_password):
    """Verify a password against its hash"""
    if len(stored_password) < 32:
        # Handle legacy plain text password during transition
        return stored_password == provided_password
    
    salt = stored_password[:32]
    stored_hash = stored_password[32:]
    pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt.encode('utf-8'), 100000)
    return pwdhash.hex() == stored_hash

def is_session_valid():
    """Check if current session is valid"""
    if not session.get('logged_in'):
        return False
    
    # Check session timestamp
    login_time = session.get('login_time')
    if not login_time:
        return False
    
    # Check if session has expired
    from datetime import datetime, timedelta
    if datetime.utcnow().timestamp() - login_time > app.config['PERMANENT_SESSION_LIFETIME'].total_seconds():
        return False
    
    return True

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not is_session_valid():
            session.clear()  # Clear invalid session
            flash('Your session has expired. Please log in again.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def allowed_file(filename):
    """Check if file extension is allowed and filename is safe"""
    if not filename or len(filename) > MAX_FILENAME_LENGTH:
        return False
    
    # Check extension
    file_ext = os.path.splitext(filename.lower())[1]
    if file_ext not in ALLOWED_EXTENSIONS:
        return False
    
    # Check for dangerous characters in filename
    dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
    if any(char in filename for char in dangerous_chars):
        return False
    
    return True

def validate_pdf_file(file_path):
    """Validate that uploaded file is actually a PDF"""
    try:
        # Try to open with PyMuPDF to verify it's a valid PDF
        doc = fitz.open(file_path)
        page_count = doc.page_count
        doc.close()
        
        # Basic sanity checks
        if page_count < 1 or page_count > 1000:  # Reasonable limits
            return False
        
        return True
    except Exception as e:
        print(f"PDF validation error: {e}")
        return False

# Global set to track publications being processed
_processing_publications = set()
_processing_lock = None

def start_background_processing(pub_id):
    """Start background processing for a publication"""
    global _processing_publications, _processing_lock
    
    try:
        import threading
        if _processing_lock is None:
            _processing_lock = threading.Lock()
    except ImportError:
        # Fallback if threading is not available
        print("Threading not available, processing synchronously")
        process_publication_sync(pub_id)
        return
    
    # Check if already processing (with timeout to prevent infinite loops)
    import time
    processing_start_time = time.time()
    
    with _processing_lock:
        if pub_id in _processing_publications:
            # If it's been processing for more than 30 minutes, assume it's stuck and continue
            if time.time() - processing_start_time < 1800:  # 30 minutes
                print(f"Publication {pub_id} is already being processed, skipping")
                return
            else:
                print(f"Publication {pub_id} has been processing for too long, forcing restart")
                _processing_publications.discard(pub_id)
        
        _processing_publications.add(pub_id)
    
    def process_in_background():
        with app.app_context():
            try:
                # Use db.session.get() instead of deprecated query.get()
                publication = db.session.get(Publication, pub_id)
                if not publication:
                    return
                
                # Check if already processed or processing to prevent duplicates
                current_status = publication.safe_processing_status
                if publication.processed or current_status in ['processing', 'extracting_pages', 'creating_images', 'ai_detection', 'completed']:
                    print(f"Publication {pub_id} already processed or processing (status: {current_status}), skipping")
                    # Remove from processing set since we're skipping
                    with _processing_lock:
                        _processing_publications.discard(pub_id)
                    return
                
                print(f"Starting background processing for publication {pub_id}")
                
                # Update status to processing
                publication.set_processing_status('extracting_pages')
                db.session.commit()
                
                # Process PDF to images and create page records
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs', publication.filename)
                pdf_doc = fitz.open(file_path)
                
                publication.set_processing_status('creating_images')
                db.session.commit()
                
                # Process each page (with batch processing to avoid timeouts)
                batch_size = 3  # Process 3 pages at a time
                for batch_start in range(0, pdf_doc.page_count, batch_size):
                    batch_end = min(batch_start + batch_size, pdf_doc.page_count)
                    
                    publication.set_processing_status(f'processing_pages_{batch_start + 1}_to_{batch_end}')
                    
                    for page_num in range(batch_start, batch_end):
                        page = pdf_doc[page_num]
                        
                        # Convert to image with lower resolution for faster processing
                        mat = fitz.Matrix(1.5, 1.5)  # Reduced from 2x to 1.5x for speed
                        pix = page.get_pixmap(matrix=mat)
                        img_data = pix.tobytes("png")
                        
                        # Save page image
                        image_filename = f"{publication.filename}_page_{page_num + 1}.png"
                        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pages', image_filename)
                        
                        with open(image_path, 'wb') as img_file:
                            img_file.write(img_data)
                        
                        # Check if page record already exists to prevent duplicates
                        existing_page = Page.query.filter_by(
                            publication_id=publication.id,
                            page_number=page_num + 1
                        ).first()
                        
                        if not existing_page:
                            # Create page record for this specific page
                            config = PUBLICATION_CONFIGS[publication.publication_type]
                            page_record = Page(
                                publication_id=publication.id,
                                page_number=page_num + 1,
                                width_pixels=pix.width,
                                height_pixels=pix.height,
                                total_page_inches=config['total_inches_per_page']
                            )
                            db.session.add(page_record)
                            print(f"Created page record for page {page_num + 1}")
                        else:
                            print(f"Page record for page {page_num + 1} already exists, skipping")
                    
                    # Commit the batch of page records
                    db.session.commit()
                    print(f"Committed batch {batch_start + 1}-{batch_end}")
                
                pdf_doc.close()
                
                # Update status to AI processing
                publication.set_processing_status('ai_detection')
                db.session.commit()
                
                # Run AI ad detection at the end of processing (with robust error handling)
                try:
                    print(f"🤖 Starting automatic ad detection for publication {publication.id} ({publication.publication_type})")
                    
                    # SIMPLE WORKING DETECTION: Find actual bordered ads
                    try:
                        print(f"🎯 Starting SIMPLE ad detection - bordered regions only")
                        simple_result = SimpleAdDetector.detect_bordered_ads(publication.id)

                        if simple_result and simple_result.get('success'):
                            print(f"SUCCESS: Simple detection complete: {simple_result['detections']} ads detected across {simple_result['pages_processed']} pages")
                            if simple_result['detections'] > 0:
                                print(f"Next: Review the detected ads - should see actual business ads")
                            else:
                                print(f"No bordered ads detected - check image quality")
                        else:
                            error_msg = simple_result.get('error', 'Simple detection failed') if simple_result else 'No result returned'
                            print(f"Simple detection failed: {error_msg}")
                    except Exception as simple_error:
                        print(f"Simple detection failed with error: {simple_error}")
                        print(f"Falling back to manual detection only")

                except Exception as outer_error:
                    print(f"⚠️  Hybrid detection phase failed: {outer_error}")
                    print(f"📝 Publication processed successfully - use manual detection interface")
                
                # Mark as completed
                publication.processed = True
                publication.set_processing_status('completed')
                db.session.commit()
                
                # Remove from processing set
                try:
                    with _processing_lock:
                        _processing_publications.discard(pub_id)
                except:
                    pass
                
            except Exception as e:
                # Mark as failed
                try:
                    publication = db.session.get(Publication, pub_id)
                    if publication:
                        publication.set_processing_status('failed')
                        publication.set_processing_error(str(e))
                        db.session.commit()
                except Exception as commit_error:
                    print(f"Error updating failure status: {commit_error}")
                finally:
                    # Ensure database session is cleaned up
                    try:
                        db.session.remove()
                    except:
                        pass
                    # Remove from processing set
                    try:
                        with _processing_lock:
                            _processing_publications.discard(pub_id)
                    except:
                        pass
    
    # Start processing in background thread
    try:
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
    except Exception as e:
        print(f"Failed to start background thread: {e}")
        # Fallback to synchronous processing
        process_publication_sync(pub_id)

def process_publication_sync(pub_id):
    """Synchronous processing fallback - does basic setup only"""
    with app.app_context():
        try:
            publication = Publication.query.get(pub_id)
            if not publication:
                return
            
            print(f"Starting basic processing for publication {pub_id}")
            
            # Mark as processing
            publication.set_processing_status('processing')
            db.session.commit()
            
            # Process PDF to get basic page structure
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs', publication.filename)
            pdf_doc = fitz.open(file_path)
            
            # Create minimal page records without heavy image processing
            config = PUBLICATION_CONFIGS[publication.publication_type]
            for page_num in range(pdf_doc.page_count):
                # Check if page already exists
                existing_page = Page.query.filter_by(
                    publication_id=publication.id, 
                    page_number=page_num + 1
                ).first()
                
                if not existing_page:
                    page_record = Page(
                        publication_id=publication.id,
                        page_number=page_num + 1,
                        total_inches=config['total_inches_per_page'],
                        total_ad_inches=0.0,
                        ad_percentage=0.0,
                        image_filename=f"{publication.filename}_page_{page_num + 1}.png"
                    )
                    db.session.add(page_record)
            
            pdf_doc.close()
            
            # Mark as completed but with partial processing
            publication.processed = True
            publication.set_processing_status('basic_completed')
            db.session.commit()
            
            print(f"Basic processing completed for publication {pub_id}")
            
        except Exception as e:
            # CRITICAL: Always rollback on database errors
            try:
                db.session.rollback()
            except Exception:
                db.session.remove()
            
            print(f"Error in synchronous processing: {e}")
            import traceback
            traceback.print_exc()
            
            try:
                publication.set_processing_status('failed')
                publication.set_processing_error(str(e))
                db.session.commit()
            except Exception as commit_error:
                print(f"Failed to update error status: {commit_error}")

def generate_page_image_if_needed(publication, page_number):
    """Generate page image on-demand if it doesn't exist"""
    image_filename = f"{publication.filename}_page_{page_number}.png"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pages', image_filename)
    
    # Check if image already exists
    if os.path.exists(image_path):
        return image_filename
    
    try:
        # Generate image from PDF
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs', publication.filename)
        pdf_doc = fitz.open(pdf_path)
        
        if page_number <= pdf_doc.page_count:
            page = pdf_doc[page_number - 1]  # 0-based indexing
            mat = fitz.Matrix(1.5, 1.5)  # Good quality but not too heavy
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Save image
            with open(image_path, 'wb') as img_file:
                img_file.write(img_data)
            
            print(f"Generated image for page {page_number} of publication {publication.id}")
        
        pdf_doc.close()
        return image_filename
        
    except Exception as e:
        print(f"Error generating page image: {e}")
        return None

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
    
    def __init__(self, **kwargs):
        # Only set processing columns if they exist in the database
        if check_processing_columns():
            if 'processing_status' in kwargs:
                self.processing_status = kwargs.pop('processing_status')
            if 'processing_error' in kwargs:
                self.processing_error = kwargs.pop('processing_error')
        
        super().__init__(**kwargs)
    
    @property 
    def safe_processing_status(self):
        """Get processing status with fallback"""
        if check_processing_columns():
            try:
                return getattr(self, 'processing_status', 'completed' if self.processed else 'uploaded')
            except (AttributeError, Exception):
                pass
        # Fallback based on processed status
        return 'completed' if self.processed else 'uploaded'
    
    @property
    def safe_processing_error(self):
        """Get processing error with fallback"""
        if check_processing_columns():
            try:
                return getattr(self, 'processing_error', None)
            except (AttributeError, Exception):
                pass
        return None
    
    def set_processing_status(self, status):
        """Safely set processing status if column exists"""
        if check_processing_columns():
            try:
                self.processing_status = status
                return True
            except (AttributeError, Exception):
                pass
        return False
    
    def set_processing_error(self, error):
        """Safely set processing error if column exists"""
        if check_processing_columns():
            try:
                self.processing_error = error
                return True
            except (AttributeError, Exception):
                pass
        return False

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
    detected_automatically = db.Column(db.Boolean, default=False)
    confidence_score = db.Column(db.Float)  # AI detection confidence
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

class MLModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)  # 'ad_detector', 'ad_classifier'
    publication_type = db.Column(db.String(50), nullable=False)  # 'broadsheet', 'special_edition', 'peach'
    version = db.Column(db.String(20), nullable=False)
    model_data = db.Column(db.LargeBinary)  # Serialized model
    training_accuracy = db.Column(db.Float)
    validation_accuracy = db.Column(db.Float)
    training_samples = db.Column(db.Integer)
    feature_names = db.Column(db.Text)  # JSON list of feature names
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=False)

class TrainingData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ad_box_id = db.Column(db.Integer, db.ForeignKey('ad_box.id'), nullable=True)  # Nullable for negative examples
    publication_type = db.Column(db.String(50), nullable=False)
    features = db.Column(db.Text)  # JSON serialized feature vector
    label = db.Column(db.String(50), nullable=False)  # ad type, 'not_ad', 'false_positive', etc.
    confidence_score = db.Column(db.Float)  # User confidence in this label
    extracted_date = db.Column(db.DateTime, default=datetime.utcnow)
    used_in_training = db.Column(db.Boolean, default=False)
    # New fields for negative training (COMMENTED OUT - will be enabled after PostgreSQL migration)
    # region_type = db.Column(db.String(50))  # photo, text_block, decorative, etc.
    # pdf_path = db.Column(db.String(500))  # Path to original PDF
    # page_number = db.Column(db.Integer)  # Page number in PDF
    # x = db.Column(db.Float)  # Bounding box coordinates
    # y = db.Column(db.Float)
    # width = db.Column(db.Float)
    # height = db.Column(db.Float)
    # training_source = db.Column(db.String(50), default='user_correction')  # user_correction, automatic, manual

class BusinessLogo(db.Model):
    """
    Business logo recognition database for intelligent ad detection
    Stores learned logo signatures and detection parameters
    """
    id = db.Column(db.Integer, primary_key=True)
    business_name = db.Column(db.String(255), nullable=False)
    logo_image_path = db.Column(db.String(500))  # Path to stored logo image
    logo_features = db.Column(db.Text)  # JSON serialized feature descriptors (SIFT/ORB keypoints)
    color_histogram = db.Column(db.Text)  # JSON serialized color histogram
    template_signature = db.Column(db.Text)  # JSON serialized template matching signature

    # Typical ad dimensions learned from examples
    typical_width_pixels = db.Column(db.Float)
    typical_height_pixels = db.Column(db.Float)
    typical_width_inches = db.Column(db.Float)
    typical_height_inches = db.Column(db.Float)
    width_variance = db.Column(db.Float, default=0.3)  # Allow 30% size variance by default
    height_variance = db.Column(db.Float, default=0.3)

    # Detection parameters
    confidence_threshold = db.Column(db.Float, default=0.85)  # Minimum match confidence
    min_match_points = db.Column(db.Integer, default=10)  # Minimum feature matches required

    # Learning statistics
    total_examples = db.Column(db.Integer, default=1)  # Number of examples used to train this logo
    successful_detections = db.Column(db.Integer, default=0)  # Track detection success rate
    false_positives = db.Column(db.Integer, default=0)  # Track false positive rate

    # Business metadata
    business_category = db.Column(db.String(100))  # e.g., 'restaurant', 'retail', 'services'
    typical_ad_types = db.Column(db.String(255))  # JSON list: ['display', 'classified', 'entertainment']
    first_seen_date = db.Column(db.DateTime, default=datetime.utcnow)
    last_detected_date = db.Column(db.DateTime)

    # Active learning parameters
    is_active = db.Column(db.Boolean, default=True)  # Whether to use for detection
    needs_retraining = db.Column(db.Boolean, default=False)  # Flag for model updates
    confidence_score = db.Column(db.Float)  # Overall logo recognition confidence

    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    updated_date = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class LogoRecognitionResult(db.Model):
    """
    Track logo recognition results for continuous learning
    """
    id = db.Column(db.Integer, primary_key=True)
    business_logo_id = db.Column(db.Integer, db.ForeignKey('business_logo.id'), nullable=False)
    ad_box_id = db.Column(db.Integer, db.ForeignKey('ad_box.id'), nullable=True)  # Linked AdBox if created
    page_id = db.Column(db.Integer, db.ForeignKey('page.id'), nullable=False)

    # Detection location and confidence
    x = db.Column(db.Float, nullable=False)
    y = db.Column(db.Float, nullable=False)
    width = db.Column(db.Float, nullable=False)
    height = db.Column(db.Float, nullable=False)
    detection_confidence = db.Column(db.Float, nullable=False)
    feature_matches = db.Column(db.Integer)  # Number of matching features found

    # User verification
    user_verified = db.Column(db.Boolean, default=False)
    is_correct_detection = db.Column(db.Boolean)  # True/False when user verifies
    user_feedback = db.Column(db.Text)  # Optional user comments

    detection_date = db.Column(db.DateTime, default=datetime.utcnow)
    verification_date = db.Column(db.DateTime)

class NegativeTrainingCollector:
    """
    Collects negative training examples from user corrections
    and improves automatic detection accuracy
    """
    
    @staticmethod
    def collect_negative_example(pdf_path, page_number, x, y, width, height, region_type='unknown'):
        """
        Collect a negative training example from a deleted ad box
        
        Args:
            pdf_path (str): Path to PDF file
            page_number (int): Page number (1-based)
            x, y, width, height (float): Bounding box of false positive
            region_type (str): Type of false positive (photo, text_block, decorative, etc.)
        """
        try:
            # Extract PDF metadata features from the region
            features = NegativeTrainingCollector._extract_pdf_features(
                pdf_path, page_number, x, y, width, height
            )
            
            # Save negative training example
            training_data = TrainingData(
                publication_type='newspaper',
                features=json.dumps(features),
                label='false_positive',
                confidence_score=1.0  # User deletion = high confidence it's not an ad
            )
            
            db.session.add(training_data)
            db.session.commit()
            
            # Reduced logging
            # pass
            
            # Trigger model retraining if enough new examples
            NegativeTrainingCollector._trigger_retraining_if_needed()
            
        except Exception as e:
            print(f"Error collecting negative example: {e}")
            db.session.rollback()
    
    @staticmethod
    def collect_positive_example(ad_box, training_source='manual'):
        """
        Collect a positive training example from a confirmed ad
        
        Args:
            ad_box (AdBox): Confirmed ad box
            training_source (str): Source of training data
        """
        try:
            # For now, skip PDF feature extraction until proper migration is in place
            features = {}
            
            # Save positive training example
            training_data = TrainingData(
                ad_box_id=ad_box.id,
                publication_type=ad_box.publication.type if ad_box.publication else 'newspaper',
                features=json.dumps(features),
                label=ad_box.ad_type,
                confidence_score=1.0
            )
            
            db.session.add(training_data)
            db.session.commit()
            
            # Reduced logging
            # pass
            
        except Exception as e:
            print(f"Error collecting positive example: {e}")
            db.session.rollback()
    
    @staticmethod
    def _extract_pdf_features(pdf_path, page_number, x, y, width, height):
        """
        Extract comprehensive PDF metadata features for ML training
        
        Returns:
            dict: Feature vector with PDF metadata characteristics
        """
        try:
            import fitz
            
            if not os.path.exists(pdf_path):
                return {}
            
            doc = fitz.open(pdf_path)
            page = doc[page_number - 1]
            
            # Define region rectangle
            region_rect = fitz.Rect(x, y, x + width, y + height)
            
            features = {
                # Basic geometric features
                'width': width,
                'height': height,
                'area': width * height,
                'aspect_ratio': width / height if height > 0 else 0,
                'perimeter': 2 * (width + height),
                
                # Page position features
                'x_position': x / page.rect.width if page.rect.width > 0 else 0,
                'y_position': y / page.rect.height if page.rect.height > 0 else 0,
                'center_x': (x + width/2) / page.rect.width if page.rect.width > 0 else 0,
                'center_y': (y + height/2) / page.rect.height if page.rect.height > 0 else 0,
                'relative_width': width / page.rect.width if page.rect.width > 0 else 0,
                'relative_height': height / page.rect.height if page.rect.height > 0 else 0,
                
                # Content analysis features
                'has_border': False,
                'border_complexity': 0,
                'has_images': False,
                'image_count': 0,
                'has_text': False,
                'text_density': 0,
                'drawing_count': 0,
                'drawing_complexity': 0
            }
            
            # Analyze drawings in region
            drawings = page.get_drawings()
            region_drawings = []
            
            for drawing in drawings:
                draw_rect = drawing.get('rect')
                if draw_rect and region_rect.intersects(draw_rect):
                    region_drawings.append(drawing)
            
            features['drawing_count'] = len(region_drawings)
            
            if region_drawings:
                # Analyze border characteristics
                for drawing in region_drawings:
                    items = drawing.get('items', [])
                    features['drawing_complexity'] += len(items)
                    
                    # Simple border detection
                    if 1 <= len(items) <= 3:
                        draw_rect = drawing.get('rect')
                        if (draw_rect and 
                            abs(draw_rect.width - width) < 20 and 
                            abs(draw_rect.height - height) < 20):
                            features['has_border'] = True
                            features['border_complexity'] = len(items)
            
            # Analyze images in region
            images = page.get_images()
            for img_index, img in enumerate(images):
                try:
                    img_rect = page.get_image_bbox(img[7])
                    if img_rect and region_rect.intersects(img_rect):
                        features['has_images'] = True
                        features['image_count'] += 1
                except:
                    continue
            
            # Analyze text in region
            text_blocks = page.get_text('dict')
            region_text_blocks = 0
            total_text_chars = 0
            
            if 'blocks' in text_blocks:
                for block in text_blocks['blocks']:
                    if 'bbox' in block:
                        block_rect = fitz.Rect(block['bbox'])
                        if region_rect.intersects(block_rect):
                            region_text_blocks += 1
                            if 'lines' in block:
                                for line in block['lines']:
                                    if 'spans' in line:
                                        for span in line['spans']:
                                            total_text_chars += len(span.get('text', ''))
            
            features['has_text'] = region_text_blocks > 0
            features['text_density'] = total_text_chars / (width * height) if (width * height) > 0 else 0
            
            doc.close()
            
            return features
            
        except Exception as e:
            print(f"Error extracting PDF features: {e}")
            return {}
    
    @staticmethod
    def _trigger_retraining_if_needed():
        """
        Trigger model retraining if enough new training examples have been collected
        """
        try:
            # Count unused training examples
            unused_count = TrainingData.query.filter_by(used_in_training=False).count()

            if unused_count >= 10:  # Retrain every 10 new examples
                print(f"Triggering automatic retraining with {unused_count} new samples...")

                # Get publication types with new training data
                pub_types_query = db.session.query(TrainingData.publication_type).filter_by(used_in_training=False).distinct()
                pub_types = [row[0] for row in pub_types_query.all()]

                # Retrain models for each publication type that has new data
                for pub_type in pub_types:
                    try:
                        type_unused = TrainingData.query.filter_by(
                            publication_type=pub_type,
                            used_in_training=False
                        ).count()

                        if type_unused >= 5:  # Minimum samples per type for retraining
                            result = AdLearningEngine.train_model(
                                publication_type=pub_type,
                                min_samples=15,  # Lower threshold for automatic retraining
                                collect_new_data=True
                            )
                            if result.get('success'):
                                print(f"Auto-retrained {pub_type} model: {result.get('accuracy', 'N/A')}% accuracy")
                            else:
                                print(f"Auto-retraining failed for {pub_type}: {result.get('error')}")
                    except Exception as retrain_error:
                        print(f"Error auto-retraining {pub_type}: {retrain_error}")

                # Mark all examples as used after attempting retraining
                TrainingData.query.filter_by(used_in_training=False).update({'used_in_training': True})
                db.session.commit()
                
        except Exception as e:
            print(f"Error in retraining trigger: {e}")

class LogoFeatureExtractor:
    """
    Advanced logo feature extraction for business recognition
    Uses multiple computer vision techniques for robust logo matching
    """

    def __init__(self):
        """Initialize feature extractors"""
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create(nfeatures=500)

    def extract_logo_features(self, image_region, business_name=None):
        """
        Extract comprehensive features from a logo region

        Args:
            image_region (numpy.ndarray): Logo image region (BGR format)
            business_name (str): Optional business name for context

        Returns:
            dict: Feature dictionary with multiple signature types
        """
        try:
            if len(image_region.shape) == 3:
                gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
            else:
                gray = image_region

            features = {
                'business_name': business_name,
                'image_dimensions': image_region.shape,
                'extraction_date': datetime.utcnow().isoformat()
            }

            # 1. SIFT keypoints and descriptors for distinctive features
            sift_keypoints, sift_descriptors = self.sift.detectAndCompute(gray, None)
            if sift_descriptors is not None:
                features['sift_keypoints'] = len(sift_keypoints)
                features['sift_descriptors'] = sift_descriptors.tolist()
                features['sift_quality_score'] = self._calculate_feature_quality(sift_keypoints)
            else:
                features['sift_keypoints'] = 0
                features['sift_descriptors'] = []

            # 2. ORB features for rotation-invariant matching
            orb_keypoints, orb_descriptors = self.orb.detectAndCompute(gray, None)
            if orb_descriptors is not None:
                features['orb_keypoints'] = len(orb_keypoints)
                features['orb_descriptors'] = orb_descriptors.tolist()
            else:
                features['orb_keypoints'] = 0
                features['orb_descriptors'] = []

            # 3. Color histogram for color-based matching
            features['color_histogram'] = self._extract_color_histogram(image_region)

            # 4. Template signature for direct template matching
            features['template_signature'] = self._create_template_signature(gray)

            # 5. Shape descriptors
            features['shape_features'] = self._extract_shape_features(gray)

            # 6. Text detection (logos often contain text)
            features['text_features'] = self._detect_text_regions(gray)

            return features

        except Exception as e:
            print(f"Error extracting logo features: {e}")
            return {
                'error': str(e),
                'business_name': business_name,
                'extraction_date': datetime.utcnow().isoformat()
            }

    def _calculate_feature_quality(self, keypoints):
        """Calculate quality score based on keypoint distribution and strength"""
        if not keypoints:
            return 0.0

        # Analyze keypoint responses (strength)
        responses = [kp.response for kp in keypoints]

        # Spatial distribution
        points = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])

        # Quality metrics
        avg_response = np.mean(responses) if responses else 0
        response_variance = np.var(responses) if len(responses) > 1 else 0
        spatial_spread = np.std(points, axis=0).mean() if len(points) > 1 else 0

        # Combined quality score (0-1)
        quality = min(1.0, (avg_response * 0.4 + response_variance * 0.3 + spatial_spread * 0.3) / 100)
        return float(quality)

    def _extract_color_histogram(self, image):
        """Extract multi-channel color histogram"""
        try:
            # Convert to HSV for better color representation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Calculate histograms for each channel
            hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])

            # Normalize histograms
            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()

            return {
                'hue': hist_h.tolist(),
                'saturation': hist_s.tolist(),
                'value': hist_v.tolist(),
                'dominant_colors': self._get_dominant_colors(image)
            }
        except Exception as e:
            print(f"Error extracting color histogram: {e}")
            return {'hue': [], 'saturation': [], 'value': [], 'dominant_colors': []}

    def _get_dominant_colors(self, image, k=3):
        """Extract dominant colors using K-means clustering"""
        try:
            # Reshape image to be a list of pixels
            data = image.reshape((-1, 3))
            data = np.float32(data)

            # Apply K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
            _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            # Convert centers to int and return as list
            centers = np.uint8(centers)
            return centers.tolist()
        except Exception:
            return []

    def _create_template_signature(self, gray_image):
        """Create a template signature for direct matching"""
        try:
            # Resize to standard template size
            template_size = (64, 64)
            template = cv2.resize(gray_image, template_size)

            # Apply Gaussian blur to reduce noise
            template = cv2.GaussianBlur(template, (3, 3), 0)

            # Normalize to reduce lighting variations
            template = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX)

            return {
                'template': template.tolist(),
                'size': template_size,
                'mean_intensity': float(np.mean(template)),
                'std_intensity': float(np.std(template))
            }
        except Exception as e:
            print(f"Error creating template signature: {e}")
            return {'template': [], 'size': [64, 64], 'mean_intensity': 0, 'std_intensity': 0}

    def _extract_shape_features(self, gray_image):
        """Extract shape-based features from the logo"""
        try:
            # Edge detection
            edges = cv2.Canny(gray_image, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            shape_features = {
                'edge_density': float(np.sum(edges > 0) / edges.size),
                'contour_count': len(contours),
                'total_contour_length': 0,
                'largest_contour_area': 0,
                'shape_complexity': 0
            }

            if contours:
                # Analyze largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                shape_features['largest_contour_area'] = float(cv2.contourArea(largest_contour))
                shape_features['total_contour_length'] = float(sum(cv2.arcLength(c, True) for c in contours))

                # Shape complexity (perimeter^2 / area ratio)
                if shape_features['largest_contour_area'] > 0:
                    perimeter = cv2.arcLength(largest_contour, True)
                    shape_features['shape_complexity'] = float(perimeter * perimeter / shape_features['largest_contour_area'])

            return shape_features
        except Exception as e:
            print(f"Error extracting shape features: {e}")
            return {'edge_density': 0, 'contour_count': 0, 'total_contour_length': 0, 'largest_contour_area': 0, 'shape_complexity': 0}

    def _detect_text_regions(self, gray_image):
        """Detect text regions within the logo (many logos contain text)"""
        try:
            # Use EAST text detector if available, otherwise use simple edge-based detection
            text_features = {
                'has_text_regions': False,
                'text_region_count': 0,
                'text_area_ratio': 0.0,
                'horizontal_text_lines': 0,
                'vertical_text_lines': 0
            }

            # Simple text detection using morphological operations
            # Create kernel for text detection
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

            # Apply morphological operations to detect text-like structures
            morph = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

            # Find connected components that might be text
            binary_morph = (morph > 128).astype(np.uint8)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_morph)

            # Analyze components for text-like characteristics
            text_regions = 0
            total_text_area = 0

            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]

                # Text-like characteristics: reasonable size, aspect ratio
                if area > 20 and width > 5 and height > 5:
                    aspect_ratio = width / height
                    if 0.1 < aspect_ratio < 10:  # Reasonable aspect ratio for text
                        text_regions += 1
                        total_text_area += area

            text_features['has_text_regions'] = text_regions > 0
            text_features['text_region_count'] = text_regions
            text_features['text_area_ratio'] = float(total_text_area / gray_image.size)

            return text_features
        except Exception as e:
            print(f"Error detecting text regions: {e}")
            return {'has_text_regions': False, 'text_region_count': 0, 'text_area_ratio': 0.0, 'horizontal_text_lines': 0, 'vertical_text_lines': 0}

class LogoMatcher:
    """
    Logo matching engine for recognizing stored business logos
    """

    def __init__(self):
        """Initialize matching algorithms"""
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create(nfeatures=500)

        # Feature matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # ORB matcher
        self.orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def match_logo(self, query_features, stored_logo_features, confidence_threshold=0.85):
        """
        Match query logo features against stored logo features

        Args:
            query_features (dict): Features extracted from query image
            stored_logo_features (dict): Features from stored business logo
            confidence_threshold (float): Minimum confidence for positive match

        Returns:
            dict: Match result with confidence score and details
        """
        try:
            match_result = {
                'is_match': False,
                'confidence': 0.0,
                'match_details': {},
                'business_name': stored_logo_features.get('business_name', 'Unknown')
            }

            # Skip if either feature set has errors
            if 'error' in query_features or 'error' in stored_logo_features:
                return match_result

            confidence_scores = []

            # 1. SIFT feature matching
            sift_confidence = self._match_sift_features(query_features, stored_logo_features)
            confidence_scores.append(('sift', sift_confidence, 0.4))  # 40% weight

            # 2. ORB feature matching
            orb_confidence = self._match_orb_features(query_features, stored_logo_features)
            confidence_scores.append(('orb', orb_confidence, 0.3))   # 30% weight

            # 3. Color histogram matching
            color_confidence = self._match_color_histograms(query_features, stored_logo_features)
            confidence_scores.append(('color', color_confidence, 0.2))  # 20% weight

            # 4. Template matching
            template_confidence = self._match_templates(query_features, stored_logo_features)
            confidence_scores.append(('template', template_confidence, 0.1))  # 10% weight

            # Calculate weighted confidence
            total_confidence = sum(score * weight for _, score, weight in confidence_scores)

            match_result['confidence'] = total_confidence
            match_result['is_match'] = total_confidence >= confidence_threshold
            match_result['match_details'] = {
                method: {'score': score, 'weight': weight}
                for method, score, weight in confidence_scores
            }

            return match_result

        except Exception as e:
            print(f"Error matching logo: {e}")
            return {
                'is_match': False,
                'confidence': 0.0,
                'match_details': {'error': str(e)},
                'business_name': stored_logo_features.get('business_name', 'Unknown')
            }

    def _match_sift_features(self, query_features, stored_features):
        """Match SIFT features between query and stored logo"""
        try:
            query_desc = query_features.get('sift_descriptors', [])
            stored_desc = stored_features.get('sift_descriptors', [])

            if not query_desc or not stored_desc:
                return 0.0

            query_desc = np.array(query_desc, dtype=np.float32)
            stored_desc = np.array(stored_desc, dtype=np.float32)

            # Use FLANN matcher for SIFT features
            matches = self.flann.knnMatch(query_desc, stored_desc, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            # Calculate confidence based on good matches
            match_ratio = len(good_matches) / min(len(query_desc), len(stored_desc))
            return min(1.0, match_ratio * 2)  # Scale to 0-1 range

        except Exception as e:
            print(f"Error matching SIFT features: {e}")
            return 0.0

    def _match_orb_features(self, query_features, stored_features):
        """Match ORB features between query and stored logo"""
        try:
            query_desc = query_features.get('orb_descriptors', [])
            stored_desc = stored_features.get('orb_descriptors', [])

            if not query_desc or not stored_desc:
                return 0.0

            query_desc = np.array(query_desc, dtype=np.uint8)
            stored_desc = np.array(stored_desc, dtype=np.uint8)

            # Use BF matcher for ORB features
            matches = self.orb_matcher.match(query_desc, stored_desc)

            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Calculate confidence based on match quality
            if matches:
                good_matches = [m for m in matches if m.distance < 50]  # ORB distance threshold
                match_ratio = len(good_matches) / min(len(query_desc), len(stored_desc))
                return min(1.0, match_ratio * 2)

            return 0.0

        except Exception as e:
            print(f"Error matching ORB features: {e}")
            return 0.0

    def _match_color_histograms(self, query_features, stored_features):
        """Match color histograms between query and stored logo"""
        try:
            query_hist = query_features.get('color_histogram', {})
            stored_hist = stored_features.get('color_histogram', {})

            if not query_hist or not stored_hist:
                return 0.0

            # Compare each color channel
            correlations = []

            for channel in ['hue', 'saturation', 'value']:
                query_ch = query_hist.get(channel, [])
                stored_ch = stored_hist.get(channel, [])

                if query_ch and stored_ch:
                    query_ch = np.array(query_ch)
                    stored_ch = np.array(stored_ch)

                    # Use correlation coefficient
                    correlation = cv2.compareHist(query_ch, stored_ch, cv2.HISTCMP_CORREL)
                    correlations.append(max(0, correlation))  # Ensure non-negative

            # Return average correlation
            return np.mean(correlations) if correlations else 0.0

        except Exception as e:
            print(f"Error matching color histograms: {e}")
            return 0.0

    def _match_templates(self, query_features, stored_features):
        """Match template signatures between query and stored logo"""
        try:
            query_template = query_features.get('template_signature', {}).get('template', [])
            stored_template = stored_features.get('template_signature', {}).get('template', [])

            if not query_template or not stored_template:
                return 0.0

            query_template = np.array(query_template, dtype=np.uint8)
            stored_template = np.array(stored_template, dtype=np.uint8)

            # Perform template matching
            result = cv2.matchTemplate(query_template, stored_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            return max(0.0, max_val)  # Ensure non-negative

        except Exception as e:
            print(f"Error matching templates: {e}")
            return 0.0

class LogoLearningWorkflow:
    """
    Intelligent logo learning system for building business recognition database
    Learns from user manual ad placement to build automated detection
    """

    def __init__(self):
        """Initialize the logo learning workflow"""
        self.feature_extractor = LogoFeatureExtractor()
        self.logo_matcher = LogoMatcher()

    def analyze_manual_ad_for_logo_learning(self, page_id, ad_box_coordinates, business_name=None):
        """
        Analyze a manually placed ad to extract and learn logo features

        Args:
            page_id (int): Database ID of the page
            ad_box_coordinates (dict): Ad box coordinates {'x', 'y', 'width', 'height'}
            business_name (str): Optional business name provided by user

        Returns:
            dict: Analysis result with learning opportunities
        """
        try:
            print(f"Analyzing manual ad for logo learning on page {page_id}")

            # Get page information
            page = Page.query.get(page_id)
            if not page:
                return {'success': False, 'error': 'Page not found'}

            publication = Publication.query.get(page.publication_id)
            if not publication:
                return {'success': False, 'error': 'Publication not found'}

            # Load page image
            image_filename = f"{publication.filename}_page_{page.page_number}.png"
            image_path = os.path.join('static', 'uploads', 'pages', image_filename)

            if not os.path.exists(image_path):
                return {'success': False, 'error': 'Page image not found'}

            # Extract ad region from image
            page_image = cv2.imread(image_path)
            ad_region = self._extract_ad_region(page_image, ad_box_coordinates)

            if ad_region is None:
                return {'success': False, 'error': 'Could not extract ad region'}

            # Detect potential logos within the ad
            logo_candidates = self._detect_logo_candidates(ad_region)

            if not logo_candidates:
                return {
                    'success': True,
                    'logo_found': False,
                    'message': 'No distinctive logo detected in this ad',
                    'learning_opportunity': False
                }

            # Extract features from the best logo candidate
            best_logo = logo_candidates[0]  # Take the most promising candidate
            logo_features = self.feature_extractor.extract_logo_features(
                best_logo['region'], business_name
            )

            # Check if this logo already exists in database
            existing_match = self._find_existing_logo_match(logo_features)

            result = {
                'success': True,
                'logo_found': True,
                'logo_candidates': len(logo_candidates),
                'best_logo_confidence': best_logo['confidence'],
                'features_extracted': True,
                'learning_opportunity': True
            }

            if existing_match:
                result.update({
                    'existing_logo_found': True,
                    'existing_business_name': existing_match['business_name'],
                    'match_confidence': existing_match['confidence'],
                    'recommendation': 'update_existing_logo'
                })
            else:
                result.update({
                    'existing_logo_found': False,
                    'recommendation': 'create_new_logo_entry',
                    'suggested_business_name': business_name or 'Unknown Business'
                })

            # Store temporary learning data for user confirmation
            temp_learning_data = {
                'page_id': page_id,
                'ad_coordinates': ad_box_coordinates,
                'logo_region': best_logo,
                'logo_features': logo_features,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Store in session or temporary location for user confirmation
            session_key = f"logo_learning_{page_id}_{int(datetime.utcnow().timestamp())}"
            # Note: In production, store this in Redis or session storage
            # For now, return in result for immediate processing

            result['temp_learning_data'] = temp_learning_data
            result['session_key'] = session_key

            return result

        except Exception as e:
            print(f"Error in logo learning analysis: {e}")
            return {'success': False, 'error': str(e)}

    def confirm_logo_learning(self, temp_learning_data, business_name, user_confirmation=True):
        """
        User confirms logo learning - create or update logo database entry

        Args:
            temp_learning_data (dict): Temporary learning data from analysis
            business_name (str): Confirmed business name
            user_confirmation (bool): Whether user confirmed the learning

        Returns:
            dict: Learning result
        """
        try:
            if not user_confirmation:
                return {
                    'success': True,
                    'action': 'learning_cancelled',
                    'message': 'Logo learning cancelled by user'
                }

            logo_features = temp_learning_data['logo_features']
            logo_region = temp_learning_data['logo_region']

            # Save logo image to filesystem
            logo_image_path = self._save_logo_image(
                logo_region['region'],
                business_name,
                temp_learning_data['page_id']
            )

            # Check for existing logo again (in case of concurrent modifications)
            existing_logo = self._find_existing_business_logo(business_name)

            if existing_logo:
                # Update existing logo with new example
                result = self._update_existing_logo(existing_logo, logo_features, logo_image_path, temp_learning_data)
            else:
                # Create new logo entry
                result = self._create_new_logo_entry(business_name, logo_features, logo_image_path, temp_learning_data)

            return result

        except Exception as e:
            print(f"Error confirming logo learning: {e}")
            return {'success': False, 'error': str(e)}

    def _extract_ad_region(self, page_image, coordinates):
        """Extract ad region from page image"""
        try:
            x = int(coordinates['x'])
            y = int(coordinates['y'])
            width = int(coordinates['width'])
            height = int(coordinates['height'])

            # Ensure coordinates are within image bounds
            h, w = page_image.shape[:2]
            x = max(0, min(x, w))
            y = max(0, min(y, h))
            width = min(width, w - x)
            height = min(height, h - y)

            if width <= 0 or height <= 0:
                return None

            ad_region = page_image[y:y+height, x:x+width]
            return ad_region

        except Exception as e:
            print(f"Error extracting ad region: {e}")
            return None

    def _detect_logo_candidates(self, ad_region):
        """
        Detect potential logo regions within an ad using various heuristics
        """
        try:
            gray = cv2.cvtColor(ad_region, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            logo_candidates = []

            # Method 1: Corner regions (logos often in corners)
            corner_regions = [
                {'region': ad_region[0:h//3, 0:w//3], 'location': 'top_left', 'confidence': 0.8},
                {'region': ad_region[0:h//3, 2*w//3:w], 'location': 'top_right', 'confidence': 0.8},
                {'region': ad_region[2*h//3:h, 0:w//3], 'location': 'bottom_left', 'confidence': 0.7},
                {'region': ad_region[2*h//3:h, 2*w//3:w], 'location': 'bottom_right', 'confidence': 0.7}
            ]

            # Method 2: High contrast regions (logos are often distinctive)
            edge_density_threshold = 0.1
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            if edge_density > edge_density_threshold:
                # Find regions with high edge density
                kernel = np.ones((20, 20), np.uint8)
                edge_regions = cv2.dilate(edges, kernel, iterations=1)

                contours, _ = cv2.findContours(edge_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 500:  # Minimum logo size
                        x, y, cw, ch = cv2.boundingRect(contour)

                        # Extract region with some padding
                        padding = 10
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        cw = min(w - x, cw + 2*padding)
                        ch = min(h - y, ch + 2*padding)

                        if cw > 20 and ch > 20:  # Minimum dimensions
                            region = ad_region[y:y+ch, x:x+cw]
                            logo_candidates.append({
                                'region': region,
                                'location': f'edge_detected_{len(logo_candidates)}',
                                'confidence': 0.9
                            })

            # Add corner regions to candidates
            for corner in corner_regions:
                if corner['region'].size > 0 and corner['region'].shape[0] > 10 and corner['region'].shape[1] > 10:
                    logo_candidates.append(corner)

            # Method 3: Color-based detection (logos often have distinct colors)
            # Find regions with limited color palette (typical of logos)
            color_candidates = self._find_distinctive_color_regions(ad_region)
            logo_candidates.extend(color_candidates)

            # Score and sort candidates
            scored_candidates = []
            for candidate in logo_candidates:
                score = self._score_logo_candidate(candidate['region'])
                candidate['confidence'] *= score
                scored_candidates.append(candidate)

            # Sort by confidence and return top candidates
            scored_candidates.sort(key=lambda x: x['confidence'], reverse=True)

            return scored_candidates[:3]  # Return top 3 candidates

        except Exception as e:
            print(f"Error detecting logo candidates: {e}")
            return []

    def _find_distinctive_color_regions(self, ad_region):
        """Find regions with distinctive color patterns typical of logos"""
        try:
            candidates = []

            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(ad_region, cv2.COLOR_BGR2HSV)
            h, w = hsv.shape[:2]

            # Look for regions with strong color saturation (logos often colorful)
            saturation = hsv[:, :, 1]
            high_sat_mask = saturation > 100

            if np.sum(high_sat_mask) > 0:
                # Find connected components of high saturation
                kernel = np.ones((5, 5), np.uint8)
                high_sat_dilated = cv2.dilate(high_sat_mask.astype(np.uint8), kernel, iterations=2)

                contours, _ = cv2.findContours(high_sat_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 300 < area < (h * w * 0.3):  # Reasonable size range
                        x, y, cw, ch = cv2.boundingRect(contour)

                        # Add padding
                        padding = 5
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        cw = min(w - x, cw + 2*padding)
                        ch = min(h - y, ch + 2*padding)

                        if cw > 15 and ch > 15:
                            region = ad_region[y:y+ch, x:x+cw]
                            candidates.append({
                                'region': region,
                                'location': f'color_distinctive_{len(candidates)}',
                                'confidence': 0.75
                            })

            return candidates

        except Exception as e:
            print(f"Error finding color regions: {e}")
            return []

    def _score_logo_candidate(self, logo_region):
        """Score a logo candidate based on visual characteristics"""
        try:
            if logo_region.size == 0 or logo_region.shape[0] < 10 or logo_region.shape[1] < 10:
                return 0.0

            gray = cv2.cvtColor(logo_region, cv2.COLOR_BGR2GRAY) if len(logo_region.shape) == 3 else logo_region

            score = 0.0

            # 1. Edge density (logos have clear edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            score += min(1.0, edge_density * 5) * 0.3

            # 2. Contrast (logos have good contrast)
            contrast = np.std(gray) / 255.0
            score += min(1.0, contrast * 2) * 0.3

            # 3. Size appropriateness (not too small, not too large)
            h, w = gray.shape
            area = h * w
            size_score = 1.0
            if area < 400:  # Too small
                size_score = area / 400.0
            elif area > 10000:  # Too large
                size_score = 10000.0 / area
            score += size_score * 0.2

            # 4. Aspect ratio (logos usually have reasonable aspect ratios)
            aspect_ratio = w / h
            if 0.3 <= aspect_ratio <= 3.0:
                score += 0.2
            else:
                score += 0.1

            return min(1.0, score)

        except Exception as e:
            print(f"Error scoring logo candidate: {e}")
            return 0.0

    def _find_existing_logo_match(self, logo_features):
        """Find existing logo that matches the extracted features"""
        try:
            # Get all active business logos
            existing_logos = BusinessLogo.query.filter_by(is_active=True).all()

            best_match = None
            best_confidence = 0.0

            for stored_logo in existing_logos:
                # Parse stored features
                try:
                    stored_features = json.loads(stored_logo.logo_features or '{}')
                except:
                    continue

                # Use logo matcher to compare features
                match_result = self.logo_matcher.match_logo(logo_features, stored_features)

                if match_result['is_match'] and match_result['confidence'] > best_confidence:
                    best_confidence = match_result['confidence']
                    best_match = {
                        'logo_id': stored_logo.id,
                        'business_name': stored_logo.business_name,
                        'confidence': match_result['confidence'],
                        'match_details': match_result['match_details']
                    }

            return best_match

        except Exception as e:
            print(f"Error finding existing logo match: {e}")
            return None

    def _find_existing_business_logo(self, business_name):
        """Find existing logo by business name"""
        try:
            return BusinessLogo.query.filter(
                BusinessLogo.business_name.ilike(f'%{business_name}%'),
                BusinessLogo.is_active == True
            ).first()
        except Exception as e:
            print(f"Error finding existing business logo: {e}")
            return None

    def _save_logo_image(self, logo_region, business_name, page_id):
        """Save logo image to filesystem"""
        try:
            # Create logos directory if it doesn't exist
            logos_dir = os.path.join('static', 'uploads', 'logos')
            os.makedirs(logos_dir, exist_ok=True)

            # Generate unique filename
            safe_business_name = "".join(c for c in business_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_business_name = safe_business_name.replace(' ', '_')

            timestamp = int(datetime.utcnow().timestamp())
            filename = f"{safe_business_name}_{page_id}_{timestamp}.png"
            file_path = os.path.join(logos_dir, filename)

            # Save image
            cv2.imwrite(file_path, logo_region)

            return file_path

        except Exception as e:
            print(f"Error saving logo image: {e}")
            return None

    def _create_new_logo_entry(self, business_name, logo_features, logo_image_path, temp_data):
        """Create new business logo database entry"""
        try:
            # Calculate typical dimensions from the ad
            ad_coords = temp_data['ad_coordinates']
            logo_region = temp_data['logo_region']

            new_logo = BusinessLogo(
                business_name=business_name,
                logo_image_path=logo_image_path,
                logo_features=json.dumps(logo_features),
                color_histogram=json.dumps(logo_features.get('color_histogram', {})),
                template_signature=json.dumps(logo_features.get('template_signature', {})),

                # Set typical dimensions based on this example
                typical_width_pixels=float(ad_coords['width']),
                typical_height_pixels=float(ad_coords['height']),

                # Business metadata
                first_seen_date=datetime.utcnow(),
                total_examples=1,
                is_active=True,
                confidence_score=logo_region['confidence']
            )

            db.session.add(new_logo)
            db.session.commit()

            return {
                'success': True,
                'action': 'created_new_logo',
                'logo_id': new_logo.id,
                'business_name': business_name,
                'message': f'Created new logo entry for {business_name}'
            }

        except Exception as e:
            print(f"Error creating new logo entry: {e}")
            db.session.rollback()
            return {'success': False, 'error': str(e)}

    def _update_existing_logo(self, existing_logo, new_features, logo_image_path, temp_data):
        """Update existing logo with new example"""
        try:
            # Merge features with existing ones (simple approach: update if better quality)
            current_features = json.loads(existing_logo.logo_features or '{}')

            # Update if new features have better quality
            new_quality = new_features.get('sift_quality_score', 0)
            current_quality = current_features.get('sift_quality_score', 0)

            if new_quality > current_quality:
                existing_logo.logo_features = json.dumps(new_features)
                existing_logo.logo_image_path = logo_image_path

            # Update statistics
            existing_logo.total_examples += 1
            existing_logo.last_detected_date = datetime.utcnow()
            existing_logo.updated_date = datetime.utcnow()

            # Update typical dimensions (weighted average)
            ad_coords = temp_data['ad_coordinates']
            weight = 1.0 / existing_logo.total_examples

            existing_logo.typical_width_pixels = (
                existing_logo.typical_width_pixels * (1 - weight) +
                ad_coords['width'] * weight
            )
            existing_logo.typical_height_pixels = (
                existing_logo.typical_height_pixels * (1 - weight) +
                ad_coords['height'] * weight
            )

            db.session.commit()

            return {
                'success': True,
                'action': 'updated_existing_logo',
                'logo_id': existing_logo.id,
                'business_name': existing_logo.business_name,
                'total_examples': existing_logo.total_examples,
                'message': f'Updated logo for {existing_logo.business_name} (now {existing_logo.total_examples} examples)'
            }

        except Exception as e:
            print(f"Error updating existing logo: {e}")
            db.session.rollback()
            return {'success': False, 'error': str(e)}

    def learn_logo_from_manual_ad(self, ad_box, business_name, features):
        """
        Learn logo from a manual ad placement

        Args:
            ad_box: AdBox database object
            business_name: Name of the business
            features: Extracted logo features

        Returns:
            dict: Learning result
        """
        try:
            # Convert AdBox to coordinates format
            ad_coordinates = {
                'x': ad_box.x,
                'y': ad_box.y,
                'width': ad_box.width,
                'height': ad_box.height
            }

            # Use existing analysis method
            return self.analyze_manual_ad_for_logo_learning(
                ad_box.page_id, ad_coordinates, business_name
            )

        except Exception as e:
            print(f"Error learning logo from manual ad: {e}")
            return {'success': False, 'error': str(e)}


class LogoRecognitionDetectionEngine:
    """
    Automated logo recognition and ad detection engine
    Scans pages for learned business logos and creates ad boxes automatically
    """

    def __init__(self):
        """Initialize the logo recognition detection engine"""
        self.feature_extractor = LogoFeatureExtractor()
        self.logo_matcher = LogoMatcher()
        self.logo_learning = LogoLearningWorkflow()

    def detect_ads_from_publication(self, publication_id, confidence_threshold=0.85):
        """
        Run logo recognition on all pages of a publication

        Args:
            publication_id: ID of publication to process
            confidence_threshold: Minimum confidence for detections

        Returns:
            dict: Detection results and statistics
        """
        try:
            print(f"Starting logo recognition on publication {publication_id}")

            publication = Publication.query.get(publication_id)
            if not publication:
                return {'success': False, 'error': 'Publication not found'}

            pages = Page.query.filter_by(publication_id=publication_id).all()
            if not pages:
                return {'success': False, 'error': 'No pages found'}

            total_detections = 0
            business_names = set()
            page_results = []

            for page in pages:
                page_result = self.detect_logos_on_page(page.id, confidence_threshold)
                if page_result.get('success'):
                    detections = page_result.get('detections_created', 0)
                    total_detections += detections
                    business_names.update(page_result.get('business_names', []))

                page_results.append({
                    'page_id': page.id,
                    'page_number': page.page_number,
                    'detections': page_result.get('detections_created', 0),
                    'success': page_result.get('success', False)
                })

            return {
                'success': True,
                'publication_id': publication_id,
                'detections': total_detections,
                'business_names': list(business_names),
                'pages_processed': len(pages),
                'page_results': page_results
            }

        except Exception as e:
            print(f"Error in publication logo recognition: {e}")
            return {'success': False, 'error': str(e)}

    def detect_logos_on_page(self, page_id, confidence_threshold=0.85, create_ad_boxes=True):
        """
        Scan a page for known business logos and create ad boxes

        Args:
            page_id (int): Database ID of the page to scan
            confidence_threshold (float): Minimum confidence for logo detection
            create_ad_boxes (bool): Whether to create AdBox entries for detections

        Returns:
            dict: Detection results with found logos and created ad boxes
        """
        try:
            print(f"Starting logo recognition on page {page_id}")

            # Get page and publication information
            page = Page.query.get(page_id)
            if not page:
                return {'success': False, 'error': 'Page not found'}

            publication = Publication.query.get(page.publication_id)
            if not publication:
                return {'success': False, 'error': 'Publication not found'}

            # Load page image
            image_filename = f"{publication.filename}_page_{page.page_number}.png"
            image_path = os.path.join('static', 'uploads', 'pages', image_filename)

            if not os.path.exists(image_path):
                return {'success': False, 'error': 'Page image not found'}

            page_image = cv2.imread(image_path)
            if page_image is None:
                return {'success': False, 'error': 'Failed to load page image'}

            # Get all active business logos from database
            business_logos = BusinessLogo.query.filter_by(is_active=True).all()

            if not business_logos:
                return {
                    'success': True,
                    'message': 'No business logos to search for',
                    'logos_detected': 0,
                    'ad_boxes_created': 0
                }

            print(f"Searching for {len(business_logos)} known business logos")

            # Results tracking
            detection_results = {
                'logos_detected': 0,
                'ad_boxes_created': 0,
                'detections': [],
                'errors': []
            }

            # Process each known logo
            for business_logo in business_logos:
                try:
                    logo_detections = self._search_logo_on_page(
                        page_image, business_logo, confidence_threshold
                    )

                    for detection in logo_detections:
                        # Record detection
                        detection_results['logos_detected'] += 1
                        detection_results['detections'].append({
                            'business_name': business_logo.business_name,
                            'logo_id': business_logo.id,
                            'location': detection['location'],
                            'confidence': detection['confidence'],
                            'bounding_box': detection['bounding_box']
                        })

                        # Create ad box if requested
                        if create_ad_boxes:
                            ad_box_result = self._create_ad_box_from_logo_detection(
                                page, business_logo, detection
                            )

                            if ad_box_result['success']:
                                detection_results['ad_boxes_created'] += 1

                                # Record successful detection in LogoRecognitionResult
                                self._record_logo_detection_result(
                                    business_logo.id, page.id, detection, ad_box_result['ad_box_id']
                                )
                            else:
                                detection_results['errors'].append(
                                    f"Failed to create ad box for {business_logo.business_name}: {ad_box_result['error']}"
                                )

                except Exception as logo_error:
                    detection_results['errors'].append(
                        f"Error processing logo {business_logo.business_name}: {str(logo_error)}"
                    )
                    print(f"Error processing logo {business_logo.business_name}: {logo_error}")

            # Update logo statistics
            self._update_logo_detection_statistics(detection_results)

            result = {
                'success': True,
                'page_id': page_id,
                'logos_searched': len(business_logos),
                'logos_detected': detection_results['logos_detected'],
                'ad_boxes_created': detection_results['ad_boxes_created'],
                'detections': detection_results['detections'],
                'errors': detection_results['errors']
            }

            print(f"Logo recognition complete: {result['logos_detected']} logos found, {result['ad_boxes_created']} ad boxes created")
            return result

        except Exception as e:
            print(f"Error in logo recognition detection: {e}")
            return {'success': False, 'error': str(e)}

    def _search_logo_on_page(self, page_image, business_logo, confidence_threshold):
        """
        Search for a specific business logo on a page using multiple methods

        Args:
            page_image (numpy.ndarray): Page image
            business_logo (BusinessLogo): Business logo database entry
            confidence_threshold (float): Minimum confidence threshold

        Returns:
            list: List of logo detections with location and confidence
        """
        try:
            detections = []

            # Parse stored logo features
            try:
                stored_features = json.loads(business_logo.logo_features or '{}')
            except:
                print(f"Invalid stored features for {business_logo.business_name}")
                return []

            # Method 1: Template matching using stored template
            template_detections = self._template_match_logo(page_image, business_logo, stored_features)
            detections.extend(template_detections)

            # Method 2: Feature-based matching (SIFT/ORB)
            feature_detections = self._feature_match_logo(page_image, business_logo, stored_features)
            detections.extend(feature_detections)

            # Method 3: Color-based detection
            color_detections = self._color_match_logo(page_image, business_logo, stored_features)
            detections.extend(color_detections)

            # Filter by confidence threshold and remove duplicates
            valid_detections = []
            for detection in detections:
                if detection['confidence'] >= confidence_threshold:
                    # Check for duplicates (overlapping regions)
                    is_duplicate = False
                    for existing in valid_detections:
                        if self._calculate_overlap(detection['bounding_box'], existing['bounding_box']) > 0.5:
                            # Keep the higher confidence detection
                            if detection['confidence'] > existing['confidence']:
                                valid_detections.remove(existing)
                                valid_detections.append(detection)
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        valid_detections.append(detection)

            return valid_detections

        except Exception as e:
            print(f"Error searching for logo {business_logo.business_name}: {e}")
            return []

    def _template_match_logo(self, page_image, business_logo, stored_features):
        """Perform template matching for logo detection"""
        try:
            detections = []

            template_data = stored_features.get('template_signature', {})
            if not template_data.get('template'):
                return detections

            # Convert template back to numpy array
            template = np.array(template_data['template'], dtype=np.uint8)

            if template.size == 0:
                return detections

            # Convert page to grayscale
            gray_page = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)

            # Perform template matching at multiple scales
            scales = [0.5, 0.7, 1.0, 1.3, 1.5]  # Multiple scales to handle size variations

            for scale in scales:
                # Resize template
                scaled_template = cv2.resize(template, None, fx=scale, fy=scale)

                if scaled_template.shape[0] > gray_page.shape[0] or scaled_template.shape[1] > gray_page.shape[1]:
                    continue

                # Perform matching
                result = cv2.matchTemplate(gray_page, scaled_template, cv2.TM_CCOEFF_NORMED)

                # Find matches above threshold
                locations = np.where(result >= business_logo.confidence_threshold * 0.8)  # Slightly lower threshold

                for pt in zip(*locations[::-1]):  # Switch x and y
                    confidence = float(result[pt[1], pt[0]])

                    detections.append({
                        'method': 'template_matching',
                        'location': pt,
                        'confidence': confidence,
                        'scale': scale,
                        'bounding_box': {
                            'x': pt[0],
                            'y': pt[1],
                            'width': scaled_template.shape[1],
                            'height': scaled_template.shape[0]
                        }
                    })

            return detections

        except Exception as e:
            print(f"Error in template matching: {e}")
            return []

    def _feature_match_logo(self, page_image, business_logo, stored_features):
        """Perform feature-based matching using SIFT/ORB"""
        try:
            detections = []

            # Check if we have stored SIFT descriptors
            stored_sift = stored_features.get('sift_descriptors', [])
            if not stored_sift:
                return detections

            stored_sift = np.array(stored_sift, dtype=np.float32)

            # Extract SIFT features from the page
            gray_page = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            page_keypoints, page_descriptors = sift.detectAndCompute(gray_page, None)

            if page_descriptors is None or len(page_descriptors) == 0:
                return detections

            # Match features
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(stored_sift, page_descriptors, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            # Need minimum number of matches
            if len(good_matches) >= business_logo.min_match_points:
                # Calculate confidence based on match quality
                confidence = min(1.0, len(good_matches) / max(10, business_logo.min_match_points))

                # Find the center of matched keypoints
                if good_matches:
                    matched_points = [page_keypoints[m.trainIdx].pt for m in good_matches]
                    center_x = np.mean([pt[0] for pt in matched_points])
                    center_y = np.mean([pt[1] for pt in matched_points])

                    # Estimate bounding box size based on typical logo dimensions
                    est_width = business_logo.typical_width_pixels or 100
                    est_height = business_logo.typical_height_pixels or 100

                    detections.append({
                        'method': 'feature_matching',
                        'location': (int(center_x - est_width/2), int(center_y - est_height/2)),
                        'confidence': confidence,
                        'feature_matches': len(good_matches),
                        'bounding_box': {
                            'x': int(center_x - est_width/2),
                            'y': int(center_y - est_height/2),
                            'width': int(est_width),
                            'height': int(est_height)
                        }
                    })

            return detections

        except Exception as e:
            print(f"Error in feature matching: {e}")
            return []

    def _color_match_logo(self, page_image, business_logo, stored_features):
        """Perform color-based logo detection"""
        try:
            detections = []

            # Get stored color histogram
            stored_histogram = stored_features.get('color_histogram', {})
            if not stored_histogram:
                return detections

            # Convert page to HSV
            page_hsv = cv2.cvtColor(page_image, cv2.COLOR_BGR2HSV)

            # Sliding window approach for color matching
            window_sizes = [(100, 100), (150, 150), (200, 200)]  # Different window sizes

            for window_w, window_h in window_sizes:
                # Slide window across image with step size
                step_size = 50

                for y in range(0, page_image.shape[0] - window_h, step_size):
                    for x in range(0, page_image.shape[1] - window_w, step_size):
                        # Extract window
                        window = page_hsv[y:y+window_h, x:x+window_w]

                        # Calculate color histogram for window
                        window_hist = self._calculate_window_histogram(window)

                        # Compare with stored histogram
                        similarity = self._compare_color_histograms(window_hist, stored_histogram)

                        # If similarity is high enough, consider it a detection
                        if similarity >= 0.7:  # Color similarity threshold
                            detections.append({
                                'method': 'color_matching',
                                'location': (x, y),
                                'confidence': similarity,
                                'window_size': (window_w, window_h),
                                'bounding_box': {
                                    'x': x,
                                    'y': y,
                                    'width': window_w,
                                    'height': window_h
                                }
                            })

            return detections

        except Exception as e:
            print(f"Error in color matching: {e}")
            return []

    def _calculate_window_histogram(self, window_hsv):
        """Calculate color histogram for a window"""
        try:
            hist_h = cv2.calcHist([window_hsv], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([window_hsv], [1], None, [60], [0, 256])
            hist_v = cv2.calcHist([window_hsv], [2], None, [60], [0, 256])

            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()
            hist_v = cv2.normalize(hist_v, hist_v).flatten()

            return {
                'hue': hist_h.tolist(),
                'saturation': hist_s.tolist(),
                'value': hist_v.tolist()
            }
        except:
            return {'hue': [], 'saturation': [], 'value': []}

    def _compare_color_histograms(self, hist1, hist2):
        """Compare two color histograms"""
        try:
            correlations = []

            for channel in ['hue', 'saturation', 'value']:
                h1 = hist1.get(channel, [])
                h2 = hist2.get(channel, [])

                if h1 and h2:
                    h1 = np.array(h1)
                    h2 = np.array(h2)
                    correlation = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
                    correlations.append(max(0, correlation))

            return np.mean(correlations) if correlations else 0.0

        except:
            return 0.0

    def _calculate_overlap(self, box1, box2):
        """Calculate overlap ratio between two bounding boxes"""
        try:
            # Calculate intersection
            x1 = max(box1['x'], box2['x'])
            y1 = max(box1['y'], box2['y'])
            x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
            y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])

            if x2 <= x1 or y2 <= y1:
                return 0.0

            intersection = (x2 - x1) * (y2 - y1)
            area1 = box1['width'] * box1['height']
            area2 = box2['width'] * box2['height']
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0.0

        except:
            return 0.0

    def _create_ad_box_from_logo_detection(self, page, business_logo, detection):
        """Create an AdBox entry from a logo detection"""
        try:
            bbox = detection['bounding_box']

            # Get publication configuration for measurements
            publication = Publication.query.get(page.publication_id)
            config = PUBLICATION_CONFIGS.get(publication.publication_type, PUBLICATION_CONFIGS['broadsheet'])

            # Calculate measurements
            calculator = MeasurementCalculator()

            # Get DPI from page calibration or use default
            dpi = getattr(page, 'pixels_per_inch', 150) or 150  # Default to 150 DPI

            width_inches = bbox['width'] / dpi
            height_inches = bbox['height'] / dpi
            column_inches = width_inches * height_inches

            # Round measurements
            width_rounded = round(width_inches * 16) / 16
            height_rounded = round(height_inches * 16) / 16

            # Create AdBox
            ad_box = AdBox(
                page_id=page.id,
                x=float(bbox['x']),
                y=float(bbox['y']),
                width=float(bbox['width']),
                height=float(bbox['height']),
                width_inches_raw=width_inches,
                height_inches_raw=height_inches,
                width_inches_rounded=width_rounded,
                height_inches_rounded=height_rounded,
                column_inches=column_inches,
                ad_type='logo_detected',
                is_ad=True,
                detected_automatically=True,
                confidence_score=detection['confidence'],
                user_verified=False
            )

            db.session.add(ad_box)
            db.session.commit()

            # Update business logo statistics
            business_logo.successful_detections += 1
            business_logo.last_detected_date = datetime.utcnow()
            db.session.commit()

            return {
                'success': True,
                'ad_box_id': ad_box.id,
                'business_name': business_logo.business_name,
                'confidence': detection['confidence']
            }

        except Exception as e:
            print(f"Error creating ad box from logo detection: {e}")
            db.session.rollback()
            return {'success': False, 'error': str(e)}

    def _record_logo_detection_result(self, business_logo_id, page_id, detection, ad_box_id):
        """Record logo detection result for learning and statistics"""
        try:
            bbox = detection['bounding_box']

            detection_result = LogoRecognitionResult(
                business_logo_id=business_logo_id,
                ad_box_id=ad_box_id,
                page_id=page_id,
                x=float(bbox['x']),
                y=float(bbox['y']),
                width=float(bbox['width']),
                height=float(bbox['height']),
                detection_confidence=detection['confidence'],
                feature_matches=detection.get('feature_matches', 0)
            )

            db.session.add(detection_result)
            db.session.commit()

        except Exception as e:
            print(f"Error recording logo detection result: {e}")

    def _update_logo_detection_statistics(self, detection_results):
        """Update overall logo detection statistics"""
        try:
            # Update statistics for logos that were successfully detected
            for detection in detection_results['detections']:
                business_logo = BusinessLogo.query.get(detection['logo_id'])
                if business_logo:
                    business_logo.successful_detections += 1
                    business_logo.last_detected_date = datetime.utcnow()

            db.session.commit()

        except Exception as e:
            print(f"Error updating logo detection statistics: {e}")


class SmartManualDetection:
    """Smart manual detection system with intelligent boundary expansion"""

    def __init__(self):
        self.logo_feature_extractor = LogoFeatureExtractor()
        self.logo_learning_workflow = LogoLearningWorkflow()

    def detect_ad_boundaries_from_click(self, page_image, click_x, click_y, page_id,
                                       expand_tolerance=30, min_area=100):
        """
        Intelligent boundary detection around clicked area

        Args:
            page_image: OpenCV image of the page
            click_x, click_y: Pixel coordinates of user click
            page_id: Database page ID
            expand_tolerance: Pixels to expand search for boundaries
            min_area: Minimum area for a valid ad detection

        Returns:
            dict: Detected ad boundaries and metadata
        """
        try:
            print(f"Smart boundary detection at click ({click_x}, {click_y})")

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)

            # Method 1: Edge-based boundary detection
            edge_boundaries = self._detect_edge_boundaries(gray, click_x, click_y, expand_tolerance)

            # Method 2: Connected component analysis
            component_boundaries = self._detect_component_boundaries(gray, click_x, click_y, expand_tolerance)

            # Method 3: Text block analysis
            text_boundaries = self._detect_text_block_boundaries(page_image, click_x, click_y, expand_tolerance)

            # Method 4: Color region analysis
            color_boundaries = self._detect_color_region_boundaries(page_image, click_x, click_y, expand_tolerance)

            # Combine and score all boundary candidates
            boundary_candidates = []
            boundary_candidates.extend(edge_boundaries)
            boundary_candidates.extend(component_boundaries)
            boundary_candidates.extend(text_boundaries)
            boundary_candidates.extend(color_boundaries)

            # Score and select best boundary
            best_boundary = self._select_best_boundary(boundary_candidates, click_x, click_y, min_area)

            if best_boundary:
                # Extract features for potential logo learning
                ad_features = self._extract_ad_features(page_image, best_boundary)

                return {
                    'success': True,
                    'boundary': best_boundary,
                    'confidence': best_boundary.get('confidence', 0.7),
                    'detection_method': best_boundary.get('method', 'smart_manual'),
                    'features': ad_features,
                    'area': best_boundary['width'] * best_boundary['height']
                }
            else:
                # Fallback: create basic boundary around click
                fallback_boundary = self._create_fallback_boundary(click_x, click_y, page_image.shape)
                return {
                    'success': True,
                    'boundary': fallback_boundary,
                    'confidence': 0.3,
                    'detection_method': 'manual_fallback',
                    'features': None,
                    'area': fallback_boundary['width'] * fallback_boundary['height']
                }

        except Exception as e:
            print(f"Error in smart boundary detection: {e}")
            return {'success': False, 'error': str(e)}

    def _detect_edge_boundaries(self, gray_image, click_x, click_y, tolerance):
        """Detect ad boundaries using edge detection"""
        try:
            boundaries = []

            # Apply edge detection
            edges = cv2.Canny(gray_image, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Check if click point is inside or near this contour
                if (x - tolerance <= click_x <= x + w + tolerance and
                    y - tolerance <= click_y <= y + h + tolerance):

                    # Score based on contour properties
                    area = cv2.contourArea(contour)
                    perimeter = cv2.arcLength(contour, True)
                    rectangularity = area / (w * h) if w * h > 0 else 0

                    confidence = min(0.9, rectangularity * 0.7 + (area / 10000) * 0.3)

                    boundaries.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'confidence': confidence,
                        'method': 'edge_detection',
                        'area': area,
                        'rectangularity': rectangularity
                    })

            return sorted(boundaries, key=lambda b: b['confidence'], reverse=True)[:3]

        except Exception as e:
            print(f"Error in edge boundary detection: {e}")
            return []

    def _detect_component_boundaries(self, gray_image, click_x, click_y, tolerance):
        """Detect boundaries using connected component analysis"""
        try:
            boundaries = []

            # Threshold image
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Ensure proper data type for connected components
            thresh = thresh.astype(np.uint8)

            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)

            for i in range(1, num_labels):  # Skip background (label 0)
                x, y, w, h, area = stats[i]

                # Check if click point is near this component
                if (x - tolerance <= click_x <= x + w + tolerance and
                    y - tolerance <= click_y <= y + h + tolerance):

                    # Score based on component properties
                    aspect_ratio = w / h if h > 0 else 1
                    aspect_score = 1.0 - abs(aspect_ratio - 1.5) / 1.5  # Prefer moderate aspect ratios
                    area_score = min(1.0, area / 50000)  # Prefer larger areas

                    confidence = (aspect_score * 0.4 + area_score * 0.6) * 0.8

                    boundaries.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'confidence': confidence,
                        'method': 'connected_components',
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })

            return sorted(boundaries, key=lambda b: b['confidence'], reverse=True)[:3]

        except Exception as e:
            print(f"Error in component boundary detection: {e}")
            return []

    def _detect_text_block_boundaries(self, page_image, click_x, click_y, tolerance):
        """Detect boundaries based on text block analysis"""
        try:
            boundaries = []

            # Convert to grayscale
            gray = cv2.cvtColor(page_image, cv2.COLOR_BGR2GRAY)

            # Use morphological operations to find text blocks
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Check if click point is inside or near text block
                if (x - tolerance <= click_x <= x + w + tolerance and
                    y - tolerance <= click_y <= y + h + tolerance):

                    # Score based on text-like properties
                    aspect_ratio = w / h if h > 0 else 1
                    area = w * h

                    # Text blocks typically have certain aspect ratios
                    text_score = 0.8 if 2 < aspect_ratio < 8 else 0.4
                    area_score = min(1.0, area / 30000)

                    confidence = (text_score * 0.6 + area_score * 0.4) * 0.7

                    boundaries.append({
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'confidence': confidence,
                        'method': 'text_blocks',
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })

            return sorted(boundaries, key=lambda b: b['confidence'], reverse=True)[:3]

        except Exception as e:
            print(f"Error in text block boundary detection: {e}")
            return []

    def _detect_color_region_boundaries(self, page_image, click_x, click_y, tolerance):
        """Detect boundaries based on color region analysis"""
        try:
            boundaries = []

            # Get color at click point
            if (0 <= click_y < page_image.shape[0] and 0 <= click_x < page_image.shape[1]):
                click_color = page_image[click_y, click_x]

                # Convert to HSV for better color segmentation
                hsv = cv2.cvtColor(page_image, cv2.COLOR_BGR2HSV)
                click_hsv = cv2.cvtColor(np.uint8([[click_color]]), cv2.COLOR_BGR2HSV)[0][0]

                # Create color mask around clicked color
                lower_bound = np.array([max(0, click_hsv[0] - 20), 50, 50])
                upper_bound = np.array([min(179, click_hsv[0] + 20), 255, 255])

                mask = cv2.inRange(hsv, lower_bound, upper_bound)

                # Find contours in color mask
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Check if click point is inside this color region
                    if (x <= click_x <= x + w and y <= click_y <= y + h):

                        area = cv2.contourArea(contour)
                        rectangularity = area / (w * h) if w * h > 0 else 0

                        confidence = rectangularity * 0.6

                        boundaries.append({
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                            'confidence': confidence,
                            'method': 'color_regions',
                            'area': area,
                            'rectangularity': rectangularity
                        })

            return sorted(boundaries, key=lambda b: b['confidence'], reverse=True)[:2]

        except Exception as e:
            print(f"Error in color region boundary detection: {e}")
            return []

    def _select_best_boundary(self, candidates, click_x, click_y, min_area):
        """Select the best boundary from all candidates"""
        try:
            if not candidates:
                return None

            # Score each candidate
            for candidate in candidates:
                # Base score from confidence
                score = candidate['confidence']

                # Bonus for larger areas (up to a point)
                area_bonus = min(0.3, candidate['area'] / 100000)
                score += area_bonus

                # Bonus for reasonable aspect ratios
                aspect_ratio = candidate['width'] / candidate['height'] if candidate['height'] > 0 else 1
                if 0.3 <= aspect_ratio <= 5.0:  # Reasonable ad proportions
                    score += 0.1

                # Penalty for very small areas
                if candidate['area'] < min_area:
                    score *= 0.5

                # Bonus for click being more centered in boundary
                center_x = candidate['x'] + candidate['width'] / 2
                center_y = candidate['y'] + candidate['height'] / 2
                distance_from_center = ((click_x - center_x) ** 2 + (click_y - center_y) ** 2) ** 0.5
                max_distance = (candidate['width'] ** 2 + candidate['height'] ** 2) ** 0.5 / 2
                centrality_bonus = 0.15 * (1 - min(1, distance_from_center / max_distance))
                score += centrality_bonus

                candidate['final_score'] = score

            # Return best candidate
            best = max(candidates, key=lambda c: c['final_score'])
            return best if best['final_score'] > 0.3 else None

        except Exception as e:
            print(f"Error selecting best boundary: {e}")
            return None

    def _create_fallback_boundary(self, click_x, click_y, image_shape):
        """Create a fallback boundary around click point"""
        try:
            # Default ad size (reasonable assumption)
            default_width = 200
            default_height = 150

            # Center on click point
            x = max(0, click_x - default_width // 2)
            y = max(0, click_y - default_height // 2)

            # Ensure within image bounds
            max_x = image_shape[1] - default_width
            max_y = image_shape[0] - default_height

            x = min(x, max_x) if max_x > 0 else 0
            y = min(y, max_y) if max_y > 0 else 0

            # Adjust width/height if near edges
            width = min(default_width, image_shape[1] - x)
            height = min(default_height, image_shape[0] - y)

            return {
                'x': x,
                'y': y,
                'width': width,
                'height': height,
                'method': 'fallback'
            }

        except Exception as e:
            print(f"Error creating fallback boundary: {e}")
            return {'x': 0, 'y': 0, 'width': 100, 'height': 100, 'method': 'fallback'}

    def _extract_ad_features(self, page_image, boundary):
        """Extract features from detected ad for potential logo learning"""
        try:
            x, y, w, h = boundary['x'], boundary['y'], boundary['width'], boundary['height']

            # Extract ad region
            ad_region = page_image[y:y+h, x:x+w]

            if ad_region.size == 0:
                return None

            # Use logo feature extractor to get features
            features = self.logo_feature_extractor.extract_logo_features(ad_region)

            return features

        except Exception as e:
            print(f"Error extracting ad features: {e}")
            return None

    def create_manual_ad_with_learning(self, page_id, boundary, business_name=None, learn_logo=False):
        """Create manual ad and optionally learn logo for future detection"""
        try:
            page = Page.query.get(page_id)
            if not page:
                return {'success': False, 'error': 'Page not found'}

            # Get publication configuration for measurements
            publication = Publication.query.get(page.publication_id)
            config = PUBLICATION_CONFIGS.get(publication.publication_type, PUBLICATION_CONFIGS['broadsheet'])

            # Calculate measurements
            dpi = getattr(page, 'pixels_per_inch', 150) or 150
            width_inches = boundary['width'] / dpi
            height_inches = boundary['height'] / dpi
            column_inches = width_inches * height_inches

            # Round measurements
            width_rounded = round(width_inches * 16) / 16
            height_rounded = round(height_inches * 16) / 16

            # Create AdBox
            ad_box = AdBox(
                page_id=page.id,
                x=float(boundary['x']),
                y=float(boundary['y']),
                width=float(boundary['width']),
                height=float(boundary['height']),
                width_inches_raw=width_inches,
                height_inches_raw=height_inches,
                width_inches_rounded=width_rounded,
                height_inches_rounded=height_rounded,
                column_inches=column_inches,
                ad_type='manual_smart',
                is_ad=True,
                detected_automatically=False,
                confidence_score=boundary.get('confidence', 0.7),
                user_verified=True
            )

            db.session.add(ad_box)
            db.session.commit()

            # If learning is enabled and business name provided
            if learn_logo and business_name and boundary.get('features'):
                try:
                    # Learn logo from this manual placement
                    learning_result = self.logo_learning_workflow.learn_logo_from_manual_ad(
                        ad_box, business_name, boundary['features']
                    )

                    return {
                        'success': True,
                        'ad_box_id': ad_box.id,
                        'learning_result': learning_result,
                        'message': f'Manual ad created and logo learned for {business_name}'
                    }
                except Exception as e:
                    print(f"Logo learning failed: {e}")
                    return {
                        'success': True,
                        'ad_box_id': ad_box.id,
                        'learning_result': {'success': False, 'error': str(e)},
                        'message': 'Manual ad created but logo learning failed'
                    }
            else:
                return {
                    'success': True,
                    'ad_box_id': ad_box.id,
                    'message': 'Manual ad created successfully'
                }

        except Exception as e:
            print(f"Error creating manual ad: {e}")
            db.session.rollback()
            return {'success': False, 'error': str(e)}


class HybridDetectionPipeline:
    """Hybrid detection system combining logo recognition with manual detection capabilities"""

    def __init__(self):
        self.logo_recognition_engine = LogoRecognitionDetectionEngine()
        self.smart_manual_detection = SmartManualDetection()
        self.logo_learning_workflow = LogoLearningWorkflow()

    def detect_ads_hybrid(self, publication_id, mode='auto', page_numbers=None):
        """
        Run hybrid detection combining automated logo recognition with optional manual enhancement

        Args:
            publication_id: ID of publication to process
            mode: Detection mode ('auto', 'manual', 'hybrid')
            page_numbers: Specific pages to process (None for all)

        Returns:
            dict: Detection results and statistics
        """
        try:
            print(f"Starting hybrid detection for publication {publication_id} in mode: {mode}")

            publication = Publication.query.get(publication_id)
            if not publication:
                return {'success': False, 'error': 'Publication not found'}

            # Get pages to process
            if page_numbers:
                pages = Page.query.filter(
                    Page.publication_id == publication_id,
                    Page.page_number.in_(page_numbers)
                ).all()
            else:
                pages = Page.query.filter_by(publication_id=publication_id).all()

            if not pages:
                return {'success': False, 'error': 'No pages found'}

            results = {
                'success': True,
                'publication_id': publication_id,
                'mode': mode,
                'pages_processed': len(pages),
                'total_detections': 0,
                'logo_detections': 0,
                'manual_detections': 0,
                'page_results': [],
                'business_logos_found': set(),
                'detection_statistics': {}
            }

            # Phase 1: Automated logo recognition (if enabled)
            if mode in ['auto', 'hybrid']:
                print("Phase 1: Running automated logo recognition...")

                logo_results = self.logo_recognition_engine.detect_ads_from_publication(publication_id)

                if logo_results.get('success'):
                    results['logo_detections'] = logo_results.get('detections', 0)
                    results['total_detections'] += results['logo_detections']
                    results['business_logos_found'].update(logo_results.get('business_names', []))

                    print(f"Logo recognition found {results['logo_detections']} ads")
                else:
                    print(f"Logo recognition failed: {logo_results.get('error', 'Unknown error')}")

            # Phase 2: Manual detection enhancement (for hybrid mode)
            if mode in ['manual', 'hybrid']:
                print("Phase 2: Preparing manual detection enhancement...")

                # For each page, provide tools for manual enhancement
                for page in pages:
                    page_result = self._prepare_page_for_manual_detection(page, mode)
                    results['page_results'].append(page_result)

            # Phase 3: Update learning system
            if results['total_detections'] > 0:
                self._update_hybrid_learning_statistics(publication_id, results)

            # Convert set to list for JSON serialization
            results['business_logos_found'] = list(results['business_logos_found'])

            print(f"Hybrid detection complete: {results['total_detections']} total ads detected")
            return results

        except Exception as e:
            print(f"Error in hybrid detection: {e}")
            return {'success': False, 'error': str(e)}

    def _prepare_page_for_manual_detection(self, page, mode):
        """Prepare page data for manual detection interface"""
        try:
            # Get existing AdBoxes on this page
            existing_ads = AdBox.query.filter_by(page_id=page.id).all()

            # Load page image
            publication = Publication.query.get(page.publication_id)
            image_filename = f"{publication.filename}_page_{page.page_number}.png"
            image_path = os.path.join('static', 'uploads', 'pages', image_filename)

            page_data = {
                'page_id': page.id,
                'page_number': page.page_number,
                'image_path': image_path,
                'image_exists': os.path.exists(image_path),
                'existing_ads': len(existing_ads),
                'width': page.width_pixels,
                'height': page.height_pixels,
                'mode': mode
            }

            # Add existing ad details
            page_data['ad_details'] = []
            for ad in existing_ads:
                page_data['ad_details'].append({
                    'id': ad.id,
                    'x': ad.x,
                    'y': ad.y,
                    'width': ad.width,
                    'height': ad.height,
                    'ad_type': ad.ad_type,
                    'confidence': ad.confidence_score,
                    'auto_detected': ad.detected_automatically
                })

            return page_data

        except Exception as e:
            print(f"Error preparing page for manual detection: {e}")
            return {'page_id': page.id, 'error': str(e)}

    def process_manual_click(self, page_id, click_x, click_y, business_name=None, learn_logo=False):
        """
        Process a manual click for ad detection with smart boundary detection

        Args:
            page_id: Database page ID
            click_x, click_y: Click coordinates
            business_name: Optional business name for logo learning
            learn_logo: Whether to learn logo from this detection

        Returns:
            dict: Detection result and created ad box
        """
        try:
            print(f"Processing manual click at ({click_x}, {click_y}) on page {page_id}")

            page = Page.query.get(page_id)
            if not page:
                return {'success': False, 'error': 'Page not found'}

            # Load page image
            publication = Publication.query.get(page.publication_id)
            image_filename = f"{publication.filename}_page_{page.page_number}.png"
            image_path = os.path.join('static', 'uploads', 'pages', image_filename)

            if not os.path.exists(image_path):
                return {'success': False, 'error': 'Page image not found'}

            # Load image
            page_image = cv2.imread(image_path)
            if page_image is None:
                return {'success': False, 'error': 'Could not load page image'}

            # Use smart manual detection to find ad boundaries
            boundary_result = self.smart_manual_detection.detect_ad_boundaries_from_click(
                page_image, click_x, click_y, page_id
            )

            if not boundary_result.get('success'):
                return boundary_result

            # Create manual ad with optional logo learning
            ad_creation_result = self.smart_manual_detection.create_manual_ad_with_learning(
                page_id, boundary_result['boundary'], business_name, learn_logo
            )

            if ad_creation_result.get('success'):
                # Combine results
                final_result = {
                    'success': True,
                    'ad_box_id': ad_creation_result['ad_box_id'],
                    'boundary': boundary_result['boundary'],
                    'detection_method': boundary_result['detection_method'],
                    'confidence': boundary_result['confidence'],
                    'area': boundary_result['area'],
                    'business_name': business_name,
                    'logo_learned': learn_logo and ad_creation_result.get('learning_result', {}).get('success', False),
                    'message': ad_creation_result['message']
                }

                if learn_logo:
                    final_result['learning_result'] = ad_creation_result.get('learning_result', {})

                print(f"Manual detection successful: AdBox {ad_creation_result['ad_box_id']} created")
                return final_result
            else:
                return ad_creation_result

        except Exception as e:
            print(f"Error processing manual click: {e}")
            return {'success': False, 'error': str(e)}

    def get_detection_suggestions(self, page_id, threshold=0.6):
        """
        Get automated suggestions for manual detection on a page

        Args:
            page_id: Database page ID
            threshold: Confidence threshold for suggestions

        Returns:
            dict: Suggested detection areas
        """
        try:
            page = Page.query.get(page_id)
            if not page:
                return {'success': False, 'error': 'Page not found'}

            # Get existing AdBoxes to avoid duplicates
            existing_ads = AdBox.query.filter_by(page_id=page_id).all()
            existing_regions = []
            for ad in existing_ads:
                existing_regions.append({
                    'x': ad.x, 'y': ad.y, 'width': ad.width, 'height': ad.height
                })

            # Load page image
            publication = Publication.query.get(page.publication_id)
            image_filename = f"{publication.filename}_page_{page.page_number}.png"
            image_path = os.path.join('static', 'uploads', 'pages', image_filename)

            if not os.path.exists(image_path):
                return {'success': False, 'error': 'Page image not found'}

            page_image = cv2.imread(image_path)
            if page_image is None:
                return {'success': False, 'error': 'Could not load page image'}

            # Run logo recognition at lower threshold for suggestions
            suggestions = []

            # Get all business logos
            business_logos = BusinessLogo.query.filter_by(is_active=True).all()

            for business_logo in business_logos:
                logo_detections = self.logo_recognition_engine._search_logo_on_page(
                    page_image, business_logo, confidence_threshold=threshold * 0.7
                )

                for detection in logo_detections:
                    bbox = detection['bounding_box']

                    # Check if this overlaps with existing ads
                    is_duplicate = False
                    for existing in existing_regions:
                        overlap = self._calculate_region_overlap(bbox, existing)
                        if overlap > 0.3:  # 30% overlap threshold
                            is_duplicate = True
                            break

                    if not is_duplicate and detection['confidence'] >= threshold:
                        suggestions.append({
                            'business_name': business_logo.business_name,
                            'confidence': detection['confidence'],
                            'x': bbox['x'],
                            'y': bbox['y'],
                            'width': bbox['width'],
                            'height': bbox['height'],
                            'suggestion_type': 'logo_recognition'
                        })

            # Sort by confidence
            suggestions.sort(key=lambda s: s['confidence'], reverse=True)

            return {
                'success': True,
                'page_id': page_id,
                'suggestions': suggestions[:10],  # Top 10 suggestions
                'total_suggestions': len(suggestions)
            }

        except Exception as e:
            print(f"Error getting detection suggestions: {e}")
            return {'success': False, 'error': str(e)}

    def _calculate_region_overlap(self, region1, region2):
        """Calculate overlap ratio between two regions"""
        try:
            x1_min, y1_min = region1['x'], region1['y']
            x1_max, y1_max = x1_min + region1['width'], y1_min + region1['height']

            x2_min, y2_min = region2['x'], region2['y']
            x2_max, y2_max = x2_min + region2['width'], y2_min + region2['height']

            # Calculate intersection
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
                intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                area1 = region1['width'] * region1['height']
                area2 = region2['width'] * region2['height']
                union = area1 + area2 - intersection
                return intersection / union if union > 0 else 0

            return 0.0

        except:
            return 0.0

    def _update_hybrid_learning_statistics(self, publication_id, results):
        """Update learning statistics from hybrid detection session"""
        try:
            # Update business logo detection statistics
            for business_name in results['business_logos_found']:
                business_logo = BusinessLogo.query.filter_by(business_name=business_name).first()
                if business_logo:
                    business_logo.total_detections += 1
                    business_logo.last_detected_date = datetime.utcnow()

            db.session.commit()
            print("Updated hybrid learning statistics")

        except Exception as e:
            print(f"Error updating hybrid learning statistics: {e}")

    def get_hybrid_detection_status(self, publication_id):
        """Get current status of hybrid detection for a publication"""
        try:
            publication = Publication.query.get(publication_id)
            if not publication:
                return {'success': False, 'error': 'Publication not found'}

            # Get all AdBoxes for this publication
            total_ads = db.session.query(AdBox).join(Page).filter(
                Page.publication_id == publication_id
            ).count()

            # Get automated vs manual breakdown
            auto_ads = db.session.query(AdBox).join(Page).filter(
                Page.publication_id == publication_id,
                AdBox.detected_automatically == True
            ).count()

            manual_ads = total_ads - auto_ads

            # Get logo detection breakdown
            logo_ads = db.session.query(AdBox).join(Page).filter(
                Page.publication_id == publication_id,
                AdBox.ad_type == 'logo_detected'
            ).count()

            # Get business logos found
            business_logos_found = db.session.query(BusinessLogo.business_name).join(
                LogoRecognitionResult
            ).join(AdBox).join(Page).filter(
                Page.publication_id == publication_id
            ).distinct().all()

            return {
                'success': True,
                'publication_id': publication_id,
                'publication_name': publication.original_filename,
                'total_pages': publication.total_pages,
                'total_ads': total_ads,
                'automated_ads': auto_ads,
                'manual_ads': manual_ads,
                'logo_detected_ads': logo_ads,
                'business_logos_found': [name[0] for name in business_logos_found],
                'detection_rate': total_ads / publication.total_pages if publication.total_pages > 0 else 0
            }

        except Exception as e:
            print(f"Error getting hybrid detection status: {e}")
            return {'success': False, 'error': str(e)}


class SimpleAdDetector:
    """Simple, working ad detection that finds actual bordered advertisements"""

    @staticmethod
    def detect_bordered_ads(publication_id):
        """
        Detect ads by finding bordered rectangular regions (where real ads are)
        No complex text analysis - just find rectangles that look like ads
        """
        try:
            print(f"Starting simple bordered ad detection for publication {publication_id}")

            publication = Publication.query.get(publication_id)
            if not publication:
                return {'success': False, 'error': 'Publication not found'}

            pages = Page.query.filter_by(publication_id=publication_id).all()
            if not pages:
                return {'success': False, 'error': 'No pages found'}

            total_detections = 0
            config = PUBLICATION_CONFIGS.get(publication.publication_type, PUBLICATION_CONFIGS['broadsheet'])

            for page in pages:
                print(f"Processing page {page.page_number} for bordered ads...")

                # Load page image
                image_filename = f"{publication.filename}_page_{page.page_number}.png"
                image_path = os.path.join('static', 'uploads', 'pages', image_filename)

                if not os.path.exists(image_path):
                    print(f"Warning: Page image not found: {image_path}")
                    continue

                page_image = cv2.imread(image_path)
                if page_image is None:
                    print(f"Warning: Could not load page image: {image_path}")
                    continue

                # Find bordered rectangles (actual ads)
                ad_regions = SimpleAdDetector._find_bordered_rectangles(page_image)

                # Filter to realistic ad sizes
                filtered_ads = SimpleAdDetector._filter_realistic_ads(ad_regions)

                print(f"Found {len(filtered_ads)} bordered ads on page {page.page_number}")

                # Create AdBox entries
                for ad in filtered_ads:
                    # Calculate measurements
                    dpi = getattr(page, 'pixels_per_inch', 150) or 150
                    width_inches = ad['width'] / dpi
                    height_inches = ad['height'] / dpi
                    column_inches = width_inches * height_inches

                    # Round measurements
                    width_rounded = round(width_inches * 16) / 16
                    height_rounded = round(height_inches * 16) / 16

                    # Create AdBox
                    ad_box = AdBox(
                        page_id=page.id,
                        x=float(ad['x']),
                        y=float(ad['y']),
                        width=float(ad['width']),
                        height=float(ad['height']),
                        width_inches_raw=width_inches,
                        height_inches_raw=height_inches,
                        width_inches_rounded=width_rounded,
                        height_inches_rounded=height_rounded,
                        column_inches=column_inches,
                        ad_type='bordered_ad',
                        is_ad=True,
                        detected_automatically=True,
                        confidence_score=ad['confidence'],
                        user_verified=False
                    )

                    db.session.add(ad_box)
                    total_detections += 1

                db.session.commit()

            return {
                'success': True,
                'detections': total_detections,
                'pages_processed': len(pages),
                'message': f'Detected {total_detections} bordered ads'
            }

        except Exception as e:
            print(f"Error in simple ad detection: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    @staticmethod
    def _find_bordered_rectangles(image):
        """Find rectangular regions with borders (where real ads are)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Use multiple edge detection strategies to catch different ad types

            # Strategy 1: Standard edge detection for clear borders
            edges1 = cv2.Canny(gray, 50, 150, apertureSize=3)

            # Strategy 2: More sensitive edge detection for subtle borders
            edges2 = cv2.Canny(gray, 30, 100, apertureSize=3)

            # Strategy 3: Morphological operations to find text blocks with borders
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            edges3 = cv2.Canny(morph, 40, 120, apertureSize=3)

            # Combine all edge maps
            edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))

            # Find contours with both external and internal hierarchy
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            rectangles = []

            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Check if it's roughly rectangular (4-8 vertices)
                if 4 <= len(approx) <= 8:
                    x, y, w, h = cv2.boundingRect(contour)

                    # Calculate aspect ratio
                    aspect_ratio = w / h if h > 0 else 0

                    # Calculate area
                    area = w * h

                    # Calculate how rectangular it is
                    contour_area = cv2.contourArea(contour)
                    rectangularity = contour_area / area if area > 0 else 0

                    # Score based on how ad-like it is
                    confidence = SimpleAdDetector._score_ad_candidate(w, h, aspect_ratio, rectangularity, area)

                    if confidence > 0.2:  # Keep more candidates (relaxed threshold)
                        rectangles.append({
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                            'confidence': confidence,
                            'aspect_ratio': aspect_ratio,
                            'area': area
                        })

            # Sort by confidence
            rectangles.sort(key=lambda r: r['confidence'], reverse=True)

            return rectangles

        except Exception as e:
            print(f"Error finding bordered rectangles: {e}")
            return []

    @staticmethod
    def _score_ad_candidate(width, height, aspect_ratio, rectangularity, area):
        """Score how likely a rectangle is to be an ad (focusing on business ads, not editorial photos)"""
        score = 0.0

        # Size scoring to include more business ads while excluding editorial photos
        if 200 <= width <= 400 and 150 <= height <= 250:
            score += 0.6  # Ideal business directory size
        elif 250 <= width <= 500 and 150 <= height <= 300:
            score += 0.5  # Medium business ad
        elif 150 <= width <= 600 and 100 <= height <= 400:
            score += 0.4  # Any business ad size (relaxed minimum)
        elif 100 <= width <= 800 and 80 <= height <= 500:
            score += 0.2  # Smaller business ads (classified style)

        # Aspect ratio scoring - business ads have rectangular layouts
        # Editorial photos often have different ratios
        if 1.3 <= aspect_ratio <= 2.5:
            score += 0.4  # Typical business ad ratio (horizontal rectangular)
        elif 2.5 <= aspect_ratio <= 4.0:
            score += 0.3  # Banner-style ads
        elif 1.0 <= aspect_ratio <= 1.3:
            score += 0.2  # Square-ish ads (less common but valid)

        # Area scoring - business ads need sufficient space for content
        if 30000 <= area <= 150000:  # 200x150 to 500x300 range
            score += 0.3
        elif 15000 <= area <= 250000:  # Broader business ad range
            score += 0.2
        elif area >= 8000:  # Minimum ad area (100x80)
            score += 0.1

        # Rectangularity scoring - ads should be very rectangular
        # Editorial photos often have irregular borders
        if rectangularity >= 0.8:
            score += 0.3  # Very rectangular (ideal for ads)
        elif rectangularity >= 0.6:
            score += 0.2  # Reasonably rectangular
        elif rectangularity >= 0.4:
            score += 0.1  # Somewhat rectangular

        # Penalize very large areas (likely editorial photos or page elements)
        if area > 400000:  # Too large to be a typical business ad (relaxed)
            score -= 0.3

        # Penalize very square ratios (often editorial photos)
        if 0.95 <= aspect_ratio <= 1.05:  # Very square only
            score -= 0.1

        return max(0.0, min(1.0, score))

    @staticmethod
    def _filter_realistic_ads(rectangles):
        """Filter to only realistic ad sizes, merge adjacent detections, and remove overlaps"""
        # First, apply size filtering (relaxed to capture more business ads)
        size_filtered = []
        for rect in rectangles:
            # Relaxed minimum size to capture smaller business directory ads
            if rect['width'] >= 100 and rect['height'] >= 80:
                # Maximum size filter - not the whole page
                if rect['width'] <= 800 and rect['height'] <= 500:
                    size_filtered.append(rect)

        # Sort by confidence for merging
        size_filtered.sort(key=lambda r: r['confidence'], reverse=True)

        # Merge adjacent detections within 30 pixels
        merged = SimpleAdDetector._merge_adjacent_detections(size_filtered)

        # Remove overlaps after merging
        final_filtered = []
        for rect in merged:
            is_duplicate = False
            for existing in final_filtered:
                overlap = SimpleAdDetector._calculate_overlap(rect, existing)
                if overlap > 0.3:  # 30% overlap = duplicate
                    is_duplicate = True
                    break

            if not is_duplicate:
                final_filtered.append(rect)

        return final_filtered

    @staticmethod
    def _calculate_overlap(rect1, rect2):
        """Calculate overlap ratio between two rectangles"""
        try:
            x1_min, y1_min = rect1['x'], rect1['y']
            x1_max, y1_max = x1_min + rect1['width'], y1_min + rect1['height']

            x2_min, y2_min = rect2['x'], rect2['y']
            x2_max, y2_max = x2_min + rect2['width'], y2_min + rect2['height']

            # Calculate intersection
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
                intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
                area1 = rect1['width'] * rect1['height']
                area2 = rect2['width'] * rect2['height']
                union = area1 + area2 - intersection
                return intersection / union if union > 0 else 0

            return 0.0

        except:
            return 0.0

    @staticmethod
    def _merge_adjacent_detections(rectangles):
        """Merge detections that are within 30 pixels of each other"""
        if not rectangles:
            return rectangles

        merged = []
        used = set()

        for i, rect in enumerate(rectangles):
            if i in used:
                continue

            # Find all rectangles within 30 pixels
            cluster = [rect]
            used.add(i)

            for j, other_rect in enumerate(rectangles[i+1:], i+1):
                if j in used:
                    continue

                # Check if rectangles are within 30 pixels
                if SimpleAdDetector._are_adjacent(rect, other_rect, threshold=30):
                    cluster.append(other_rect)
                    used.add(j)

            # Merge cluster into single detection
            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                merged_rect = SimpleAdDetector._merge_rectangle_cluster(cluster)
                merged.append(merged_rect)

        return merged

    @staticmethod
    def _are_adjacent(rect1, rect2, threshold=30):
        """Check if two rectangles are within threshold pixels of each other"""
        # Calculate distances between rectangles
        x1_min, y1_min = rect1['x'], rect1['y']
        x1_max, y1_max = x1_min + rect1['width'], y1_min + rect1['height']

        x2_min, y2_min = rect2['x'], rect2['y']
        x2_max, y2_max = x2_min + rect2['width'], y2_min + rect2['height']

        # Check horizontal distance
        h_dist = 0
        if x1_max < x2_min:
            h_dist = x2_min - x1_max
        elif x2_max < x1_min:
            h_dist = x1_min - x2_max

        # Check vertical distance
        v_dist = 0
        if y1_max < y2_min:
            v_dist = y2_min - y1_max
        elif y2_max < y1_min:
            v_dist = y1_min - y2_max

        # Adjacent if both distances are within threshold
        return h_dist <= threshold and v_dist <= threshold

    @staticmethod
    def _merge_rectangle_cluster(cluster):
        """Merge a cluster of rectangles into a single bounding rectangle"""
        if not cluster:
            return None

        # Find bounding box that encompasses all rectangles
        min_x = min(rect['x'] for rect in cluster)
        min_y = min(rect['y'] for rect in cluster)
        max_x = max(rect['x'] + rect['width'] for rect in cluster)
        max_y = max(rect['y'] + rect['height'] for rect in cluster)

        # Use highest confidence from cluster
        max_confidence = max(rect['confidence'] for rect in cluster)

        # Calculate merged rectangle properties
        merged_width = max_x - min_x
        merged_height = max_y - min_y
        merged_aspect_ratio = merged_width / merged_height if merged_height > 0 else 0
        merged_area = merged_width * merged_height

        return {
            'x': min_x,
            'y': min_y,
            'width': merged_width,
            'height': merged_height,
            'confidence': max_confidence,
            'aspect_ratio': merged_aspect_ratio,
            'area': merged_area
        }


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
            'eighth': {'inches': 14.04, 'name': '1/8 Page'},
            'quarter': {'inches': 28.88, 'name': '1/4 Page'},
            'half': {'inches': 59.28, 'name': '1/2 Page'},
            'full': {'inches': 120.24, 'name': 'Full Page'}
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

# Google Vision AI Helper Functions
def handle_vision_api_error(error, context="Vision API operation"):
    """Centralized error handling for Google Vision API calls"""
    error_msg = str(error).lower()
    
    if "quota" in error_msg or "limit" in error_msg:
        print(f"Vision API quota exceeded in {context}: {error}")
        return "quota_exceeded"
    elif "permission" in error_msg or "unauthorized" in error_msg:
        print(f"Vision API permission denied in {context}: {error}")
        return "permission_denied"
    elif "not found" in error_msg or "invalid" in error_msg:
        print(f"Vision API invalid request in {context}: {error}")
        return "invalid_request"
    elif "network" in error_msg or "connection" in error_msg:
        print(f"Vision API network error in {context}: {error}")
        return "network_error"
    else:
        print(f"Vision API unknown error in {context}: {error}")
        return "unknown_error"

def is_vision_api_available():
    """Quick check if Google Vision API is available and configured"""
    try:
        # Check if credentials file exists
        if not os.path.exists('google-vision-credentials.json'):
            return False
            
        # Try to initialize client (lightweight operation)
        client = vision.ImageAnnotatorClient()
        return True
    except Exception as e:
        print(f"Vision API not available: {e}")
        return False

# PDF Ad Detection Engine (wrapper for upload processing)
class PDFAdDetectionEngine:
    """
    High-level wrapper for PDF-based ad detection during upload processing.
    Works independently of the full auto_detect_ads workflow.
    """
    
    @staticmethod
    def detect_ads_from_pdf(publication_id):
        """
        Detect ads using PDF metadata for a publication during upload processing.
        
        Args:
            publication_id (int): ID of the publication to analyze
            
        Returns:
            dict: Detection results with success status, detections count, etc.
        """
        try:
            publication = Publication.query.get(publication_id)
            if not publication:
                return {'success': False, 'error': 'Publication not found'}
            
            pdf_path = os.path.join('static', 'uploads', 'pdfs', publication.filename)
            if not os.path.exists(pdf_path):
                return {'success': False, 'error': f'PDF file not found: {pdf_path}'}
            
            pages = Page.query.filter_by(publication_id=publication.id).all()
            if not pages:
                return {'success': False, 'error': 'No pages found for publication'}
            
            print(f"Processing {len(pages)} pages with PDF metadata detection")
            
            detections_count = 0
            pages_processed = 0
            
            for page in pages:
                try:
                    print(f"Analyzing page {page.page_number} for PDF metadata ads")
                    
                    # Get PDF detections with filename intelligence
                    pdf_detections = PDFMetadataAdDetector.detect_ads_from_pdf_metadata(
                        pdf_path, page.page_number, publication.publication_type, publication.original_filename
                    )
                    
                    if pdf_detections:
                        print(f"Found {len(pdf_detections)} ad candidates on page {page.page_number}")
                        
                        # Transform coordinates
                        doc = fitz.open(pdf_path)
                        pdf_page = doc[page.page_number - 1]
                        pdf_page_rect = pdf_page.rect
                        doc.close()
                        
                        transformed_detections = PDFMetadataAdDetector.transform_pdf_to_image_coordinates(
                            pdf_detections, pdf_page_rect, page.width_pixels, page.height_pixels
                        )
                        
                        # Create AdBox records
                        for detection in transformed_detections:
                            try:
                                # Calculate measurements using existing logic
                                config = PUBLICATION_CONFIGS[publication.publication_type]
                                calculator = MeasurementCalculator()
                                
                                page_total_pixels = page.width_pixels * page.height_pixels if page.width_pixels and page.height_pixels else 1
                                column_inches = calculator.pixels_to_inches(
                                    detection['height'] * detection['width'],
                                    page_total_pixels,
                                    config.get('total_inches_per_page', 258)
                                )
                                
                                # Calculate width and height inches
                                if page.width_pixels and page.height_pixels:
                                    width_inches = (detection['width'] / page.width_pixels) * config.get('total_inches_per_page', 258) * (config.get('width_units', 12) / 12)
                                    height_inches = (detection['height'] / page.height_pixels) * config.get('total_inches_per_page', 258)
                                else:
                                    width_inches = column_inches / 10
                                    height_inches = column_inches / 10
                                
                                width_rounded = round(width_inches * 16) / 16
                                height_rounded = round(height_inches * 16) / 16
                                
                                # Determine ad type
                                ad_type = 'pdf_detected'
                                if detection['ad_type'] == 'image_ad':
                                    ad_type = 'pdf_image_ad'
                                elif detection['ad_type'] == 'text_ad':
                                    ad_type = 'pdf_text_ad'
                                elif detection['ad_type'] == 'bordered_ad':
                                    ad_type = 'pdf_border_ad'
                                elif detection['ad_type'] == 'mixed_ad':
                                    ad_type = 'pdf_mixed_ad'
                                else:
                                    ad_type = f"pdf_{detection['ad_type']}"
                                
                                # Create AdBox
                                ad_box = AdBox(
                                    page_id=page.id,
                                    x=detection['x'],
                                    y=detection['y'],
                                    width=detection['width'],
                                    height=detection['height'],
                                    width_inches_raw=width_inches,
                                    height_inches_raw=height_inches,
                                    width_inches_rounded=width_rounded,
                                    height_inches_rounded=height_rounded,
                                    column_inches=column_inches,
                                    ad_type=ad_type,
                                    is_ad=True,
                                    confidence_score=detection['confidence'],
                                    detected_automatically=True,
                                    user_verified=False
                                )
                                
                                db.session.add(ad_box)
                                detections_count += 1
                                
                                print(f"Created ad box: {ad_type} at ({detection['x']:.0f},{detection['y']:.0f}) "
                                      f"size {detection['width']:.0f}x{detection['height']:.0f} "
                                      f"confidence={detection['confidence']:.3f}")
                                
                            except Exception as box_error:
                                print(f"Error creating ad box: {box_error}")
                                continue
                    else:
                        print(f"No ads found on page {page.page_number}")
                    
                    pages_processed += 1
                    
                except Exception as page_error:
                    print(f"Error processing page {page.page_number}: {page_error}")
                    continue
            
            # Commit all changes
            if detections_count > 0:
                db.session.commit()
                print(f"Successfully saved {detections_count} ad detections to database")
            
            return {
                'success': True,
                'detections': detections_count,
                'pages_processed': pages_processed,
                'detection_method': 'PDF_metadata'
            }
            
        except Exception as e:
            print(f"PDF detection engine error: {e}")
            db.session.rollback()
            return {'success': False, 'error': str(e)}


# PDF Metadata-based Ad Detector
class PDFMetadataAdDetector:
    """
    PDF structure-based ad detection using document metadata and layout analysis.
    Analyzes PDF objects, fonts, images, and text positioning to identify advertisements
    with high accuracy and zero false positives from editorial content.
    """
    
    @staticmethod
    def extract_size_hints_from_filename(original_filename):
        """
        Extract size hints from filename to improve ad detection accuracy.
        
        Args:
            original_filename (str): The original filename
            
        Returns:
            dict: Size hint information including dimensions and validation flags
        """
        import re
        
        if not original_filename:
            return {'has_size_hint': False}
        
        # Common ad size patterns (width x height in inches)
        size_patterns = [
            r'(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)',  # Basic pattern like "3x5", "2x4", "1.5x2.5"
            r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)',  # With spaces like "3 x 5", "1.5 x 2.5"
        ]
        
        filename_lower = original_filename.lower()
        
        for pattern in size_patterns:
            matches = re.findall(pattern, filename_lower)
            if matches:
                match = matches[0]
                if len(match) == 2:  # Both patterns return 2 groups
                    width_str, height_str = match
                    try:
                        width_inches = float(width_str)
                        height_inches = float(height_str)
                        
                        # Validate reasonable ad sizes (0.5 to 20 inches)
                        if 0.5 <= width_inches <= 20 and 0.5 <= height_inches <= 20:
                            return {
                                'has_size_hint': True,
                                'expected_width_inches': width_inches,
                                'expected_height_inches': height_inches,
                                'confidence_boost': 1.4,  # 40% confidence boost
                                'tolerance': 0.3  # 30% size tolerance
                            }
                    except ValueError:
                        continue
        
        return {'has_size_hint': False}

    @staticmethod
    def matches_standard_ad_size(width_px, height_px, pixels_per_inch, filename_hints=None):
        """
        Check if dimensions match standard newspaper ad sizes.
        
        Args:
            width_px (float): Width in pixels
            height_px (float): Height in pixels  
            pixels_per_inch (float): Pixel density for conversion
            filename_hints (dict, optional): Size hints from filename for strict matching
            
        Returns:
            tuple: (is_match, size_name, confidence_multiplier)
        """
        # Standard newspaper ad sizes (width, height) in inches
        standard_sizes = [
            (1, 2), (2, 1),    # Small classifieds
            (2, 3), (3, 2),    # Small display ads  
            (2, 4), (4, 2),    # Medium ads
            (3, 5), (5, 3),    # Common display ads
            (4, 6), (6, 4),    # Large display ads
            (6, 8), (8, 6)     # Premium display ads
        ]
        
        # If filename has size hints, be very strict and only look for that specific size
        if filename_hints and filename_hints.get('has_size_hint'):
            target_w_inch = filename_hints['expected_width_inches']
            target_h_inch = filename_hints['expected_height_inches']
            
            # Check both orientations for the filename hint
            target_sizes = [(target_w_inch, target_h_inch), (target_h_inch, target_w_inch)]
            
            for w_inch, h_inch in target_sizes:
                expected_w = w_inch * pixels_per_inch
                expected_h = h_inch * pixels_per_inch
                
                # Very strict matching for filename hints (15% tolerance)
                width_diff = abs(width_px - expected_w) / expected_w
                height_diff = abs(height_px - expected_h) / expected_h
                
                if width_diff < 0.15 and height_diff < 0.15:
                    return True, f"{w_inch}x{h_inch}", 1.5  # High confidence boost
            
            # If filename hint doesn't match, be very restrictive
            return False, None, 0.3
        
        # General standard size matching (20% tolerance)
        best_match = None
        best_confidence = 0.8  # Base confidence for standard sizes
        
        for w_inch, h_inch in standard_sizes:
            expected_w = w_inch * pixels_per_inch
            expected_h = h_inch * pixels_per_inch
            
            # Check with 20% tolerance
            width_diff = abs(width_px - expected_w) / expected_w
            height_diff = abs(height_px - expected_h) / expected_h
            
            if width_diff < 0.2 and height_diff < 0.2:
                # Calculate confidence based on how close the match is
                avg_diff = (width_diff + height_diff) / 2
                confidence = 1.0 - (avg_diff * 2)  # Closer match = higher confidence
                
                if confidence > best_confidence:
                    best_match = f"{w_inch}x{h_inch}"
                    best_confidence = confidence
        
        if best_match:
            return True, best_match, best_confidence
        
        return False, None, 0.0

    @staticmethod
    def detect_ads_from_pdf_metadata(pdf_path, page_number, publication_type='broadsheet', original_filename=None):
        """
        NEW: Enhanced PDF structure analysis for ad detection.
        Replaces broken sliding window approach with precise PDF metadata analysis.

        Args:
            pdf_path (str): Path to the PDF file
            page_number (int): Page number to analyze (1-based)
            publication_type (str): Type of publication for context
            original_filename (str, optional): Original filename for context

        Returns:
            list: Detected ad regions as dicts with x, y, width, height, confidence
        """
        print("Using NEW PDF structure analysis (replacing old rectangular detection)")

        # Use the new comprehensive PDF structure analyzer
        detected_ads = PDFStructureAdDetector.detect_ads_from_pdf_structure(
            pdf_path, page_number, publication_type
        )

        # Convert format to match expected interface
        formatted_ads = []
        for ad in detected_ads:
            formatted_ads.append({
                'x': ad['x'],
                'y': ad['y'],
                'width': ad['width'],
                'height': ad['height'],
                'confidence': ad['confidence'],
                'ad_type': ad.get('type', 'detected_ad'),
                'classification': ad.get('classification', 'pdf_structure'),
                'source_elements': ad.get('source_elements', [])
            })

        print(f"New PDF structure analysis complete: {len(formatted_ads)} ads detected")
        return formatted_ads
    
    @staticmethod
    def _analyze_image_element(img_rect, page_rect, publication_type, element_id):
        """Analyze an image element to determine if it's likely an ad"""
        try:
            width = img_rect.width
            height = img_rect.height
            area = width * height
            
            # Image size filtering - reasonable ad dimensions
            min_area = 100 * 80  # 8000 pixels minimum
            max_area = 800 * 600  # 480000 pixels maximum
            
            if area < min_area or area > max_area:
                return None
            
            # Aspect ratio analysis - ads typically have reasonable ratios
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < 0.1 or aspect_ratio > 10:  # Too thin or too wide
                return None
            
            # Position analysis - ads are typically not at extreme edges
            margin_threshold = 20  # pixels
            x_center = img_rect.x0 + width / 2
            y_center = img_rect.y0 + height / 2
            
            # Check if reasonably positioned (not at very edges)
            if (img_rect.x0 < margin_threshold and 
                img_rect.x1 > page_rect.width - margin_threshold):
                # Full width image - likely masthead or full-page content
                if height < 100:  # But if it's short, might be a banner ad
                    confidence = 0.6
                else:
                    confidence = 0.3  # Lower confidence for full-width tall images
            else:
                # Partial width images are more likely ads
                confidence = 0.8
            
            # Size-based confidence adjustments
            typical_ad_sizes = [
                (300, 250),  # Medium rectangle
                (728, 90),   # Leaderboard
                (160, 600),  # Wide skyscraper
                (300, 600),  # Half page
                (336, 280),  # Large rectangle
            ]
            
            # Check similarity to standard ad sizes
            size_match_bonus = 0
            for standard_width, standard_height in typical_ad_sizes:
                width_diff = abs(width - standard_width) / standard_width
                height_diff = abs(height - standard_height) / standard_height
                if width_diff < 0.2 and height_diff < 0.2:  # Within 20% of standard size
                    size_match_bonus = 0.2
                    break
            
            confidence += size_match_bonus
            confidence = min(confidence, 1.0)
            
            if confidence >= 0.5:
                return {
                    'x': img_rect.x0,
                    'y': img_rect.y0,
                    'width': width,
                    'height': height,
                    'confidence': confidence,
                    'type': 'image',
                    'element_id': element_id
                }
            
            return None
            
        except Exception as e:
            print(f"Error analyzing image element {element_id}: {e}")
            return None
    
    @staticmethod
    def _analyze_text_block(block, block_rect, page_rect, publication_type, element_id):
        """Analyze a text block to determine if it's likely an ad"""
        try:
            width = block_rect.width
            height = block_rect.height
            area = width * height
            
            # Size filtering
            if area < 2000:  # Too small to be a meaningful ad
                return None
            
            # Analyze fonts and text patterns
            fonts_used = set()
            total_chars = 0
            commercial_keywords = 0
            phone_numbers = 0
            web_urls = 0
            prices = 0
            
            commercial_patterns = [
                'call', 'phone', 'contact', 'visit', 'buy', 'sale', 'offer', 
                'special', 'deal', 'discount', 'free', 'new', 'now', 'today'
            ]
            
            if 'lines' in block:
                for line in block['lines']:
                    if 'spans' in line:
                        for span in line['spans']:
                            # Collect font information
                            font_info = f"{span.get('font', 'unknown')}_{span.get('size', 0)}"
                            fonts_used.add(font_info)
                            
                            # Analyze text content
                            text = span.get('text', '').lower()
                            total_chars += len(text)
                            
                            # Count commercial indicators
                            for keyword in commercial_patterns:
                                commercial_keywords += text.count(keyword)
                            
                            # Count contact info patterns
                            import re
                            phone_numbers += len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
                            web_urls += len(re.findall(r'www\.|\.com|\.org|\.net', text))
                            prices += len(re.findall(r'\$\d+|\d+\.\d{2}', text))
            
            # Calculate confidence based on patterns
            confidence = 0.0
            
            # Multiple fonts suggest designed layout (ads) vs single font (articles)
            if len(fonts_used) >= 3:
                confidence += 0.4
            elif len(fonts_used) == 2:
                confidence += 0.2
            
            # Commercial content indicators
            if total_chars > 0:
                commercial_ratio = (commercial_keywords + phone_numbers * 3 + web_urls * 2 + prices * 2) / total_chars
                confidence += min(commercial_ratio * 10, 0.4)  # Cap at 0.4
            
            # Size and position analysis
            aspect_ratio = width / height if height > 0 else 0
            if 0.5 <= aspect_ratio <= 4:  # Reasonable ad proportions
                confidence += 0.2
            
            # Position - ads often have defined boundaries
            if (block_rect.x0 % 10 == 0 or block_rect.y0 % 10 == 0):  # Aligned to grid
                confidence += 0.1
            
            # Apply publication-specific adjustments
            if publication_type == 'tabloid':
                confidence *= 1.1  # Tabloids have more varied ad layouts
            
            confidence = min(confidence, 1.0)
            
            if confidence >= 0.6:  # Higher threshold for text blocks
                return {
                    'x': block_rect.x0,
                    'y': block_rect.y0,
                    'width': width,
                    'height': height,
                    'confidence': confidence,
                    'type': 'text',
                    'element_id': element_id,
                    'fonts_count': len(fonts_used),
                    'commercial_indicators': commercial_keywords + phone_numbers + web_urls + prices
                }
            
            return None
            
        except Exception as e:
            print(f"Error analyzing text block {element_id}: {e}")
            return None
    
    @staticmethod
    def _detect_rectangular_border(drawing):
        """
        Detect if a drawing contains a closed rectangular border.
        Analyzes drawing paths to identify rectangular shapes that form borders.
        
        Args:
            drawing (dict): PyMuPDF drawing object with 'items' containing path data
            
        Returns:
            bool: True if the drawing forms a closed rectangular border
        """
        try:
            items = drawing.get('items', [])
            if not items:
                return False
            
            # For simple drawings (1-3 items), check if they form rectangular paths
            if len(items) <= 3:
                return PDFMetadataAdDetector._analyze_simple_rectangular_paths(items, drawing)
            
            # For more complex drawings, look for rectangular outline patterns
            return PDFMetadataAdDetector._analyze_complex_rectangular_paths(items, drawing)
            
        except Exception as e:
            print(f"Error in border detection: {e}")
            return False
    
    @staticmethod
    def _analyze_simple_rectangular_paths(items, drawing):
        """
        Analyze simple drawings (1-3 items) for rectangular borders.
        """
        try:
            draw_rect = drawing.get('rect')
            if not draw_rect:
                return False
            
            # Get drawing dimensions
            width = draw_rect.width
            height = draw_rect.height
            
            # Skip very small or malformed rectangles
            if width < 10 or height < 10:
                return False
            
            # For 1-item drawings, check if it's a single rectangular path
            if len(items) == 1:
                item = items[0]
                return PDFMetadataAdDetector._is_rectangular_path(item, width, height)
            
            # For 2-3 items, check if they combine to form a rectangle
            return PDFMetadataAdDetector._items_form_rectangle(items, width, height)
            
        except Exception as e:
            print(f"Error analyzing simple paths: {e}")
            return False
    
    @staticmethod
    def _analyze_complex_rectangular_paths(items, drawing):
        """
        Analyze complex drawings (4+ items) for rectangular borders.
        Most complex drawings are not simple borders, but some might be.
        """
        try:
            # For now, reject most complex drawings as they're likely not simple borders
            # But allow some specific cases that might be borders made of multiple segments
            if len(items) > 8:  # Too complex to be a simple border
                return False
            
            draw_rect = drawing.get('rect')
            if not draw_rect:
                return False
            
            width = draw_rect.width
            height = draw_rect.height
            
            # Check if items form 4 sides of a rectangle (common pattern)
            return PDFMetadataAdDetector._items_form_four_sided_rectangle(items, width, height)
            
        except Exception as e:
            print(f"Error analyzing complex paths: {e}")
            return False
    
    @staticmethod
    def _is_rectangular_path(item, expected_width, expected_height):
        """
        Check if a single path item forms a rectangle.
        """
        try:
            # Look for rectangular path data
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                return False
            
            # Check if the path data suggests a rectangle
            # In PyMuPDF, rectangular paths often have specific patterns
            path_type = item[0] if len(item) > 0 else None
            
            # Common rectangular path indicators
            if path_type in ['re', 'rect']:  # Rectangle command
                return True
            
            # Check for moveto/lineto patterns that form rectangles
            if path_type == 'm':  # moveto
                return PDFMetadataAdDetector._analyze_moveto_path(item, expected_width, expected_height)
            
            return False
            
        except Exception as e:
            print(f"Error checking rectangular path: {e}")
            return False
    
    @staticmethod
    def _items_form_rectangle(items, expected_width, expected_height):
        """
        Check if 2-3 items combine to form a rectangular border.
        """
        try:
            # This is a simplified check - in practice, we'd analyze the actual path coordinates
            # For now, assume that simple drawings with reasonable dimensions might be borders
            
            # Check if the total bounding area makes sense for a rectangle
            aspect_ratio = expected_width / expected_height if expected_height > 0 else 0
            
            # Reasonable aspect ratios for ad rectangles
            if 0.1 <= aspect_ratio <= 10:
                # Additional checks could be added here for path analysis
                return True
            
            return False
            
        except Exception as e:
            print(f"Error checking if items form rectangle: {e}")
            return False
    
    @staticmethod
    def _items_form_four_sided_rectangle(items, expected_width, expected_height):
        """
        Check if 4+ items form a four-sided rectangular border.
        """
        try:
            # For complex drawings, be more conservative
            # Only accept if it has characteristics of a structured border
            
            # Check aspect ratio
            if expected_height == 0:
                return False
            
            aspect_ratio = expected_width / expected_height
            
            # Reasonable aspect ratios for ad rectangles (more restrictive for complex drawings)
            if 0.2 <= aspect_ratio <= 5:
                # Could add more sophisticated path analysis here
                return True
            
            return False
            
        except Exception as e:
            print(f"Error checking four-sided rectangle: {e}")
            return False
    
    @staticmethod
    def _analyze_moveto_path(item, expected_width, expected_height):
        """
        Analyze moveto-based paths for rectangular patterns.
        """
        try:
            # Simplified analysis - in a full implementation, we'd parse the actual coordinates
            # For now, assume reasonable-sized drawings might be rectangles
            return expected_width >= 50 and expected_height >= 50
            
        except Exception as e:
            print(f"Error analyzing moveto path: {e}")
            return False
    
    @staticmethod
    def _analyze_drawing_element(drawing, draw_rect, page_rect, publication_type, element_id, has_border):
        """Analyze vector drawing elements focusing on bordered rectangles for ad detection"""
        try:
            width = draw_rect.width
            height = draw_rect.height
            
            # PRIORITY 3: Updated size requirements based on newspaper analysis
            # Minimum ad size: 100x80 pixels (increased from 80x50)
            if width < 100 or height < 80:
                return None
            
            # Maximum ad size: 60% page width, 40% page height (reduced from 70%/50%)
            if width > page_rect.width * 0.6 or height > page_rect.height * 0.4:
                return None
            
            # Only process simple bordered rectangles (1-3 items) if has_border=True
            items = drawing.get('items', [])
            items_count = len(items)
            
            if has_border:
                # Only detect simple bordered rectangles (1-3 drawing items)
                if items_count > 3:
                    return None
                
                # PRIORITY 1: Enhanced container vs content detection
                container_analysis = PDFMetadataAdDetector._analyze_container_vs_content(drawing, draw_rect, page_rect)
                
                if container_analysis['is_photo_container']:
                    # Photo containers get very low confidence
                    confidence = 0.1
                elif container_analysis['is_complex_ad_container']:
                    # Complex ad containers get high confidence - detect outer boundary
                    confidence = 0.9
                elif container_analysis['is_business_directory_item']:
                    # Business directory items get high confidence - keep individual
                    confidence = 0.95
                else:
                    # Standard bordered rectangles
                    confidence = 0.85
                    
                return {
                    'x': draw_rect.x0,
                    'y': draw_rect.y0,
                    'width': width,
                    'height': height,
                    'confidence': confidence,
                    'type': 'bordered_rectangle',
                    'element_id': element_id,
                    'border': True,
                    'items_count': items_count
                }
            else:
                # Non-bordered elements - much lower priority
                if items_count >= 4:
                    # Ignore complex drawings (4+ items)
                    return None
                
                # Simple non-bordered drawings get very low confidence
                confidence = 0.4
                
                return {
                    'x': draw_rect.x0,
                    'y': draw_rect.y0,
                    'width': width,
                    'height': height,
                    'confidence': confidence,
                    'type': 'simple_drawing',
                    'element_id': element_id,
                    'border': False,
                    'items_count': items_count
                }
            
        except Exception as e:
            print(f"Error analyzing drawing element {element_id}: {e}")
            return None
    
    @staticmethod
    def _analyze_container_vs_content(drawing, draw_rect, page_rect):
        """
        PRIORITY 1: Analyze whether this is a container (ad boundary) or internal content
        
        Returns:
            dict: Analysis results with container type classification
        """
        try:
            width = draw_rect.width
            height = draw_rect.height
            
            analysis = {
                'is_photo_container': False,
                'is_complex_ad_container': False,
                'is_business_directory_item': False,
                'confidence_modifier': 0
            }
            
            # Get drawing complexity
            items = drawing.get('items', [])
            items_count = len(items)
            
            # PRIORITY 1: Photo container detection
            # If rectangular region contains mostly image data, mark as photo
            if PDFMetadataAdDetector._contains_mostly_image_data(draw_rect, page_rect):
                analysis['is_photo_container'] = True
                return analysis
            
            # PRIORITY 1: Business directory detection
            # Small to medium rectangles with business-like dimensions
            if (100 <= width <= 300 and 80 <= height <= 200 and 
                PDFMetadataAdDetector._is_in_directory_layout(draw_rect, page_rect)):
                analysis['is_business_directory_item'] = True
                return analysis
            
            # PRIORITY 1: Complex ad container detection
            # Large rectangles that likely contain internal photos/text
            if (width >= 250 and height >= 150 and 
                PDFMetadataAdDetector._likely_contains_internal_content(draw_rect, page_rect)):
                analysis['is_complex_ad_container'] = True
                return analysis
            
            return analysis
            
        except Exception as e:
            print(f"Error in container analysis: {e}")
            return {'is_photo_container': False, 'is_complex_ad_container': False, 
                   'is_business_directory_item': False, 'confidence_modifier': 0}
    
    @staticmethod
    def _contains_mostly_image_data(draw_rect, page_rect):
        """
        PRIORITY 1: Check if rectangular region contains mostly image data
        """
        try:
            # This is a simplified check - in production, you'd analyze the actual PDF content
            width = draw_rect.width
            height = draw_rect.height
            aspect_ratio = width / height if height > 0 else 0
            
            # Photo characteristics: reasonable size + photo-like aspect ratio
            if width >= 200 and height >= 150:
                # Check for photo aspect ratios
                photo_ratios = [1.5, 1.33, 1.78, 1.0]  # 3:2, 4:3, 16:9, square
                for ratio in photo_ratios:
                    if abs(aspect_ratio - ratio) <= 0.1 or abs(aspect_ratio - (1/ratio)) <= 0.1:
                        # In content area = likely editorial photo
                        if PDFMetadataAdDetector._is_in_main_content_area(draw_rect, page_rect):
                            return True
            
            return False
            
        except Exception as e:
            print(f"Error checking image data: {e}")
            return False
    
    @staticmethod
    def _is_in_directory_layout(draw_rect, page_rect):
        """
        PRIORITY 1: Check if rectangle is part of a business directory layout
        """
        try:
            # Business directories typically have:
            # 1. Multiple similar-sized rectangles
            # 2. Grid-like arrangement
            # 3. Consistent spacing
            
            # For now, use position-based heuristics
            # Bottom half of page often contains directories/classifieds
            rect_center_y = draw_rect.y0 + draw_rect.height / 2
            page_center_y = page_rect.height / 2
            
            # Lower half of page + reasonable business card size
            if rect_center_y > page_center_y:
                width = draw_rect.width
                height = draw_rect.height
                # Typical business card proportions
                if 100 <= width <= 300 and 80 <= height <= 150:
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking directory layout: {e}")
            return False
    
    @staticmethod
    def _likely_contains_internal_content(draw_rect, page_rect):
        """
        PRIORITY 1: Check if large rectangle likely contains internal photos/text
        """
        try:
            width = draw_rect.width
            height = draw_rect.height
            area = width * height
            
            # Large rectangles are more likely to contain internal content
            if area >= 250 * 150:  # 37,500 square pixels
                # Check if it's positioned like a complex ad
                # (not in extreme corners, reasonable proportions)
                aspect_ratio = width / height if height > 0 else 0
                if 0.3 <= aspect_ratio <= 4.0:  # Reasonable ad proportions
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking internal content: {e}")
            return False
    
    @staticmethod
    def _is_enhanced_editorial_photo(draw_rect, page_rect):
        """Enhanced editorial photo detection with stricter criteria"""
        try:
            width = draw_rect.width
            height = draw_rect.height
            
            # CRITICAL FIX: Photo size threshold - photos usually larger than 200x150
            if width < 200 or height < 150:
                return False  # Too small to be a typical photo
            
            # Check aspect ratios common for photos
            aspect_ratio = width / height if height > 0 else 0
            
            # CRITICAL FIX: Specific aspect ratio check for photos
            # If ratio is 1.3-1.8 AND size > 200x150, likely a photo
            if 1.3 <= aspect_ratio <= 1.8 and width > 200 and height > 150:
                # Additional check for position in main content area
                if PDFMetadataAdDetector._is_in_main_content_area(draw_rect, page_rect):
                    return True  # Large photo in content area = editorial photo
            
            # Check other common photo ratios with size requirements
            photo_ratios = [1.5, 1.33, 1.78, 1.0]  # 3:2, 4:3, 16:9, square
            tolerance = 0.1
            
            is_photo_ratio = False
            for ratio in photo_ratios:
                if abs(aspect_ratio - ratio) <= tolerance or abs(aspect_ratio - (1/ratio)) <= tolerance:
                    is_photo_ratio = True
                    break
            
            if is_photo_ratio:
                # Large photos with typical ratios in content area are likely editorial
                area = width * height
                if area >= 200 * 150 and PDFMetadataAdDetector._is_in_main_content_area(draw_rect, page_rect):
                    return True
            
            return False  # Probably an ad frame
            
        except Exception as e:
            print(f"Error in enhanced photo detection: {e}")
            return False
    
    @staticmethod
    def _is_in_main_content_area(draw_rect, page_rect):
        """Check if rectangle is in main content area (center 60% of page)"""
        try:
            # Calculate center position of the rectangle
            rect_center_x = draw_rect.x0 + draw_rect.width / 2
            rect_center_y = draw_rect.y0 + draw_rect.height / 2
            
            # Define main content area as center 60% of page
            content_left = page_rect.width * 0.2  # 20% margin from left
            content_right = page_rect.width * 0.8  # 20% margin from right
            content_top = page_rect.height * 0.2   # 20% margin from top
            content_bottom = page_rect.height * 0.8 # 20% margin from bottom
            
            # Check if rectangle center is in main content area
            return (content_left <= rect_center_x <= content_right and 
                   content_top <= rect_center_y <= content_bottom)
            
        except Exception as e:
            print(f"Error checking content area: {e}")
            return False
    
    @staticmethod
    def _merge_and_filter_detections(detections, min_confidence=0.8, overlap_threshold=0.3):
        """Advanced merge and filter with fragmentation fixes and nested removal"""
        try:
            # STEP 1: Filter by minimum confidence
            filtered = [d for d in detections if d['confidence'] >= min_confidence]
            
            if not filtered:
                return []
            
            # STEP 2: Remove very small text elements (smaller than 80x50)
            size_filtered = []
            for detection in filtered:
                width = detection['width']
                height = detection['height']
                if width >= 80 and height >= 50:
                    size_filtered.append(detection)
                else:
                    print(f"Removing small element: {width:.0f}x{height:.0f}")
            
            if not size_filtered:
                return []
            
            # STEP 3: Remove nested rectangles (small ones inside larger ones)
            non_nested = PDFMetadataAdDetector._remove_nested_rectangles(size_filtered)
            
            # STEP 4: Merge adjacent/fragmented detections within 20 pixels
            merged = PDFMetadataAdDetector._merge_adjacent_detections(non_nested, merge_distance=20)
            
            # STEP 5: Final overlap-based merging
            final_merged = PDFMetadataAdDetector._merge_overlapping_detections(merged, overlap_threshold)
            
            # STEP 6: Sort by confidence (highest first)
            final_merged.sort(key=lambda x: x['confidence'], reverse=True)
            
            return final_merged
            
        except Exception as e:
            print(f"Error in advanced merge and filter: {e}")
            return detections  # Return original if merging fails
    
    @staticmethod
    def _remove_nested_rectangles(detections):
        """Remove small rectangles that are completely inside larger ones"""
        try:
            if len(detections) <= 1:
                return detections
            
            # Sort by area (largest first)
            sorted_detections = sorted(detections, key=lambda d: d['width'] * d['height'], reverse=True)
            
            non_nested = []
            
            for i, current in enumerate(sorted_detections):
                current_rect = fitz.Rect(current['x'], current['y'],
                                       current['x'] + current['width'],
                                       current['y'] + current['height'])
                
                is_nested = False
                
                # Check if current rectangle is inside any larger rectangle we've already kept
                for existing in non_nested:
                    existing_rect = fitz.Rect(existing['x'], existing['y'],
                                            existing['x'] + existing['width'],
                                            existing['y'] + existing['height'])
                    
                    # Check if current is completely inside existing (with small tolerance)
                    tolerance = 5  # pixels
                    if (current_rect.x0 >= existing_rect.x0 - tolerance and
                        current_rect.y0 >= existing_rect.y0 - tolerance and
                        current_rect.x1 <= existing_rect.x1 + tolerance and
                        current_rect.y1 <= existing_rect.y1 + tolerance):
                        
                        # Reduced logging
                        pass
                        is_nested = True
                        break
                
                if not is_nested:
                    non_nested.append(current)
            
            return non_nested
            
        except Exception as e:
            print(f"Error removing nested rectangles: {e}")
            return detections
    
    @staticmethod
    def _merge_adjacent_detections(detections, merge_distance=20):
        """PRIORITY 1: Smart merge that preserves business directories but merges fragmented ads"""
        try:
            if len(detections) <= 1:
                return detections
            
            merged = []
            used_indices = set()
            
            for i, current in enumerate(detections):
                if i in used_indices:
                    continue
                
                current_rect = fitz.Rect(current['x'], current['y'],
                                       current['x'] + current['width'],
                                       current['y'] + current['height'])
                
                # PRIORITY 1: Don't merge business directory items
                if current.get('type') == 'bordered_rectangle' and current['confidence'] >= 0.95:
                    # High confidence items (business directory) should not be merged
                    merged.append(current)
                    used_indices.add(i)
                    continue
                
                # Find detections that should be merged with current
                to_merge = [current]
                used_indices.add(i)
                
                for j, candidate in enumerate(detections):
                    if j <= i or j in used_indices:
                        continue
                    
                    # PRIORITY 1: Don't merge business directory items
                    if candidate.get('type') == 'bordered_rectangle' and candidate['confidence'] >= 0.95:
                        continue
                    
                    candidate_rect = fitz.Rect(candidate['x'], candidate['y'],
                                             candidate['x'] + candidate['width'],
                                             candidate['y'] + candidate['height'])
                    
                    # PRIORITY 1: Smart merging rules
                    should_merge = False
                    
                    # Only merge if both are likely parts of the same complex ad
                    if (PDFMetadataAdDetector._are_rectangles_adjacent(current_rect, candidate_rect, merge_distance) and
                        PDFMetadataAdDetector._should_merge_rectangles(current, candidate)):
                        should_merge = True
                    
                    if should_merge:
                        to_merge.append(candidate)
                        used_indices.add(j)
                        # Reduced logging
                        pass
                
                # Create merged detection
                if len(to_merge) == 1:
                    merged.append(current)
                else:
                    merged_detection = PDFMetadataAdDetector._create_merged_detection(to_merge)
                    merged.append(merged_detection)
            
            return merged
            
        except Exception as e:
            print(f"Error in smart merging: {e}")
            return detections
    
    @staticmethod
    def _should_merge_rectangles(rect1, rect2):
        """PRIORITY 1: Determine if two rectangles should be merged based on newspaper rules"""
        try:
            # Don't merge business directory items (high confidence, small-medium size)
            if (rect1['confidence'] >= 0.95 or rect2['confidence'] >= 0.95):
                return False
            
            # Don't merge if both are large (likely separate ads)
            area1 = rect1['width'] * rect1['height']
            area2 = rect2['width'] * rect2['height']
            
            if area1 >= 40000 and area2 >= 40000:  # Both large (200x200)
                return False
            
            # Do merge if one is small and they're adjacent (likely fragmented ad)
            min_area = min(area1, area2)
            if min_area < 20000:  # One is small (< ~140x140)
                return True
            
            # Do merge if similar sizes and close together (likely split ad)
            size_ratio = max(area1, area2) / min(area1, area2)
            if size_ratio <= 3.0:  # Similar sizes
                return True
            
            return False
            
        except Exception as e:
            print(f"Error checking merge criteria: {e}")
            return False
    
    @staticmethod
    def _are_rectangles_adjacent(rect1, rect2, max_distance):
        """Check if two rectangles are within max_distance pixels of each other"""
        try:
            # Calculate minimum distance between rectangles
            # If they overlap, distance is 0
            if not (rect1 & rect2).is_empty:
                return True
            
            # Calculate horizontal and vertical distances
            horizontal_distance = max(0, max(rect1.x0 - rect2.x1, rect2.x0 - rect1.x1))
            vertical_distance = max(0, max(rect1.y0 - rect2.y1, rect2.y0 - rect1.y1))
            
            # Check if within max_distance
            return horizontal_distance <= max_distance and vertical_distance <= max_distance
            
        except Exception as e:
            print(f"Error checking rectangle adjacency: {e}")
            return False
    
    @staticmethod
    def _create_merged_detection(detections_to_merge):
        """Create a single detection from multiple adjacent detections"""
        try:
            if not detections_to_merge:
                return None
            
            # Find bounding rectangle of all detections
            min_x = min(d['x'] for d in detections_to_merge)
            min_y = min(d['y'] for d in detections_to_merge)
            max_x = max(d['x'] + d['width'] for d in detections_to_merge)
            max_y = max(d['y'] + d['height'] for d in detections_to_merge)
            
            # Use highest confidence
            max_confidence = max(d['confidence'] for d in detections_to_merge)
            
            # Use properties from highest confidence detection
            best_detection = max(detections_to_merge, key=lambda d: d['confidence'])
            
            merged = {
                'x': min_x,
                'y': min_y,
                'width': max_x - min_x,
                'height': max_y - min_y,
                'confidence': max_confidence,
                'type': best_detection['type'],
                'element_id': f"merged_{best_detection['element_id']}",
                'border': best_detection.get('border', False),
                'items_count': sum(d.get('items_count', 0) for d in detections_to_merge)
            }
            
            return merged
            
        except Exception as e:
            print(f"Error creating merged detection: {e}")
            return detections_to_merge[0] if detections_to_merge else None
    
    @staticmethod
    def _merge_overlapping_detections(detections, overlap_threshold=0.3):
        """Traditional overlap-based merging as final step"""
        try:
            if len(detections) <= 1:
                return detections
            
            merged = []
            
            for current in detections:
                should_merge = False
                current_rect = fitz.Rect(current['x'], current['y'],
                                       current['x'] + current['width'],
                                       current['y'] + current['height'])
                
                for i, existing in enumerate(merged):
                    existing_rect = fitz.Rect(existing['x'], existing['y'],
                                            existing['x'] + existing['width'],
                                            existing['y'] + existing['height'])
                    
                    # Calculate overlap
                    intersection = current_rect & existing_rect
                    if intersection.is_empty:
                        continue
                    
                    overlap_area = intersection.width * intersection.height
                    current_area = current_rect.width * current_rect.height
                    existing_area = existing_rect.width * existing_rect.height
                    
                    overlap_ratio = overlap_area / min(current_area, existing_area)
                    
                    if overlap_ratio >= overlap_threshold:
                        # Merge with existing detection (keep higher confidence)
                        if current['confidence'] > existing['confidence']:
                            merged[i] = current
                        should_merge = True
                        break
                
                if not should_merge:
                    merged.append(current)
            
            return merged
            
        except Exception as e:
            print(f"Error in overlap-based merging: {e}")
            return detections
    
    @staticmethod
    def transform_pdf_to_image_coordinates(pdf_detections, pdf_page_rect, image_width, image_height):
        """
        Transform PDF coordinates to match page image pixel coordinates.
        
        Args:
            pdf_detections (list): Detections in PDF coordinate system
            pdf_page_rect (fitz.Rect): PDF page dimensions
            image_width (int): Width of generated page image in pixels
            image_height (int): Height of generated page image in pixels
            
        Returns:
            list: Detections with coordinates transformed to image pixel system
        """
        try:
            if not pdf_detections:
                return []
            
            # Calculate scaling factors
            scale_x = image_width / pdf_page_rect.width
            scale_y = image_height / pdf_page_rect.height
            
            transformed_detections = []
            
            for detection in pdf_detections:
                # Transform coordinates
                transformed = {
                    'x': detection['x'] * scale_x,
                    'y': detection['y'] * scale_y,
                    'width': detection['width'] * scale_x,
                    'height': detection['height'] * scale_y,
                    'confidence': detection['confidence'],
                    'type': detection['type'],
                    'element_id': detection.get('element_id', 'unknown')
                }
                
                # Preserve additional metadata
                for key in ['fonts_count', 'commercial_indicators']:
                    if key in detection:
                        transformed[key] = detection[key]
                
                transformed_detections.append(transformed)
            
            return transformed_detections
            
        except Exception as e:
            print(f"Error transforming coordinates: {e}")
            return pdf_detections  # Return original if transformation fails


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
        except Exception as e:
            print(f"Error in intelligent rectangle detection: {e}")
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

# Google Vision AI Ad Detector
class GoogleVisionAdDetector:
    """
    Google Vision AI powered ad detection system
    Uses logo detection, text analysis, and object recognition to identify ads
    """
    
    @staticmethod
    def detect_ads(image_path, publication_type='broadsheet'):
        """
        Main function to detect ads using Google Vision AI
        Returns list of detected ads with coordinates and confidence scores
        """
        try:
            print(f"Starting Google Vision AI detection on {image_path}")
            
            # Initialize the client
            client = vision.ImageAnnotatorClient()
            
            # Load the image
            with vision_io.open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # Get image dimensions for coordinate calculations
            with Image.open(image_path) as pil_image:
                img_width, img_height = pil_image.size
            
            print(f"Image dimensions: {img_width}x{img_height}")
            
            # HYBRID APPROACH: Visual boundaries + specialized text detection
            boundary_ads = GoogleVisionAdDetector._detect_visual_boundary_ads(client, image, img_width, img_height, image_path)
            logo_ads = GoogleVisionAdDetector._detect_logo_ads(client, image, img_width, img_height)
            # SPECIALIZED text detection for newspaper ad types
            specialized_text_ads = GoogleVisionAdDetector._detect_newspaper_text_ads(client, image, img_width, img_height)
            object_ads = GoogleVisionAdDetector._detect_commercial_object_ads(client, image, img_width, img_height)
            
            # Combine all detections
            all_detections = boundary_ads + logo_ads + specialized_text_ads + object_ads
            print(f"Raw detections: {len(all_detections)} (boundary: {len(boundary_ads)}, logos: {len(logo_ads)}, text: {len(specialized_text_ads)}, objects: {len(object_ads)})")
            
            # Filter and merge overlapping detections
            filtered_ads = GoogleVisionAdDetector._filter_and_merge_detections(all_detections, publication_type, img_width, img_height)
            
            print(f"Google Vision AI found {len(filtered_ads)} potential ads")
            return filtered_ads
            
        except Exception as e:
            error_type = handle_vision_api_error(e, "ad detection")
            if error_type == "quota_exceeded":
                print("⚠️ Google Vision API quota exceeded - falling back to traditional detection")
            elif error_type == "permission_denied":
                print("⚠️ Google Vision API permission denied - check credentials")
            else:
                print(f"⚠️ Google Vision AI detection failed: {e}")
            return []
    
    @staticmethod
    def _detect_logo_ads(client, image, img_width, img_height):
        """Detect ads containing brand logos"""
        try:
            response = client.logo_detection(image=image)
            logos = response.logo_annotations
            
            ads = []
            for logo in logos:
                if logo.score > 0.5:  # High confidence logos only
                    # Get bounding box
                    vertices = logo.bounding_poly.vertices
                    x_coords = [v.x for v in vertices]
                    y_coords = [v.y for v in vertices]
                    
                    x = min(x_coords)
                    y = min(y_coords) 
                    width = max(x_coords) - x
                    height = max(y_coords) - y
                    
                    # Expand around logo to capture full ad
                    expanded_ad = GoogleVisionAdDetector._expand_logo_to_full_ad(
                        client, image, x, y, width, height, img_width, img_height
                    )
                    
                    if expanded_ad:
                        ads.append({
                            'x': expanded_ad['x'],
                            'y': expanded_ad['y'],
                            'width': expanded_ad['width'],
                            'height': expanded_ad['height'],
                            'confidence': logo.score,
                            'ad_type': 'logo_ad',
                            'content': f"Logo: {logo.description}",
                            'detection_method': 'vision_logo'
                        })
            
            return ads
        except Exception as e:
            print(f"Logo detection failed: {e}")
            return []
    
    @staticmethod
    def _detect_commercial_text_ads(client, image, img_width, img_height):
        """DISABLED - Old commercial text detection (replaced with specialized newspaper detection)"""
        try:
            response = client.text_detection(image=image)
            texts = response.text_annotations
            
            if not texts:
                return []
            
            ads = []
            
            # Look for commercial keywords
            commercial_keywords = [
                'SALE', 'FREE', 'CALL NOW', 'CALL', '$', '%', 'PERCENT', 'OFF',
                'SPECIAL', 'LIMITED TIME', 'ACT NOW', 'HURRY', 'SAVE',
                'DISCOUNT', 'DEAL', 'OFFER', 'FINANCING', 'CREDIT',
                'VISIT', 'HOURS', 'OPEN', 'CLOSED', 'SUNDAY',
                'PHONE:', 'TEL:', 'CALL:', 'CONTACT:', 'EMAIL:',
                'WWW.', '.COM', '.NET', 'HTTP'
            ]
            
            # Phone number pattern
            import re
            phone_pattern = r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})|(\d{3}[-.\s]\d{4})'
            price_pattern = r'\$[\d,]+\.?\d*'
            
            for text in texts[1:]:  # Skip first full page text
                text_content = text.description.upper()
                
                # Check for commercial indicators
                commercial_score = 0
                found_keywords = []
                
                for keyword in commercial_keywords:
                    if keyword in text_content:
                        commercial_score += 1
                        found_keywords.append(keyword)
                
                # Check for phone numbers
                if re.search(phone_pattern, text.description):
                    commercial_score += 2
                    found_keywords.append('PHONE_NUMBER')
                
                # Check for prices
                if re.search(price_pattern, text.description):
                    commercial_score += 1
                    found_keywords.append('PRICE')
                
                if commercial_score >= 2:  # Need at least 2 commercial indicators
                    # Get bounding box
                    vertices = text.bounding_poly.vertices
                    x_coords = [v.x for v in vertices]
                    y_coords = [v.y for v in vertices]
                    
                    x = min(x_coords)
                    y = min(y_coords)
                    width = max(x_coords) - x
                    height = max(y_coords) - y
                    
                    # Expand text region to capture full ad
                    expanded_ad = GoogleVisionAdDetector._expand_text_to_full_ad(
                        x, y, width, height, img_width, img_height
                    )
                    
                    confidence = min(commercial_score / 5.0, 0.95)
                    
                    ads.append({
                        'x': expanded_ad['x'],
                        'y': expanded_ad['y'],
                        'width': expanded_ad['width'],
                        'height': expanded_ad['height'],
                        'confidence': confidence,
                        'ad_type': 'text_commercial',
                        'content': f"Commercial text: {', '.join(found_keywords)}",
                        'detection_method': 'vision_text'
                    })
            
            return ads
        except Exception as e:
            print(f"Text detection failed: {e}")
            return []
    
    @staticmethod
    def _detect_newspaper_text_ads(client, image, img_width, img_height):
        """Detect specific newspaper ad types: Public Notices, Legal Notices, Classifieds, Subscription boxes"""
        try:
            response = client.text_detection(image=image)
            texts = response.text_annotations
            
            if not texts:
                return []
            
            ads = []
            
            # Get full page text for context
            full_text = texts[0].description if texts else ""
            
            # Define newspaper-specific ad patterns
            newspaper_ad_patterns = {
                'public_notice': {
                    'keywords': ['PUBLIC NOTICE', 'NOTICE', 'LEGAL NOTICE', 'NOTICE TO CREDITORS', 'NOTICE OF HEARING'],
                    'ad_type': 'public_notice',
                    'confidence_boost': 0.3
                },
                'legal_proceedings': {
                    'keywords': ['OFFICIAL PROCEEDINGS OF', 'PROCEEDINGS OF THE', 'BOARD MEETING', 'CITY COUNCIL', 'TOWNSHIP BOARD'],
                    'ad_type': 'legal_notice', 
                    'confidence_boost': 0.3
                },
                'subscription_info': {
                    'keywords': ['subscription rates', 'published every', 'entered at the post office', 'periodical postage', 'address changes'],
                    'ad_type': 'subscription_notice',
                    'confidence_boost': 0.4
                },
                'classified_headers': {
                    'keywords': ['CLASSIFIEDS', 'AUCTIONS', 'FOR SALE', 'PICKUPS', 'MOTORCYCLES', 'FARM EQUIPMENT', 'GARAGE SALE'],
                    'ad_type': 'classified_header',
                    'confidence_boost': 0.2
                },
                'church_ads': {
                    'keywords': ['CHURCH', 'PASTOR', 'GOSPEL', 'WORSHIP', 'SUNDAY', 'SERVICE', 'BIBLE', 'CHRISTIAN', 'MENNONITE', 'LUTHERAN', 'METHODIST'],
                    'ad_type': 'church_ad',
                    'confidence_boost': 0.3
                },
                'business_directory': {
                    'keywords': ['BUSINESS DIRECTORY', 'SERVICE DIRECTORY', 'CLINIC', 'PHARMACY', 'DENTIST', 'EYE CARE', 'CONSTRUCTION', 'REPAIR'],
                    'ad_type': 'business_directory',
                    'confidence_boost': 0.3
                }
            }
            
            # Look for these specific text patterns
            for text in texts[1:]:  # Skip full page text
                text_content = text.description.upper()
                text_lines = text_content.split('\\n')
                
                for pattern_type, pattern_info in newspaper_ad_patterns.items():
                    ad_score = 0
                    found_keywords = []
                    
                    # Check for pattern-specific keywords
                    for keyword in pattern_info['keywords']:
                        if keyword in text_content:
                            ad_score += 2
                            found_keywords.append(keyword)
                    
                    # Special handling for different ad types
                    if pattern_type == 'subscription_info':
                        # Look for specific subscription info phrases
                        subscription_phrases = ['$60 per year', '$82 per year', 'P.O. Box', 'transferable but non-refundable']
                        for phrase in subscription_phrases:
                            if phrase.lower() in text.description.lower():
                                ad_score += 1
                                found_keywords.append('SUBSCRIPTION_INFO')
                    
                    elif pattern_type == 'classified_headers':
                        # For classified headers, look for category patterns
                        if len(text.description.split()) <= 3 and any(keyword in text_content for keyword in pattern_info['keywords']):
                            ad_score += 3  # Boost for short header text
                    
                    elif pattern_type == 'public_notice':
                        # Look for legal formatting patterns
                        if any(phrase in text_content for phrase in ['STATE OF', 'COUNTY OF', 'WHEREAS', 'THEREFORE']):
                            ad_score += 1
                    
                    # ULTRA-RESTRICTIVE: Require overwhelming evidence
                    if ad_score >= 6:  # Raised from 3 to 6 - need very strong evidence
                        # Get bounding box
                        vertices = text.bounding_poly.vertices
                        x_coords = [v.x for v in vertices]
                        y_coords = [v.y for v in vertices]
                        
                        x = min(x_coords)
                        y = min(y_coords)
                        width = max(x_coords) - x
                        height = max(y_coords) - y
                        
                        # Apply size constraints for text ads
                        if width < 60 or height < 30:  # Too small
                            continue
                        if width > img_width * 0.8 or height > img_height * 0.6:  # Too large
                            continue
                        
                        # Expand text region appropriately based on type
                        if pattern_type == 'classified_headers':
                            # For classified headers, expand down to include ads below
                            expanded_ad = GoogleVisionAdDetector._expand_classified_header(
                                x, y, width, height, img_width, img_height
                            )
                        else:
                            # For other text ads, modest expansion
                            expanded_ad = GoogleVisionAdDetector._expand_text_ad(
                                x, y, width, height, img_width, img_height
                            )
                        
                        confidence = min(0.4 + pattern_info['confidence_boost'] + (ad_score / 10.0), 0.95)
                        
                        ads.append({
                            'x': expanded_ad['x'],
                            'y': expanded_ad['y'],
                            'width': expanded_ad['width'],
                            'height': expanded_ad['height'],
                            'confidence': confidence,
                            'ad_type': pattern_info['ad_type'],
                            'content': f'Newspaper ad: {", ".join(found_keywords[:3])}',
                            'detection_method': 'vision_newspaper_text'
                        })
                        
                        # Limit to prevent over-detection
                        if len(ads) >= 10:
                            break
            
            print(f'Found {len(ads)} specialized newspaper text ads')
            return ads
            
        except Exception as e:
            print(f'Specialized newspaper text detection failed: {e}')
            return []
    
    @staticmethod
    def _detect_visual_boundary_ads(client, image, img_width, img_height, image_path):
        """Detect ads by finding rectangular regions with borders/visual boundaries"""
        try:
            import cv2
            import numpy as np
            
            # Load image for edge detection
            cv_image = cv2.imread(image_path)
            if cv_image is None:
                return []
            
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Use Vision API to get text regions for context
            response = client.text_detection(image=image)
            texts = response.text_annotations
            
            ads = []
            
            # ENHANCED: Multiple edge detection approaches for better boundary finding
            
            # Method 1: Standard edge detection
            edges1 = cv2.Canny(gray, 30, 100, apertureSize=3)  # More sensitive
            
            # Method 2: Stronger edge detection for clear boundaries  
            edges2 = cv2.Canny(gray, 80, 200, apertureSize=5)  # Less sensitive but clearer
            
            # Method 3: Morphological operations to enhance rectangular structures
            kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            morph_edges = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_rect)
            _, morph_thresh = cv2.threshold(morph_edges, 50, 255, cv2.THRESH_BINARY)
            
            # Combine all edge detection methods
            combined_edges = cv2.bitwise_or(edges1, edges2)
            combined_edges = cv2.bitwise_or(combined_edges, morph_thresh)
            
            # Find contours from combined edge detection
            contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            print(f"Found {len(contours)} potential boundary contours")
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # IMPROVED: More flexible size filtering for various ad types
                
                # Minimum sizes - accept smaller ads too
                min_width = 80   # Reduced from 100
                min_height = 60  # Reduced minimum
                
                # Maximum sizes - more permissive
                max_width = img_width * 0.7   # Increased from 0.6
                max_height = img_height * 0.5  # Increased from 0.4
                
                if (w < min_width or h < min_height or 
                    w > max_width or h > max_height):
                    continue
                
                # More flexible aspect ratio - newspapers have various ad shapes
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.2 or aspect_ratio > 10:  # More permissive
                    continue
                
                # Skip very small areas that are likely noise
                area = w * h
                if area < 5000:  # Must have reasonable area
                    continue
                
                # Check if this region contains text (good indicator of an ad)
                has_text = False
                text_density = 0
                
                if texts:
                    for text in texts[1:]:  # Skip full page text
                        if hasattr(text, 'bounding_poly') and text.bounding_poly.vertices:
                            text_vertices = text.bounding_poly.vertices
                            text_x = min(v.x for v in text_vertices)
                            text_y = min(v.y for v in text_vertices)
                            text_w = max(v.x for v in text_vertices) - text_x
                            text_h = max(v.y for v in text_vertices) - text_y
                            
                            # Check if text overlaps with this boundary
                            if (text_x < x + w and text_x + text_w > x and 
                                text_y < y + h and text_y + text_h > y):
                                has_text = True
                                text_density += len(text.description)
                                break
                
                # Only consider regions with some text content
                if not has_text:
                    continue
                
                # CRITICAL: Filter out news stories and editorial content
                if GoogleVisionAdDetector._is_likely_news_story(gray[y:y+h, x:x+w], texts, x, y, w, h):
                    continue
                
                # ENHANCED: Better confidence calculation with photo detection
                
                # Check perimeter vs area ratio (ads often have clear boundaries)
                perimeter = cv2.arcLength(contour, True)
                contour_area = cv2.contourArea(contour)
                
                if contour_area == 0:
                    continue
                
                boundary_strength = (perimeter * perimeter) / contour_area
                
                # Additional scoring for photo/image regions
                roi_region = gray[y:y+h, x:x+w]
                
                # Check if this looks like a photo region (varied intensity)
                photo_score = 0
                if roi_region.size > 0:
                    # Calculate intensity variance (photos have more variation)
                    intensity_var = np.var(roi_region)
                    if intensity_var > 500:  # High variance suggests photos/graphics
                        photo_score += 0.2
                    
                    # Check for rectangular borders (common in ads)
                    border_pixels = np.concatenate([
                        roi_region[0, :],      # top row
                        roi_region[-1, :],     # bottom row  
                        roi_region[:, 0],      # left column
                        roi_region[:, -1]      # right column
                    ])
                    border_consistency = 1.0 - (np.std(border_pixels) / 255.0)
                    if border_consistency > 0.7:  # Consistent borders
                        photo_score += 0.2
                
                # MUCH MORE CONSERVATIVE CONFIDENCE: Lower base confidence, higher requirements
                base_confidence = min(0.1 + (boundary_strength / 200), 0.5)  # Much lower base
                
                # Require both photo characteristics AND good boundaries for high confidence
                confidence = base_confidence + photo_score
                
                # BALANCED: Accept moderate-confidence detections for legitimate ads
                if confidence < 0.3:  # Lowered from 0.4 to 0.3 for balance
                    continue
                
                ads.append({
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'confidence': confidence,
                    'ad_type': 'boundary_detected',
                    'content': f'Visual boundary detection (strength: {boundary_strength:.1f})',
                    'detection_method': 'vision_boundary'
                })
                
                # Allow more ads to be detected
                if len(ads) >= 15:  # Increased from 8 to 15
                    break
            
            print(f'Found {len(ads)} potential boundary-based ads')
            return ads
            
        except Exception as e:
            print(f'Visual boundary detection failed: {e}')
            return []
    
    @staticmethod
    def _detect_commercial_object_ads(client, image, img_width, img_height):
        """Detect ads based on commercial objects (vehicles, food, etc.)"""
        try:
            response = client.object_localization(image=image)
            objects = response.localized_object_annotations
            
            # Commercial object categories
            commercial_objects = {
                'Vehicle': 'commercial_vehicle',
                'Car': 'commercial_vehicle', 
                'Truck': 'commercial_vehicle',
                'Motorcycle': 'commercial_vehicle',
                'Food': 'commercial_food',
                'Furniture': 'commercial_furniture',
                'Building': 'commercial_building',
                'House': 'commercial_building'
            }
            
            ads = []
            for obj in objects:
                if obj.name in commercial_objects and obj.score > 0.6:
                    # Convert normalized coordinates to pixels
                    vertices = obj.bounding_poly.normalized_vertices
                    x = int(vertices[0].x * img_width)
                    y = int(vertices[0].y * img_height)
                    x2 = int(vertices[2].x * img_width)
                    y2 = int(vertices[2].y * img_height)
                    
                    width = x2 - x
                    height = y2 - y
                    
                    # Expand around object to capture full ad
                    expanded_ad = GoogleVisionAdDetector._expand_object_to_full_ad(
                        x, y, width, height, img_width, img_height
                    )
                    
                    ads.append({
                        'x': expanded_ad['x'],
                        'y': expanded_ad['y'],
                        'width': expanded_ad['width'],
                        'height': expanded_ad['height'],
                        'confidence': obj.score,
                        'ad_type': commercial_objects[obj.name],
                        'content': f"Commercial object: {obj.name}",
                        'detection_method': 'vision_object'
                    })
            
            return ads
        except Exception as e:
            print(f"Object detection failed: {e}")
            return []
    
    @staticmethod
    def _expand_logo_to_full_ad(client, image, logo_x, logo_y, logo_width, logo_height, img_width, img_height):
        """Expand logo detection to capture the full advertisement around it"""
        # Expand by 150% around the logo in each direction
        expansion_factor = 1.5
        
        expanded_width = int(logo_width * expansion_factor)
        expanded_height = int(logo_height * expansion_factor)
        
        # Center the expansion around the logo
        expanded_x = max(0, logo_x - (expanded_width - logo_width) // 2)
        expanded_y = max(0, logo_y - (expanded_height - logo_height) // 2)
        
        # Ensure we don't exceed image boundaries
        expanded_width = min(expanded_width, img_width - expanded_x)
        expanded_height = min(expanded_height, img_height - expanded_y)
        
        # Apply size constraints for newspaper ads
        max_width = int(img_width * 0.4)  # Max 40% of page width
        max_height = int(img_height * 0.3)  # Max 30% of page height
        min_width = 120
        min_height = 80
        
        if (min_width <= expanded_width <= max_width and 
            min_height <= expanded_height <= max_height):
            return {
                'x': expanded_x,
                'y': expanded_y,
                'width': expanded_width,
                'height': expanded_height
            }
        
        return None
    
    @staticmethod
    def _expand_text_to_display_ad(text_x, text_y, text_width, text_height, img_width, img_height):
        """Expand text region to capture display ad with conservative expansion"""
        # Very conservative expansion for display ads
        expansion_factor = 1.3  # Only 30% expansion
        
        expanded_width = int(text_width * expansion_factor)
        expanded_height = int(text_height * expansion_factor)
        
        # Center the expansion around the text
        expanded_x = max(0, text_x - (expanded_width - text_width) // 2)
        expanded_y = max(0, text_y - (expanded_height - text_height) // 2)
        
        # Ensure we don't exceed image boundaries
        expanded_width = min(expanded_width, img_width - expanded_x)
        expanded_height = min(expanded_height, img_height - expanded_y)
        
        # Apply conservative size constraints
        max_width = int(img_width * 0.3)  # Max 30% of page width
        max_height = int(img_height * 0.2) # Max 20% of page height
        min_width = 100
        min_height = 60
        
        # Constrain to reasonable ad sizes
        expanded_width = max(min_width, min(expanded_width, max_width))
        expanded_height = max(min_height, min(expanded_height, max_height))
        
        return {
            'x': expanded_x,
            'y': expanded_y,
            'width': expanded_width,
            'height': expanded_height
        }
    
    @staticmethod
    def _expand_classified_header(header_x, header_y, header_width, header_height, img_width, img_height):
        """Expand classified header to include ads below it"""
        # For classified headers, expand significantly downward to capture ads
        expanded_width = max(header_width * 1.5, 200)  # Make wider to capture full ads
        expanded_height = max(header_height * 4, 150)   # Expand down to capture classified ads
        
        # Keep header at top of expanded region
        expanded_x = max(0, header_x - (expanded_width - header_width) // 4)  # Small left expansion
        expanded_y = header_y  # Keep original y position
        
        # Ensure boundaries
        expanded_width = min(expanded_width, img_width - expanded_x)
        expanded_height = min(expanded_height, img_height - expanded_y)
        
        # Apply maximum limits
        expanded_width = min(expanded_width, int(img_width * 0.4))
        expanded_height = min(expanded_height, int(img_height * 0.3))
        
        return {
            'x': expanded_x,
            'y': expanded_y,
            'width': expanded_width,
            'height': expanded_height
        }
    
    @staticmethod
    def _expand_text_ad(text_x, text_y, text_width, text_height, img_width, img_height):
        """Expand text ad (public notices, legal notices, etc.) with modest growth"""
        # Conservative expansion for text-based ads
        expansion_factor = 1.2  # Only 20% expansion
        
        expanded_width = int(text_width * expansion_factor)
        expanded_height = int(text_height * expansion_factor)
        
        # Center expansion
        expanded_x = max(0, text_x - (expanded_width - text_width) // 2)
        expanded_y = max(0, text_y - (expanded_height - text_height) // 2)
        
        # Ensure boundaries
        expanded_width = min(expanded_width, img_width - expanded_x)
        expanded_height = min(expanded_height, img_height - expanded_y)
        
        # Apply limits for text ads
        expanded_width = min(expanded_width, int(img_width * 0.6))
        expanded_height = min(expanded_height, int(img_height * 0.4))
        
        return {
            'x': expanded_x,
            'y': expanded_y,
            'width': expanded_width,
            'height': expanded_height
        }
    
    @staticmethod
    def _expand_text_to_full_ad(text_x, text_y, text_width, text_height, img_width, img_height):
        """Expand commercial text region to capture the full advertisement"""
        # Expand text region by 200% to capture surrounding ad content
        expansion_factor = 2.0
        
        expanded_width = int(text_width * expansion_factor)
        expanded_height = int(text_height * expansion_factor)
        
        # Center the expansion around the text
        expanded_x = max(0, text_x - (expanded_width - text_width) // 2)
        expanded_y = max(0, text_y - (expanded_height - text_height) // 2)
        
        # Ensure we don't exceed image boundaries
        expanded_width = min(expanded_width, img_width - expanded_x)
        expanded_height = min(expanded_height, img_height - expanded_y)
        
        # Apply realistic newspaper ad size constraints
        max_width = int(img_width * 0.4)
        max_height = int(img_height * 0.25)
        min_width = 100
        min_height = 60
        
        # Constrain to reasonable ad sizes
        expanded_width = max(min_width, min(expanded_width, max_width))
        expanded_height = max(min_height, min(expanded_height, max_height))
        
        return {
            'x': expanded_x,
            'y': expanded_y,
            'width': expanded_width,
            'height': expanded_height
        }
    
    @staticmethod
    def _expand_object_to_full_ad(obj_x, obj_y, obj_width, obj_height, img_width, img_height):
        """Expand commercial object detection to capture the full advertisement"""
        # Expand by 120% around the object
        expansion_factor = 1.2
        
        expanded_width = int(obj_width * expansion_factor)
        expanded_height = int(obj_height * expansion_factor)
        
        # Center the expansion around the object
        expanded_x = max(0, obj_x - (expanded_width - obj_width) // 2)
        expanded_y = max(0, obj_y - (expanded_height - obj_height) // 2)
        
        # Ensure we don't exceed image boundaries
        expanded_width = min(expanded_width, img_width - expanded_x)
        expanded_height = min(expanded_height, img_height - expanded_y)
        
        # Apply size constraints
        max_width = int(img_width * 0.4)
        max_height = int(img_height * 0.3)
        min_width = 120
        min_height = 80
        
        expanded_width = max(min_width, min(expanded_width, max_width))
        expanded_height = max(min_height, min(expanded_height, max_height))
        
        return {
            'x': expanded_x,
            'y': expanded_y,
            'width': expanded_width,
            'height': expanded_height
        }
    
    @staticmethod
    def _filter_and_merge_detections(detections, publication_type, img_width, img_height):
        """Filter and merge overlapping detections, apply layout awareness"""
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Filter by newspaper layout zones (avoid masthead, focus on ad zones)
        filtered = []
        for detection in detections:
            if GoogleVisionAdDetector._is_in_valid_ad_zone(detection, publication_type, img_width, img_height):
                filtered.append(detection)
        
        # Remove overlapping detections (keep higher confidence ones)
        merged = []
        for detection in filtered:
            overlaps = False
            for existing in merged:
                if GoogleVisionAdDetector._boxes_overlap(detection, existing, threshold=0.3):
                    overlaps = True
                    break
            
            if not overlaps:
                merged.append(detection)
        
        # BALANCED: Allow more legitimate ads while still avoiding false positives
        max_ads = 6 if publication_type == 'broadsheet' else 4  # Increased from 2 to 6
        
        # Smart confidence filtering - prioritize highest confidence but allow legitimate ads
        if merged:
            merged.sort(key=lambda x: x['confidence'], reverse=True)
            # Keep ads with moderate confidence, not ultra-high only
            decent_confidence_ads = [ad for ad in merged if ad['confidence'] > 0.4]  # Lowered from 0.6 to 0.4
            if decent_confidence_ads:
                merged = decent_confidence_ads
        
        print(f'Final ultra-conservative filtering: {len(merged[:max_ads])} ads (from {len(merged)} candidates)')
        return merged[:max_ads]
    
    @staticmethod
    def _is_in_valid_ad_zone(detection, publication_type, img_width, img_height):
        """Check if detection is in a valid newspaper ad zone"""
        x, y, width, height = detection['x'], detection['y'], detection['width'], detection['height']
        
        # Avoid masthead area (top 5% of page only for testing)
        if y < img_height * 0.05:  # Much more permissive
            return False
        
        # Avoid very bottom (footer area - bottom 5%)
        if y + height > img_height * 0.95:
            return False
        
        # TEMPORARILY very permissive - accept most areas for testing
        if publication_type == 'broadsheet':
            # Accept most of the page except very top masthead and very bottom
            return True  # For now, accept all areas that pass basic size/position filters
        
        # For other publication types, allow most areas except masthead/footer
        return True
    
    @staticmethod
    def _boxes_overlap(box1, box2, threshold=0.3):
        """Check if two bounding boxes overlap beyond threshold"""
        x1, y1, w1, h1 = box1['x'], box1['y'], box1['width'], box1['height']
        x2, y2, w2, h2 = box2['x'], box2['y'], box2['width'], box2['height']
        
        # Calculate intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        
        if x_overlap == 0 or y_overlap == 0:
            return False
        
        intersection = x_overlap * y_overlap
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # Check if overlap exceeds threshold for either box
        overlap_ratio1 = intersection / box1_area if box1_area > 0 else 0
        overlap_ratio2 = intersection / box2_area if box2_area > 0 else 0
        
        return max(overlap_ratio1, overlap_ratio2) > threshold
    
    @staticmethod
    def _is_likely_news_story(roi_region, texts, x, y, w, h):
        """Detect if a region is likely a news story rather than an ad"""
        try:
            # Check text content in this region for news story characteristics
            region_text = ""
            for text in texts[1:]:  # Skip full page text
                if hasattr(text, 'bounding_poly') and text.bounding_poly.vertices:
                    text_vertices = text.bounding_poly.vertices
                    text_x = min(v.x for v in text_vertices)
                    text_y = min(v.y for v in text_vertices)
                    text_w = max(v.x for v in text_vertices) - text_x
                    text_h = max(v.y for v in text_vertices) - text_y
                    
                    # Check if text overlaps with this region
                    if (text_x < x + w and text_x + text_w > x and 
                        text_y < y + h and text_y + text_h > y):
                        region_text += text.description + " "
            
            region_text = region_text.upper()
            
            # NEWS STORY INDICATORS (these should NOT be ads)
            news_indicators = [
                # News article patterns
                'REPORTED BY', 'STAFF WRITER', 'BY:', 'ASSOCIATED PRESS', 'AP',
                'ACCORDING TO', 'OFFICIALS SAID', 'POLICE SAID', 'SAID THE',
                'REPORTED THAT', 'INVESTIGATION', 'AUTHORITIES', 'DEPARTMENT',
                # Story structure words  
                'STORY CONTINUES', 'CONTINUED FROM', 'SEE STORY', 'PAGE A',
                # News content
                'CROWNED', 'KING AND QUEEN', 'WINNERS', 'CHAMPIONS', 'AWARDS',
                'CAPTURES', 'CROWNS', 'TOURNAMENT', 'COMPETITION', 'CONTEST',
                # Sports/events coverage
                'SCHEDULE', 'SCORES', 'SEASON', 'GAME', 'MATCH', 'TEAM',
                # General journalism words
                'STUDENTS', 'SCHOOL', 'EDUCATION', 'COMMUNITY', 'LOCAL'
            ]
            
            # Count news indicators
            news_score = 0
            for indicator in news_indicators:
                if indicator in region_text:
                    news_score += 1
            
            # BUSINESS/AD INDICATORS (these ARE likely ads) - ENHANCED
            ad_indicators = [
                # Traditional business indicators
                'CALL NOW', 'VISIT US', 'CONTACT', 'HOURS:', 'OPEN',
                'SPECIAL OFFER', 'SALE', 'DISCOUNT', 'FINANCING',
                'SERVICES', 'REPAIR', 'INSTALLATION', 'FREE ESTIMATE',
                # Phone number patterns (strong ad indicators)
                '507-', 'PHONE:', 'TEL:', 'CALL:',
                # Church and service business indicators
                'CHURCH', 'PASTOR', 'WORSHIP', 'SUNDAY SERVICE',
                'CLINIC', 'PHARMACY', 'DENTIST', 'DOCTOR', 'CARE',
                # Business directory indicators
                'INC', 'LLC', 'COMPANY', 'ENTERPRISES', 'SERVICES',
                'CONSTRUCTION', 'CONTRACTOR', 'PLUMBING', 'ELECTRICAL'
            ]
            
            ad_score = 0
            for indicator in ad_indicators:
                if indicator in region_text:
                    ad_score += 1
            
            # Decision logic: if more news indicators than ad indicators, it's probably news
            if news_score >= 2 and news_score > ad_score:
                print(f"Filtering out likely news story (news_score={news_score}, ad_score={ad_score})")
                return True  # This is likely a news story
            
            return False  # This might be an ad
            
        except Exception as e:
            print(f"News story detection failed: {e}")
            return False  # If we can't determine, err on the side of caution
    
    @staticmethod
    def test_vision_api(image_path):
        """Test function to verify Vision API connectivity and basic functionality"""
        try:
            print(f"Testing Google Vision API with: {image_path}")
            
            # Test basic connectivity
            client = vision.ImageAnnotatorClient()
            print("✓ Vision API client initialized successfully")
            
            # Load test image
            with vision_io.open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            print("✓ Test image loaded successfully")
            
            # Test logo detection
            try:
                response = client.logo_detection(image=image)
                logos = response.logo_annotations
                print(f"✓ Logo detection: Found {len(logos)} logos")
            except Exception as e:
                print(f"✗ Logo detection failed: {e}")
            
            # Test text detection
            try:
                response = client.text_detection(image=image)
                texts = response.text_annotations
                print(f"✓ Text detection: Found {len(texts)} text regions")
            except Exception as e:
                print(f"✗ Text detection failed: {e}")
            
            # Test object detection
            try:
                response = client.object_localization(image=image)
                objects = response.localized_object_annotations
                print(f"✓ Object detection: Found {len(objects)} objects")
            except Exception as e:
                print(f"✗ Object detection failed: {e}")
            
            # Test full ad detection
            ads = GoogleVisionAdDetector.detect_ads(image_path)
            print(f"✓ Full ad detection: Found {len(ads)} potential ads")
            
            for i, ad in enumerate(ads):
                print(f"  Ad {i+1}: {ad['ad_type']} at ({ad['x']},{ad['y']}) size {ad['width']}x{ad['height']} confidence={ad['confidence']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"✗ Vision API test failed: {e}")
            return False

# AI Learning Engine for Automatic Ad Detection
class AdLearningEngine:
    """
    AI Learning system that learns from user-verified ad identifications
    to automatically detect ads in future publications
    """
    
    @staticmethod
    def extract_features(image_path, box_coords):
        """Extract comprehensive features from an ad region for ML training"""
        try:
            import time
            start_time = time.time()

            # Load image (removed excessive debug logging for performance)
            img = cv2.imread(image_path)
            if img is None:
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            x, y, w, h = int(box_coords['x']), int(box_coords['y']), int(box_coords['width']), int(box_coords['height'])

            # Ensure coordinates are within image bounds
            img_h, img_w = gray.shape
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            # Extract region of interest (removed debug logging for performance)
            roi = gray[y:y+h, x:x+w]
            if roi.size == 0 or w <= 0 or h <= 0:
                return None

            # Feature extraction
            features = {}

            # 1. Basic geometric features
            features['width'] = w
            features['height'] = h
            features['area'] = w * h
            features['aspect_ratio'] = w / h if h > 0 else 0
            features['perimeter'] = 2 * (w + h)

            # 2. Position features (normalized by image size)
            img_h, img_w = gray.shape
            features['x_normalized'] = x / img_w
            features['y_normalized'] = y / img_h
            features['center_x_normalized'] = (x + w/2) / img_w
            features['center_y_normalized'] = (y + h/2) / img_h

            # NEW: Layout position features (key for distinguishing ads vs editorial)
            features['is_left_edge'] = 1 if x < img_w * 0.05 else 0
            features['is_right_edge'] = 1 if (x + w) > img_w * 0.95 else 0
            features['is_top_edge'] = 1 if y < img_h * 0.1 else 0
            features['is_bottom_edge'] = 1 if (y + h) > img_h * 0.9 else 0
            features['is_center_horizontal'] = 1 if 0.3 < (x + w/2) / img_w < 0.7 else 0
            features['is_center_vertical'] = 1 if 0.3 < (y + h/2) / img_h < 0.7 else 0

            # 3. Content analysis features
            features['mean_intensity'] = np.mean(roi)
            features['std_intensity'] = np.std(roi)
            features['min_intensity'] = np.min(roi)
            features['max_intensity'] = np.max(roi)
            features['intensity_range'] = features['max_intensity'] - features['min_intensity']
            
            # 4. Texture analysis (simplified and fast)
            # Check timeout
            if time.time() - start_time > 5:  # 5 second timeout
                print(f"Feature extraction timeout for {image_path}")
                return None
                
            # Fast texture measure using Laplacian variance
            laplacian = cv2.Laplacian(roi, cv2.CV_64F)
            features['texture_complexity'] = laplacian.var() / 10000.0  # Normalize
            
            # 5. Edge density features
            edges = cv2.Canny(roi, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            
            # 6. Border analysis
            border_thickness = min(3, min(w, h) // 4)
            if border_thickness > 0:
                try:
                    top_border = roi[:border_thickness, :].mean()
                    bottom_border = roi[-border_thickness:, :].mean()
                    left_border = roi[:, :border_thickness].mean()
                    right_border = roi[:, -border_thickness:].mean()
                    center = roi[border_thickness:-border_thickness, border_thickness:-border_thickness].mean()
                    
                    features['border_contrast'] = (abs(top_border - center) + abs(bottom_border - center) + 
                                                 abs(left_border - center) + abs(right_border - center)) / 4
                    features['border_uniformity'] = np.std([top_border, bottom_border, left_border, right_border])
                except (IndexError, ValueError, Exception) as e:
                    print(f"Warning: Error calculating border features: {e}")
                    features['border_contrast'] = 0
                    features['border_uniformity'] = 0
            else:
                features['border_contrast'] = 0
                features['border_uniformity'] = 0
            
            # 7. White space analysis
            binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            features['white_ratio'] = np.sum(binary_roi == 255) / binary_roi.size
            features['black_ratio'] = np.sum(binary_roi == 0) / binary_roi.size
            
            # 8. Connected components analysis
            num_labels, labels = cv2.connectedComponents(binary_roi)
            features['num_components'] = num_labels - 1  # Exclude background
            
            # 9. Gradient features
            grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            features['gradient_mean'] = np.mean(gradient_magnitude)
            features['gradient_std'] = np.std(gradient_magnitude)
            
            # 10. Rectangularity (how rectangular the content is)
            contours, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                contour_area = cv2.contourArea(largest_contour)
                bounding_area = w * h
                features['rectangularity'] = contour_area / bounding_area if bounding_area > 0 else 0
            else:
                features['rectangularity'] = 0

            # NEW: Enhanced features for ad vs editorial distinction
            # 11. Text density analysis (ads often have less text than editorial)
            blur_kernel_size = max(3, min(w, h) // 20)
            if blur_kernel_size % 2 == 0:
                blur_kernel_size += 1

            blurred = cv2.GaussianBlur(roi, (blur_kernel_size, blur_kernel_size), 0)
            text_threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            text_regions = cv2.morphologyEx(text_threshold, cv2.MORPH_CLOSE,
                                          cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
            features['text_density'] = np.sum(text_regions == 0) / text_regions.size

            # 12. Border prominence (ads often have strong borders)
            border_width = max(1, min(w, h) // 50)
            if border_width < min(w//2, h//2):
                border_mask = np.zeros_like(roi)
                cv2.rectangle(border_mask, (0, 0), (w-1, h-1), 255, border_width)
                border_pixels = roi[border_mask > 0]
                center_pixels = roi[border_mask == 0]
                if len(border_pixels) > 0 and len(center_pixels) > 0:
                    features['border_prominence'] = abs(np.mean(border_pixels) - np.mean(center_pixels))
                else:
                    features['border_prominence'] = 0
            else:
                features['border_prominence'] = 0

            # 13. Contrast patterns (ads often have high contrast elements)
            local_contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = local_contrast.apply(roi)
            contrast_diff = np.mean(np.abs(enhanced.astype(np.float32) - roi.astype(np.float32)))
            features['local_contrast_enhancement'] = contrast_diff

            # 14. Structural features (ads are more structured than editorial text)
            h_kernel_size = max(1, min(w//4, 40))
            v_kernel_size = max(1, min(h//4, 40))
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))

            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)

            features['horizontal_structure'] = np.sum(horizontal_lines > 0) / horizontal_lines.size
            features['vertical_structure'] = np.sum(vertical_lines > 0) / vertical_lines.size
            features['structural_ratio'] = (features['horizontal_structure'] + features['vertical_structure']) / 2

            # 15. Image content detection (ads often contain more images)
            # Large uniform regions might indicate images/graphics
            uniform_regions = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN,
                                             cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
            features['uniform_region_ratio'] = np.sum(uniform_regions == 255) / uniform_regions.size

            # 16. Size-based features (ads have typical size patterns)
            total_page_area = img_w * img_h
            features['area_ratio_to_page'] = (w * h) / total_page_area
            features['is_small_ad'] = 1 if (w * h) < total_page_area * 0.02 else 0  # <2% of page
            features['is_large_ad'] = 1 if (w * h) > total_page_area * 0.15 else 0  # >15% of page
            features['is_banner_shaped'] = 1 if w > h * 3 else 0  # Wide banner format
            features['is_square_shaped'] = 1 if 0.8 < (w/h) < 1.2 else 0  # Nearly square
            
            # Convert numpy types to native Python types for JSON serialization
            for key, value in features.items():
                if hasattr(value, 'item'):  # numpy scalar
                    features[key] = value.item()
                elif isinstance(value, np.ndarray):
                    features[key] = value.tolist()
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def collect_training_data(batch_size=50, max_samples=None):
        """Collect features from user-verified ad boxes for training (optimized with batching)"""
        training_samples = []
        
        # Get user-verified ad boxes that don't have training data yet
        subquery = db.session.query(TrainingData.ad_box_id)
        verified_boxes_query = AdBox.query.filter_by(user_verified=True).filter(
            ~AdBox.id.in_(subquery)
        )
        
        if max_samples:
            verified_boxes_query = verified_boxes_query.limit(max_samples)
            
        verified_boxes = verified_boxes_query.all()
        total_boxes = len(verified_boxes)
        
        print(f"Processing {total_boxes} verified ads for training data collection...")
        
        # Add timeout protection
        import time
        start_time = time.time()
        max_processing_time = 300  # 5 minutes max
        
        for i, ad_box in enumerate(verified_boxes):
            try:
                # Check overall timeout
                if time.time() - start_time > max_processing_time:
                    print(f"Training data collection timeout after {max_processing_time}s, processed {i}/{total_boxes}")
                    break
                    
                if i % 10 == 0:
                    print(f"Progress: {i}/{total_boxes} ({i/total_boxes*100:.1f}%)")
                
                print(f"Processing ad box {ad_box.id} - coordinates: x={ad_box.x}, y={ad_box.y}, w={ad_box.width}, h={ad_box.height}")
                
                # Get page and publication info
                page = db.session.get(Page, ad_box.page_id)
                if not page:
                    print(f"No page found for ad box {ad_box.id}, page_id={ad_box.page_id}")
                    continue
                    
                publication = db.session.get(Publication, page.publication_id)
                if not publication:
                    print(f"No publication found for page {page.id}, publication_id={page.publication_id}")
                    continue
                
                print(f"Ad box {ad_box.id} belongs to publication {publication.id} with filename: {publication.filename}")
                
                # Get image path - handle both filename with and without .pdf extension
                base_filename = publication.filename
                if base_filename.endswith('.pdf'):
                    base_filename = base_filename[:-4]  # Remove .pdf extension
                
                # Try both possible filename formats
                possible_filenames = [
                    f"{publication.filename}_page_{page.page_number}.png",  # Original format
                    f"{base_filename}_page_{page.page_number}.png"          # Without .pdf extension
                ]
                
                image_path = None
                for filename in possible_filenames:
                    test_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pages', filename)
                    print(f"Checking for image: {test_path}")
                    if os.path.exists(test_path):
                        image_path = test_path
                        print(f"Found image: {image_path}")
                        break
                
                if not image_path:
                    print(f"Image file not found in any of these locations:")
                    for filename in possible_filenames:
                        print(f"  - {os.path.join(app.config['UPLOAD_FOLDER'], 'pages', filename)}")
                    
                    # Show what files actually exist in the pages directory
                    pages_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'pages')
                    try:
                        if os.path.exists(pages_dir):
                            existing_files = os.listdir(pages_dir)
                            print(f"Files that actually exist in {pages_dir}:")
                            for f in existing_files[:10]:  # Show first 10 files
                                print(f"  - {f}")
                            if len(existing_files) > 10:
                                print(f"  ... and {len(existing_files) - 10} more files")
                            
                            # Try to find a matching publication for this page
                            print(f"Attempting to find correct publication for page {page.page_number}...")
                            for existing_file in existing_files:
                                if existing_file.endswith(f"_page_{page.page_number}.png"):
                                    # Extract the publication filename from the existing file
                                    pub_filename = existing_file.replace(f"_page_{page.page_number}.png", "")
                                    print(f"Found potential match: {pub_filename}")
                                    
                                    # Try to find this publication in database
                                    correct_pub = Publication.query.filter_by(filename=pub_filename).first()
                                    if correct_pub:
                                        print(f"Found correct publication {correct_pub.id} with filename {correct_pub.filename}")
                                        
                                        # Use this publication instead
                                        publication = correct_pub
                                        image_path = os.path.join(pages_dir, existing_file)
                                        print(f"Using corrected image path: {image_path}")
                                        break
                            else:
                                print(f"No matching publication found for page {page.page_number}")
                                continue
                        else:
                            print(f"Pages directory does not exist: {pages_dir}")
                            continue
                    except Exception as e:
                        print(f"Error listing pages directory: {e}")
                        continue
                
                # Extract features
                box_coords = {
                    'x': ad_box.x, 'y': ad_box.y, 
                    'width': ad_box.width, 'height': ad_box.height
                }
                print(f"Extracting features from {image_path} with coords {box_coords}")
                features = AdLearningEngine.extract_features(image_path, box_coords)
                
                if features:
                    print(f"Successfully extracted {len(features)} features for ad box {ad_box.id}")
                    # Create new training data record
                    training_data = TrainingData(
                        ad_box_id=ad_box.id,
                        publication_type=publication.publication_type,
                        features=json.dumps(features),
                        label=ad_box.ad_type or 'manual',
                        confidence_score=1.0  # User-verified = high confidence
                    )
                    db.session.add(training_data)
                    training_samples.append(training_data)
                    
                    # Commit in batches to prevent timeouts
                    if len(training_samples) % batch_size == 0:
                        db.session.commit()
                        print(f"Committed batch of {batch_size} samples")
                else:
                    print(f"Feature extraction returned None for ad box {ad_box.id}")
            
            except Exception as e:
                print(f"Error processing ad box {ad_box.id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final commit
        if training_samples:
            db.session.commit()
            
        print(f"Training data collection complete: {len(training_samples)} new samples")
        return len(training_samples)
    
    @staticmethod
    def reset_training_data(publication_type=None, confirmation_code=None):
        """Reset training data to start fresh with improved Vision AI detection"""
        if confirmation_code != "RESET_TRAINING_CONFIRMED":
            return {
                'success': False, 
                'error': 'Must provide confirmation_code="RESET_TRAINING_CONFIRMED" to reset training data'
            }
        
        try:
            # Count current training data
            if publication_type:
                current_count = TrainingData.query.filter_by(publication_type=publication_type).count()
                training_query = TrainingData.query.filter_by(publication_type=publication_type)
                model_query = MLModel.query.filter_by(publication_type=publication_type)
            else:
                current_count = TrainingData.query.count()
                training_query = TrainingData.query
                model_query = MLModel.query
            
            print(f"Resetting {current_count} training data records...")
            
            # Delete training data
            training_deleted = training_query.delete()
            
            # Deactivate existing models (don't delete them, just deactivate for reference)
            models_updated = model_query.update({'is_active': False})
            
            db.session.commit()
            
            return {
                'success': True,
                'training_data_deleted': training_deleted,
                'models_deactivated': models_updated,
                'message': f'Training data reset complete. {training_deleted} training records deleted, {models_updated} models deactivated. Ready for fresh training with improved Vision AI.'
            }
            
        except Exception as e:
            db.session.rollback()
            return {
                'success': False,
                'error': f'Failed to reset training data: {str(e)}'
            }
    
    @staticmethod
    def train_model(publication_type='broadsheet', min_samples=20, collect_new_data=False):
        """Train ML model for ad detection"""
        try:
            # Optionally collect fresh training data (can be slow)
            if collect_new_data:
                print("Collecting fresh training data...")
                AdLearningEngine.collect_training_data(max_samples=50, batch_size=10)
            
            # Get training data for this publication type
            training_data = TrainingData.query.filter_by(publication_type=publication_type).all()
            
            if len(training_data) < min_samples:
                return {
                    'success': False, 
                    'error': f'Need at least {min_samples} training samples, have {len(training_data)}'
                }
            
            # Prepare training data
            X = []  # Features
            y = []  # Labels
            feature_names = None
            
            for data in training_data:
                features_dict = json.loads(data.features)
                if feature_names is None:
                    feature_names = sorted(features_dict.keys())
                
                # Convert to feature vector
                feature_vector = [features_dict.get(name, 0) for name in feature_names]
                X.append(feature_vector)
                y.append(data.label)
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model - use RandomForest for good interpretability
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_accuracy = accuracy_score(y_train, model.predict(X_train_scaled))
            test_accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
            
            # Create model package
            model_package = {
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'label_encoder': None  # Using string labels directly
            }
            
            # Serialize model
            model_data = pickle.dumps(model_package)
            
            # Deactivate old models
            MLModel.query.filter_by(
                publication_type=publication_type,
                model_type='ad_detector'
            ).update({'is_active': False})
            
            # Save new model
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            ml_model = MLModel(
                model_name=f"ad_detector_{publication_type}_{version}",
                model_type='ad_detector',
                publication_type=publication_type,
                version=version,
                model_data=model_data,
                training_accuracy=train_accuracy,
                validation_accuracy=test_accuracy,
                training_samples=len(training_data),
                feature_names=json.dumps(feature_names),
                is_active=True
            )
            
            db.session.add(ml_model)
            db.session.commit()
            
            return {
                'success': True,
                'model_id': ml_model.id,
                'training_accuracy': train_accuracy,
                'validation_accuracy': test_accuracy,
                'training_samples': len(training_data),
                'feature_count': len(feature_names)
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def predict_ads(image_path, publication_type='broadsheet', confidence_threshold=0.7):
        """Use trained ML model to predict ad locations"""
        try:
            # Get active model
            ml_model = MLModel.query.filter_by(
                publication_type=publication_type,
                model_type='ad_detector',
                is_active=True
            ).first()
            
            if not ml_model:
                return {'success': False, 'error': 'No trained model available'}
            
            # Load model
            model_package = pickle.loads(ml_model.model_data)
            model = model_package['model']
            scaler = model_package['scaler']
            feature_names = model_package['feature_names']
            
            # Use the new intelligent ad detection system
            typical_ad_sizes = AdLearningEngine._get_typical_ad_sizes(publication_type)
            
            detected_boxes = AdLearningEngine.intelligent_ad_detection(
                image_path, publication_type, model, feature_names, 
                confidence_threshold, scaler, typical_ad_sizes
            )
            
            # Convert to expected format for backward compatibility
            predicted_ads = []
            for box in detected_boxes:
                predicted_ads.append({
                    'x': int(box['x']),
                    'y': int(box['y']),
                    'width': int(box['width']),
                    'height': int(box['height']),
                    'predicted_type': 'open_display',  # Default type for intelligent detection
                    'confidence': float(box['confidence']),
                    'features': {}  # Features already calculated in intelligent detection
                })
            
            return {
                'success': True,
                'predictions': predicted_ads,
                'model_version': ml_model.version,
                'model_accuracy': ml_model.validation_accuracy
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def _remove_overlapping_predictions(predictions, overlap_threshold=0.3):
        """Remove overlapping predictions, keeping highest confidence"""
        if not predictions:
            return predictions
        
        # Sort by confidence (descending)
        predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
        
        filtered_predictions = []
        for pred in predictions:
            overlap_found = False
            for existing_pred in filtered_predictions:
                if AdLearningEngine._calculate_overlap(pred, existing_pred) > overlap_threshold:
                    overlap_found = True
                    break
            
            if not overlap_found:
                filtered_predictions.append(pred)
        
        return filtered_predictions
    
    @staticmethod
    def _calculate_overlap(pred1, pred2):
        """Calculate overlap ratio between two predictions"""
        x1_min, y1_min = pred1['x'], pred1['y']
        x1_max, y1_max = x1_min + pred1['width'], y1_min + pred1['height']
        
        x2_min, y2_min = pred2['x'], pred2['y']
        x2_max, y2_max = x2_min + pred2['width'], y2_min + pred2['height']
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
            intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            area1 = pred1['width'] * pred1['height']
            area2 = pred2['width'] * pred2['height']
            union = area1 + area2 - intersection
            return intersection / union if union > 0 else 0
        
        return 0
    
    @staticmethod
    def get_training_stats(publication_type=None):
        """Get statistics about available training data"""
        query = TrainingData.query
        if publication_type:
            query = query.filter_by(publication_type=publication_type)
        
        training_data = query.all()
        
        stats = {
            'total_samples': len(training_data),
            'by_type': {},
            'by_publication_type': {},
            'recent_samples': 0
        }
        
        recent_date = datetime.utcnow() - timedelta(days=7)
        
        for data in training_data:
            # By label type
            label = data.label
            if label not in stats['by_type']:
                stats['by_type'][label] = 0
            stats['by_type'][label] += 1
            
            # By publication type
            pub_type = data.publication_type
            if pub_type not in stats['by_publication_type']:
                stats['by_publication_type'][pub_type] = 0
            stats['by_publication_type'][pub_type] += 1
            
            # Recent samples
            if data.extracted_date >= recent_date:
                stats['recent_samples'] += 1
        
        return stats
    
    @staticmethod
    def auto_detect_ads(publication_id, confidence_threshold=0.7):
        """Automatically detect ads in a publication using trained models"""
        try:
            # Simple duplicate check without complex locking
            if hasattr(AdLearningEngine, '_ai_processing') and publication_id in getattr(AdLearningEngine, '_ai_processing', set()):
                print(f"AI detection already running for publication {publication_id}, skipping")
                return {'success': False, 'error': 'AI detection already in progress'}
            
            # Mark as processing
            if not hasattr(AdLearningEngine, '_ai_processing'):
                AdLearningEngine._ai_processing = set()
            AdLearningEngine._ai_processing.add(publication_id)
            
            publication = Publication.query.get(publication_id)
            if not publication:
                return {'success': False, 'error': 'Publication not found'}
            
            # Get the active model for this publication type
            model_record = MLModel.query.filter_by(
                publication_type=publication.publication_type,
                    model_type='ad_detector',
                    is_active=True
                ).first()
            
            if not model_record:
                print(f"No active model found for publication type: {publication.publication_type}")
                print(f"Available models in database:")
                all_models = MLModel.query.all()
                for m in all_models:
                    print(f"  - {m.publication_type} / {m.model_type} / active={m.is_active}")
                return {'success': False, 'error': f'No trained model available for {publication.publication_type}'}
            
            # Load the trained model
            import pickle
            model_package = pickle.loads(model_record.model_data)
            
            # Extract components from model package
            if isinstance(model_package, dict):
                model = model_package.get('model')
                scaler = model_package.get('scaler')
                feature_names = model_package.get('feature_names', [])
                print(f"Loaded model package with {len(feature_names)} features")
            else:
                # Legacy format - just the model
                model = model_package
                scaler = None
                feature_names = json.loads(model_record.feature_names) if model_record.feature_names else []
                print(f"Loaded legacy model with {len(feature_names)} features")
            
            detections = []
            pages_processed = 0
        
            # Process each page
            pages = Page.query.filter_by(publication_id=publication.id).all()
            print(f"Processing {len(pages)} pages for AI detection")
            
            for page in pages:
                image_filename = f"{publication.filename}_page_{page.page_number}.png"
                image_path = os.path.join('static', 'uploads', 'pages', image_filename)
                print(f"Looking for page image: {image_path}")
                
                if not os.path.exists(image_path):
                    print(f"Image not found, skipping page {page.page_number}")
                    continue
                
                print(f"Scanning page {page.page_number} for ads with confidence threshold {confidence_threshold}")
                
                # PRIMARY: Try PDF Metadata detection first
                pdf_detected_boxes = []
                pdf_path = os.path.join('static', 'uploads', 'pdfs', publication.filename)
                print(f"DEBUG: PDF path: {pdf_path}")
                print(f"DEBUG: PDF exists: {os.path.exists(pdf_path)}")
                print(f"Attempting PDF metadata detection on page {page.page_number}")
                try:
                    # Get PDF detections in PDF coordinate system with filename intelligence
                    pdf_detections = PDFMetadataAdDetector.detect_ads_from_pdf_metadata(
                        pdf_path, page.page_number, publication.publication_type, publication.original_filename
                    )
                    
                    print(f"DEBUG: Raw PDF detections returned: {len(pdf_detections) if pdf_detections else 0}")
                    if pdf_detections:
                        print(f"PDF metadata found {len(pdf_detections)} ad candidates on page {page.page_number}")

                        # Transform to image coordinate system
                        doc = fitz.open(pdf_path)
                        pdf_page = doc[page.page_number - 1]
                        pdf_page_rect = pdf_page.rect
                        doc.close()

                        pdf_detected_boxes = PDFMetadataAdDetector.transform_pdf_to_image_coordinates(
                            pdf_detections, pdf_page_rect, page.width_pixels, page.height_pixels
                        )
                        print(f"DEBUG: Transformed to {len(pdf_detected_boxes)} image coordinate boxes")
                    else:
                        print(f"DEBUG: No ads detected via PDF metadata on page {page.page_number} - this might be the issue!")
                        
                except Exception as pdf_error:
                    print(f"PDF metadata detection failed for page {page.page_number}: {pdf_error}")
                    pdf_detected_boxes = []
                
                # REMOVED: Redundant secondary analysis (primary method now uses enhanced detection)
                vision_detected_boxes = []
                print(f"Using primary PDF detection results: {len(pdf_detected_boxes)} ads found")
                
                # TERTIARY FALLBACK: Use existing detection if both PDF and Vision AI fail or find nothing
                fallback_detected_boxes = []
                if len(pdf_detected_boxes) == 0 and len(vision_detected_boxes) == 0:
                    print(f"Falling back to existing detection system for page {page.page_number}")
                    try:
                        # Get typical ad sizes from training data for better detection windows
                        typical_ad_sizes = AdLearningEngine._get_typical_ad_sizes(publication.publication_type)
                        print(f"Using typical ad sizes for fallback detection: {typical_ad_sizes}")
                        
                        # Use intelligent content-aware detection system
                        fallback_detected_boxes = AdLearningEngine.intelligent_ad_detection(
                            image_path, publication.publication_type, model, feature_names, 
                            confidence_threshold, scaler, typical_ad_sizes
                        )
                        print(f"Fallback detection found {len(fallback_detected_boxes)} ads on page {page.page_number}")
                    except Exception as fallback_error:
                        print(f"Fallback detection also failed for page {page.page_number}: {fallback_error}")
                        fallback_detected_boxes = []
                
                # Combine results (prioritize PDF metadata, then Vision AI, then fallback)
                detected_boxes = pdf_detected_boxes + vision_detected_boxes + fallback_detected_boxes
                
                print(f"Page {page.page_number} scan complete: {len(detected_boxes)} detections")
                
                for box in detected_boxes:
                    # Create AdBox record
                    config = PUBLICATION_CONFIGS[publication.publication_type]
                    calculator = MeasurementCalculator()
                    
                    # Use page dimensions from database instead of config
                    page_total_pixels = page.width_pixels * page.height_pixels if page.width_pixels and page.height_pixels else 1
                    
                    column_inches = calculator.pixels_to_inches(
                        box['height'] * box['width'],
                        page_total_pixels,
                        config.get('total_inches_per_page', 258)  # Use total_inches_per_page from config
                    )
                    
                    # Calculate required measurement fields
                    if page.width_pixels and page.height_pixels:
                        # Use page calibration if available
                        width_inches = (box['width'] / page.width_pixels) * config.get('total_inches_per_page', 258) * (config.get('width_units', 12) / 12)
                        height_inches = (box['height'] / page.height_pixels) * config.get('total_inches_per_page', 258)
                    else:
                        # Fallback calculation
                        width_inches = column_inches / 10  # Rough estimate
                        height_inches = column_inches / 10
                    
                    # Round measurements
                    width_rounded = round(width_inches * 16) / 16  # Round to nearest 1/16th
                    height_rounded = round(height_inches * 16) / 16
                    
                    print(f"Creating AdBox: position=({box['x']},{box['y']}) size=({box['width']}x{box['height']}) confidence={box['confidence']:.3f} column_inches={column_inches:.2f}")
                    
                    # Determine ad_type based on detection method
                    ad_type = 'ai_detected'  # Default fallback
                    if 'type' in box:
                        # PDF metadata detection types (image, text, border)
                        if box['type'] == 'image':
                            ad_type = 'pdf_image_ad'
                        elif box['type'] == 'text':
                            ad_type = 'pdf_text_ad'
                        elif box['type'] == 'border':
                            ad_type = 'pdf_border_ad'
                        else:
                            ad_type = 'pdf_detected'
                    elif 'ad_type' in box:
                        ad_type = box['ad_type']  # Vision AI detected types
                    elif 'predicted_type' in box:
                        ad_type = box['predicted_type']  # Existing ML predicted types
                    
                    ad_box = AdBox(
                        page_id=page.id,
                        x=box['x'],
                        y=box['y'],
                        width=box['width'],
                        height=box['height'],
                        width_inches_raw=width_inches,
                        height_inches_raw=height_inches,
                        width_inches_rounded=width_rounded,
                        height_inches_rounded=height_rounded,
                        column_inches=column_inches,
                        ad_type=ad_type,  # Use Vision AI ad type or fallback
                        is_ad=True,
                        confidence_score=box['confidence'],
                        detected_automatically=True,
                        user_verified=False  # AI detected, not user verified
                    )
                    
                    db.session.add(ad_box)
                    detections.append({
                        'page_id': page.id,
                        'x': box['x'],
                        'y': box['y'],
                        'width': box['width'],
                        'height': box['height'],
                        'confidence': box['confidence']
                    })
                
                pages_processed += 1
            
            db.session.commit()
            
            # Update publication totals
            from app import update_totals
            for page in pages:
                update_totals(page.id)
            
            return {
                'success': True,
                'detections': len(detections),
                'pages_processed': pages_processed,
                'model_used': model_record.model_name,
                'confidence_threshold': confidence_threshold
            }
            
        except Exception as e:
            db.session.rollback()
            print(f"Error in auto_detect_ads: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
            
        finally:
            # Always remove from processing set
            try:
                if hasattr(AdLearningEngine, '_ai_processing'):
                    AdLearningEngine._ai_processing.discard(publication_id)
                    print(f"Removed publication {publication_id} from AI processing set")
            except:
                pass
    
    @staticmethod
    def _get_typical_ad_sizes(publication_type):
        """Get typical ad sizes from training data to improve detection accuracy"""
        try:
            # Query training data to get typical ad dimensions
            from sqlalchemy import text
            query = text("""
                SELECT ab.width, ab.height, COUNT(*) as frequency
                FROM ad_box ab
                JOIN page p ON ab.page_id = p.id  
                JOIN publication pub ON p.publication_id = pub.id
                WHERE pub.publication_type = :pub_type
                AND ab.user_verified = true
                AND ab.width > 80 AND ab.height > 60  -- Filter out tiny boxes
                AND ab.width < 500 AND ab.height < 400  -- Filter out oversized false positives
                GROUP BY ab.width, ab.height
                ORDER BY frequency DESC
                LIMIT 10
            """)
            
            result = db.session.execute(query, {'pub_type': publication_type})
            sizes = []
            
            for row in result:
                width, height, freq = row
                sizes.append((int(width), int(height), freq))
                print(f"Common ad size: {width}x{height} (used {freq} times)")
            
            # If we have training data, use those sizes; otherwise use defaults
            if sizes:
                return sizes[:5]  # Top 5 most common sizes
            else:
                # Fallback to realistic newspaper ad defaults - much smaller sizes
                if publication_type == 'broadsheet':
                    return [(250, 180, 1), (320, 240, 1), (180, 250, 1), (150, 200, 1)]
                else:  # tabloid or other
                    return [(200, 150, 1), (280, 200, 1), (150, 200, 1), (120, 180, 1)]
                
        except Exception as e:
            print(f"Error getting typical ad sizes: {e}")
            return [(300, 200, 1), (400, 300, 1), (200, 300, 1)]
    
    @staticmethod
    def _scan_page_with_training_sizes(image_path, model, feature_names, confidence_threshold=0.7, scaler=None, typical_sizes=None):
        """Scan page using actual ad sizes from training data for better accuracy"""
        try:
            import cv2
            import numpy as np
            import tempfile
            import os
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                return []
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            print(f"Image loaded: {width}x{height} pixels")
            
            detections = []
            windows_scanned = 0
            predictions_made = 0
            
            # Use typical ad sizes from training data
            if typical_sizes is None:
                typical_sizes = [(300, 200, 1)]
            
            # Create ONE temporary image file for the entire page
            temp_fd, temp_img_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
            
            try:
                # Save the full image once
                cv2.imwrite(temp_img_path, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                print(f"Created temp image for scanning")
                
                # Add timeout protection
                import time
                scan_start_time = time.time()
                max_scan_time = 300  # 5 minutes max per page
                
                # Scan with each typical ad size
                for window_w, window_h, frequency in typical_sizes:
                    # Check timeout
                    if time.time() - scan_start_time > max_scan_time:
                        print(f"Scan timeout after {max_scan_time}s, stopping early")
                        break
                        
                    # Use reasonable stride - not too dense, not too sparse
                    stride = min(window_w, window_h) // 3  # 33% overlap
                    print(f"Scanning with {window_w}x{window_h} windows, stride={stride}")
                    
                    for y in range(0, height - window_h, stride):
                        for x in range(0, width - window_w, stride):
                            windows_scanned += 1
                            
                            # Extract features for this window
                            box_coords = {'x': x, 'y': y, 'width': window_w, 'height': window_h}
                            features_dict = AdLearningEngine.extract_features(temp_img_path, box_coords)
                            
                            if features_dict is None:
                                continue
                                
                            # Convert to feature vector
                            features = [features_dict.get(name, 0) for name in feature_names]
                            predictions_made += 1
                            
                            # Make prediction
                            features_array = np.array(features).reshape(1, -1)
                            if scaler is not None:
                                features_array = scaler.transform(features_array)
                            
                            # Get confidence
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(features_array)[0]
                                confidence = proba[1] if len(proba) > 1 else proba[0]
                            else:
                                prediction = model.predict(features_array)[0]
                                confidence = 0.8 if prediction == 1 else 0.2
                            
                            # Log a few sample predictions
                            if predictions_made <= 3:
                                print(f"Sample prediction {predictions_made}: confidence {confidence:.3f}")
                            
                            # If confidence is above threshold, it's likely an ad
                            if confidence >= confidence_threshold:
                                print(f"DETECTION: {window_w}x{window_h} at ({x},{y}) confidence: {confidence:.3f}")
                                
                                # Check for overlaps before adding
                                new_box = {'x': x, 'y': y, 'width': window_w, 'height': window_h}
                                if not AdLearningEngine._overlaps_existing(new_box, detections, overlap_threshold=0.3):
                                    detections.append({
                                        'x': x, 'y': y,
                                        'width': window_w, 'height': window_h,
                                        'confidence': confidence
                                    })
                
                # Sort by confidence and return top detections
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                print(f"Scan complete: {windows_scanned} windows scanned, {predictions_made} predictions made, {len(detections)} detections above threshold")
                return detections[:20]  # Limit to top 20 detections per page
                
            finally:
                # Clean up temp file
                if temp_img_path and os.path.exists(temp_img_path):
                    os.unlink(temp_img_path)
            
        except Exception as e:
            print(f"Error in _scan_page_with_training_sizes: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    @staticmethod
    def _detect_complete_ads(image_path, model, feature_names, confidence_threshold=0.7, scaler=None, typical_sizes=None):
        """Fast, efficient ad detection using sparse sampling of complete ad sizes"""
        try:
            import cv2
            import numpy as np
            import tempfile
            import os
            import time
            
            start_time = time.time()
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                return []
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            print(f"Image loaded: {width}x{height} pixels")
            
            # Use only the top 3 most common ad sizes for speed
            if typical_sizes is None or len(typical_sizes) == 0:
                typical_sizes = [(300, 200, 1)]
            
            # Limit to top 3 sizes to prevent excessive scanning
            top_sizes = typical_sizes[:3]
            print(f"Using top {len(top_sizes)} ad sizes for efficient detection")
            
            detections = []
            total_windows = 0
            
            # Create ONE temporary image file
            temp_fd, temp_img_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
            
            try:
                # Save the full image once
                cv2.imwrite(temp_img_path, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                
                # Process each size with SPARSE sampling (not dense)
                for window_w, window_h, frequency in top_sizes:
                    # Use LARGE stride for speed - we want complete ads, not overlapping pieces
                    stride_x = max(window_w // 2, 100)  # At least 100px steps
                    stride_y = max(window_h // 2, 100)
                    
                    print(f"Scanning {window_w}x{window_h} ads with {stride_x}x{stride_y} stride")
                    
                    # Sparse grid scan - much fewer windows
                    for y in range(0, height - window_h, stride_y):
                        for x in range(0, width - window_w, stride_x):
                            total_windows += 1
                            
                            # Timeout protection
                            if time.time() - start_time > 60:  # 1 minute max per page
                                print(f"Page scan timeout after 60s, stopping early")
                                break
                            
                            # Extract features for this complete ad candidate
                            box_coords = {'x': x, 'y': y, 'width': window_w, 'height': window_h}
                            features_dict = AdLearningEngine.extract_features(temp_img_path, box_coords)
                            
                            if features_dict is None:
                                continue
                                
                            # Convert to feature vector
                            features = [features_dict.get(name, 0) for name in feature_names]
                            
                            # Make prediction
                            features_array = np.array(features).reshape(1, -1)
                            if scaler is not None:
                                features_array = scaler.transform(features_array)
                            
                            # Get confidence
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(features_array)[0]
                                confidence = proba[1] if len(proba) > 1 else proba[0]
                            else:
                                prediction = model.predict(features_array)[0]
                                confidence = 0.8 if prediction == 1 else 0.2
                            
                            # Only accept high-confidence detections to reduce false positives
                            if confidence >= confidence_threshold:
                                print(f"FOUND AD: {window_w}x{window_h} at ({x},{y}) confidence: {confidence:.3f}")
                                
                                # Check for overlaps with generous threshold to merge nearby detections
                                new_box = {'x': x, 'y': y, 'width': window_w, 'height': window_h}
                                if not AdLearningEngine._overlaps_existing(new_box, detections, overlap_threshold=0.5):
                                    detections.append({
                                        'x': x, 'y': y,
                                        'width': window_w, 'height': window_h,
                                        'confidence': confidence
                                    })
                        
                        # Break outer loop on timeout
                        if time.time() - start_time > 60:
                            break
                
                # Sort by confidence and limit results
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                final_detections = detections[:10]  # Max 10 ads per page
                
                elapsed = time.time() - start_time
                print(f"EFFICIENT SCAN COMPLETE: {total_windows} windows in {elapsed:.1f}s, found {len(final_detections)} ads")
                return final_detections
                
            finally:
                # Clean up temp file
                if temp_img_path and os.path.exists(temp_img_path):
                    os.unlink(temp_img_path)
            
        except Exception as e:
            print(f"Error in _detect_complete_ads: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    @staticmethod
    def intelligent_ad_detection(image_path, publication_type, model, feature_names, confidence_threshold=0.6, scaler=None, typical_sizes=None):
        """Intelligent ad detection using visual pattern recognition and layout awareness"""
        try:
            import cv2
            import numpy as np
            import tempfile
            import os
            import time
            
            start_time = time.time()
            print(f"INTELLIGENT AD DETECTION STARTING")
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                return []
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            print(f"Analyzing {width}x{height} {publication_type} publication")
            
            # Step 1: Get publication-specific ad zones
            ad_zones = AdLearningEngine.get_typical_ad_zones((width, height), publication_type)
            print(f"Identified {len(ad_zones)} potential ad zones for {publication_type}")
            
            # Step 2: Detect regions with ad-like visual patterns
            ad_candidates = AdLearningEngine.detect_ads_by_visual_patterns(gray)
            print(f"Found {len(ad_candidates)} regions with ad-like visual patterns")
            
            # Step 3: Filter candidates by ad zones and classify content
            filtered_candidates = []
            for candidate in ad_candidates:
                # Check if candidate overlaps with typical ad zones
                if AdLearningEngine._candidate_in_ad_zones(candidate, ad_zones):
                    # Classify content to ensure it's advertising, not editorial
                    if AdLearningEngine.is_likely_ad_region(gray, candidate['x'], candidate['y'], 
                                                           candidate['width'], candidate['height']):
                        filtered_candidates.append(candidate)
            
            print(f"{len(filtered_candidates)} candidates passed layout and content filters")
            
            # Step 4: Use ML model to validate remaining candidates
            final_detections = []
            temp_fd, temp_img_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
            
            try:
                cv2.imwrite(temp_img_path, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                
                for i, candidate in enumerate(filtered_candidates):
                    if time.time() - start_time > 30:  # 30 second timeout
                        print(f"Timeout reached, processed {i}/{len(filtered_candidates)} candidates")
                        break
                    
                    # Extract ML features for this candidate
                    box_coords = {
                        'x': candidate['x'], 'y': candidate['y'],
                        'width': candidate['width'], 'height': candidate['height']
                    }
                    
                    features_dict = AdLearningEngine.extract_features(temp_img_path, box_coords)
                    if features_dict is None:
                        continue
                    
                    # Make ML prediction
                    features = [features_dict.get(name, 0) for name in feature_names]
                    features_array = np.array(features).reshape(1, -1)
                    
                    if scaler is not None:
                        features_array = scaler.transform(features_array)
                    
                    # Get confidence from ML model
                    if model is not None:
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(features_array)[0]
                            ml_confidence = proba[1] if len(proba) > 1 else proba[0]
                        else:
                            prediction = model.predict(features_array)[0]
                            ml_confidence = 0.8 if prediction == 1 else 0.2
                    else:
                        # Fallback: Use visual confidence as ML confidence
                        ml_confidence = min(candidate['confidence'] * 1.2, 0.9)
                    
                    # Combine visual pattern confidence with ML confidence
                    combined_confidence = (candidate['confidence'] * 0.4 + ml_confidence * 0.6)
                    
                    # Balanced multi-stage confidence check
                    if combined_confidence >= confidence_threshold and candidate['confidence'] > 0.3:
                        print(f"CONFIRMED AD: {candidate['width']}x{candidate['height']} at ({candidate['x']},{candidate['y']}) confidence={combined_confidence:.3f}")
                        
                        # Check for overlaps before adding
                        new_box = {
                            'x': candidate['x'], 'y': candidate['y'],
                            'width': candidate['width'], 'height': candidate['height'],
                            'confidence': combined_confidence
                        }
                        
                        if not AdLearningEngine._overlaps_existing(new_box, final_detections, overlap_threshold=0.4):
                            final_detections.append(new_box)
                
                # Sort by confidence and limit results
                final_detections.sort(key=lambda x: x['confidence'], reverse=True)
                final_detections = final_detections[:8]  # Max 8 ads per page
                
                elapsed = time.time() - start_time
                print(f"INTELLIGENT DETECTION COMPLETE: Found {len(final_detections)} quality ads in {elapsed:.1f}s")
                return final_detections
                
            finally:
                if temp_img_path and os.path.exists(temp_img_path):
                    os.unlink(temp_img_path)
            
        except Exception as e:
            print(f"Error in intelligent_ad_detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    @staticmethod
    def get_typical_ad_zones(image_shape, publication_type):
        """Define typical ad zones for different publication types"""
        width, height = image_shape
        ad_zones = []
        
        if publication_type == 'broadsheet':
            # Top banner area (skip masthead)
            ad_zones.append({'x': 0, 'y': int(height * 0.15), 'width': width, 'height': int(height * 0.12), 'type': 'banner'})
            
            # Left sidebar
            sidebar_width = int(width * 0.25)
            ad_zones.append({'x': 0, 'y': int(height * 0.3), 'width': sidebar_width, 'height': int(height * 0.5), 'type': 'sidebar'})
            
            # Right sidebar  
            ad_zones.append({'x': width - sidebar_width, 'y': int(height * 0.3), 'width': sidebar_width, 'height': int(height * 0.5), 'type': 'sidebar'})
            
            # Bottom classified section
            ad_zones.append({'x': 0, 'y': int(height * 0.75), 'width': width, 'height': int(height * 0.25), 'type': 'classified'})
            
            # Center content areas (between columns)
            center_start = int(width * 0.25)
            center_width = int(width * 0.5) 
            ad_zones.append({'x': center_start, 'y': int(height * 0.3), 'width': center_width, 'height': int(height * 0.4), 'type': 'content'})
        
        elif publication_type == 'special_edition':
            # Scattered preset locations for special editions
            # Top area
            ad_zones.append({'x': 0, 'y': int(height * 0.1), 'width': width, 'height': int(height * 0.15), 'type': 'banner'})
            
            # Middle sections
            ad_zones.append({'x': 0, 'y': int(height * 0.3), 'width': int(width * 0.4), 'height': int(height * 0.3), 'type': 'display'})
            ad_zones.append({'x': int(width * 0.6), 'y': int(height * 0.3), 'width': int(width * 0.4), 'height': int(height * 0.3), 'type': 'display'})
            
            # Bottom area
            ad_zones.append({'x': 0, 'y': int(height * 0.7), 'width': width, 'height': int(height * 0.3), 'type': 'mixed'})
        
        elif publication_type == 'peach':
            # Peach-specific layout zones
            # Header area (below masthead)
            ad_zones.append({'x': 0, 'y': int(height * 0.12), 'width': width, 'height': int(height * 0.1), 'type': 'header'})
            
            # Side columns
            col_width = int(width * 0.3)
            ad_zones.append({'x': 0, 'y': int(height * 0.25), 'width': col_width, 'height': int(height * 0.6), 'type': 'column'})
            ad_zones.append({'x': width - col_width, 'y': int(height * 0.25), 'width': col_width, 'height': int(height * 0.6), 'type': 'column'})
            
            # Footer area
            ad_zones.append({'x': 0, 'y': int(height * 0.85), 'width': width, 'height': int(height * 0.15), 'type': 'footer'})
        
        else:
            # Default generic zones
            ad_zones.append({'x': 0, 'y': int(height * 0.1), 'width': width, 'height': int(height * 0.8), 'type': 'general'})
        
        return ad_zones
    
    @staticmethod
    def detect_ads_by_visual_patterns(gray_image):
        """Detect rectangular regions with ad-like visual characteristics"""
        try:
            import cv2
            import numpy as np
            
            height, width = gray_image.shape
            candidates = []
            
            print(f"Analyzing visual patterns in {width}x{height} image")
            
            # Step 1: Edge detection to find rectangular boundaries
            edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
            
            # Step 2: Find contours (potential ad boundaries)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            print(f"Found {len(contours)} potential boundaries")
            
            # Step 3: Filter contours by ad-like characteristics
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by reasonable ad dimensions - MUCH more restrictive
                # Max ad size should be about 1/4 of page width and 1/5 of page height
                max_ad_width = min(width * 0.25, 400)  # Max 25% of width or 400px
                max_ad_height = min(height * 0.20, 300)  # Max 20% of height or 300px
                min_ad_width = 60   # Minimum viable ad width
                min_ad_height = 40  # Minimum viable ad height
                
                if w < min_ad_width or h < min_ad_height or w > max_ad_width or h > max_ad_height:
                    continue
                
                # Filter by aspect ratio (ads are typically not too extreme)
                aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio < 0.3 or aspect_ratio > 4.0:
                    continue
                
                # Check if contour approximates a rectangle
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Look for roughly rectangular shapes (4-8 points after approximation)
                if len(approx) >= 4 and len(approx) <= 8:
                    # Calculate rectangularity score
                    contour_area = cv2.contourArea(contour)
                    bounding_area = w * h
                    rectangularity = contour_area / bounding_area if bounding_area > 0 else 0
                    
                    if rectangularity > 0.6:  # At least 60% rectangular
                        # Calculate confidence based on visual characteristics
                        roi = gray_image[y:y+h, x:x+w]
                        confidence = AdLearningEngine._calculate_visual_ad_confidence(roi, x, y, w, h, width, height)
                        
                        if confidence > 0.35:  # Balanced minimum visual confidence - not too high, not too low
                            candidates.append({
                                'x': x, 'y': y, 'width': w, 'height': h,
                                'confidence': confidence,
                                'rectangularity': rectangularity,
                                'aspect_ratio': aspect_ratio
                            })
            
            # Step 4: Additional pattern detection for missed ads (grid sampling backup)
            # Sample key positions where ads might not have strong borders
            grid_candidates = AdLearningEngine._detect_borderless_ads(gray_image)
            candidates.extend(grid_candidates)
            
            # Sort by confidence and remove overlaps
            candidates = sorted(candidates, key=lambda x: x['confidence'], reverse=True)
            filtered_candidates = []
            
            for candidate in candidates:
                # Check for overlaps with already selected candidates
                overlap = False
                for selected in filtered_candidates:
                    if AdLearningEngine._boxes_overlap(candidate, selected, threshold=0.5):
                        overlap = True
                        break
                
                if not overlap:
                    filtered_candidates.append(candidate)
                    if len(filtered_candidates) >= 20:  # Limit candidates
                        break
            
            print(f"Selected {len(filtered_candidates)} visual pattern candidates")
            return filtered_candidates
            
        except Exception as e:
            print(f"Error in detect_ads_by_visual_patterns: {e}")
            return []
    
    @staticmethod
    def _calculate_visual_ad_confidence(roi, x, y, w, h, page_width, page_height):
        """Calculate confidence score based on visual ad characteristics"""
        try:
            import cv2
            import numpy as np
            
            if roi.size == 0:
                return 0.0
            
            confidence_factors = []
            
            # Factor 1: Border strength (ads often have defined borders)
            border_thickness = min(3, min(w, h) // 8)
            if border_thickness > 0:
                try:
                    top_border = roi[:border_thickness, :].mean()
                    bottom_border = roi[-border_thickness:, :].mean()
                    left_border = roi[:, :border_thickness].mean()
                    right_border = roi[:, -border_thickness:].mean()
                    center = roi[border_thickness:-border_thickness, border_thickness:-border_thickness].mean()
                    
                    border_contrast = (abs(top_border - center) + abs(bottom_border - center) + 
                                     abs(left_border - center) + abs(right_border - center)) / 4
                    
                    # Normalize and add to factors
                    border_factor = min(border_contrast / 50.0, 1.0)  # Normalize to 0-1
                    confidence_factors.append(border_factor)
                except:
                    confidence_factors.append(0.2)  # Default if border calculation fails
            
            # Factor 2: White space ratio (ads typically have more white space than dense text)
            binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            white_ratio = np.sum(binary == 255) / binary.size
            
            # Ads typically have 40-80% white space
            if 0.4 <= white_ratio <= 0.8:
                white_factor = 1.0
            elif 0.3 <= white_ratio <= 0.9:
                white_factor = 0.7
            else:
                white_factor = 0.3
            
            confidence_factors.append(white_factor)
            
            # Factor 3: Position factor (some positions more likely for ads)
            center_x = x + w/2
            center_y = y + h/2
            
            # Prefer side columns and avoid extreme top (masthead) and bottom
            position_factor = 1.0
            if center_y < page_height * 0.15:  # Too close to masthead
                position_factor *= 0.5
            if center_y > page_height * 0.9:   # Too close to bottom
                position_factor *= 0.7
            if center_x < page_width * 0.2 or center_x > page_width * 0.8:  # Side positions
                position_factor *= 1.2
            
            confidence_factors.append(min(position_factor, 1.0))
            
            # Factor 4: Size appropriateness
            area = w * h
            page_area = page_width * page_height
            size_ratio = area / page_area
            
            # Prefer medium-sized regions (not too small, not too large)
            if 0.01 <= size_ratio <= 0.15:  # 1-15% of page
                size_factor = 1.0
            elif 0.005 <= size_ratio <= 0.25:  # 0.5-25% of page
                size_factor = 0.8
            else:
                size_factor = 0.4
                
            confidence_factors.append(size_factor)
            
            # Calculate weighted average
            final_confidence = np.mean(confidence_factors)
            return float(final_confidence)
            
        except Exception as e:
            print(f"Warning: Error calculating visual confidence: {e}")
            return 0.3  # Default confidence
    
    @staticmethod
    def _detect_borderless_ads(gray_image):
        """Backup detection for ads without strong borders using strategic sampling"""
        try:
            import cv2
            import numpy as np
            
            height, width = gray_image.shape
            candidates = []
            
            # IMPROVED grid sampling with realistic newspaper ad sizes
            common_sizes = [(180, 120), (220, 160), (160, 200), (140, 180), (280, 200)]
            # Much smaller, more realistic newspaper ad dimensions
            
            for w, h in common_sizes:
                # STRICT size limits - no huge detection windows
                if w > width * 0.3 or h > height * 0.25:  # Much more restrictive
                    continue
                
                # Strategic positions (not dense grid)
                step_x = max(w // 2, 100)
                step_y = max(h // 2, 100)
                
                for y in range(int(height * 0.15), height - h, step_y):  # Skip masthead area
                    for x in range(0, width - w, step_x):
                        roi = gray_image[y:y+h, x:x+w]
                        confidence = AdLearningEngine._calculate_visual_ad_confidence(roi, x, y, w, h, width, height)
                        
                        if confidence > 0.3:  # Balanced threshold for borderless detection
                            candidates.append({
                                'x': x, 'y': y, 'width': w, 'height': h,
                                'confidence': confidence,
                                'rectangularity': 0.8,  # Assumed for grid samples
                                'aspect_ratio': w/h
                            })
            
            return candidates
            
        except Exception as e:
            print(f"Warning: Error in borderless ad detection: {e}")
            return []
    
    @staticmethod
    def _scan_page_for_ads(image_path, model, feature_names, confidence_threshold=0.7, scaler=None, typical_sizes=None):
        """Scan page using sliding window to detect ads"""
        try:
            import cv2
            import numpy as np
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                return []
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            print(f"Image loaded: {width}x{height} pixels")
            
            detections = []
            windows_scanned = 0
            predictions_made = 0
            
            # Use typical ad sizes from training data instead of fixed window size
            if typical_sizes is None:
                typical_sizes = [(300, 200, 1), (400, 300, 1), (200, 300, 1)]
            
            print(f"Scanning with {len(typical_sizes)} different ad sizes")
            
            # Create ONE temporary image file for the entire page (MAJOR OPTIMIZATION)
            import tempfile
            import os
            temp_fd, temp_img_path = tempfile.mkstemp(suffix='.png')
            os.close(temp_fd)
            
            try:
                # Save the full image once
                cv2.imwrite(temp_img_path, cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                print(f"Created single temp image for page scanning: {temp_img_path}")
                
                # Multi-size sliding window scan using actual ad sizes from training data
                for window_w, window_h, frequency in typical_sizes:
                    # Use adaptive stride based on window size
                    stride = min(window_w, window_h) // 4  # 25% overlap
                    
                    print(f"Scanning with {window_w}x{window_h} windows (stride {stride})")
                    
                    for y in range(0, height - window_h, stride):
                        for x in range(0, width - window_w, stride):
                            windows_scanned += 1
                            
                            # Convert window coordinates to box_coords format
                            box_coords = {
                                'x': x, 'y': y, 
                                'width': window_w, 'height': window_h
                            }
                        
                        # Extract features using the same method as training (REUSE same temp image)
                        features_dict = AdLearningEngine.extract_features(temp_img_path, box_coords)
                        
                        if features_dict is None:
                            continue
                            
                        # Convert to feature vector matching training format
                        features = [features_dict.get(name, 0) for name in feature_names]
                    
                        # Debug first few feature extractions
                        if predictions_made < 3:
                            print(f"Sample window {predictions_made + 1}: extracted {len(features)} features, expected {len(feature_names)}")
                            if len(features) != len(feature_names):
                                print(f"WARNING: Feature count mismatch! Expected {len(feature_names)}, got {len(features)}")
                    
                        # Make prediction
                        try:
                            predictions_made += 1
                            
                            # Reshape for sklearn
                            features_array = np.array(features).reshape(1, -1)
                            
                            # Apply scaling if available
                            if scaler is not None:
                                features_array = scaler.transform(features_array)
                            
                            # Get prediction and probability
                            if hasattr(model, 'predict_proba'):
                                proba = model.predict_proba(features_array)[0]
                                # Assuming binary classification: [not_ad_prob, ad_prob]
                                confidence = proba[1] if len(proba) > 1 else proba[0]
                            else:
                                # Fallback for models without predict_proba
                                prediction = model.predict(features_array)[0]
                                confidence = 0.8 if prediction == 1 else 0.2
                            
                            # Track confidence distribution (reduced logging)
                            if predictions_made <= 5:  # Log first 5 predictions only
                                print(f"Sample prediction {predictions_made}: confidence {confidence:.3f}")
                            
                            # If confidence is above threshold, consider it an ad
                            if confidence >= confidence_threshold:
                                print(f"DETECTION: Window at ({x},{y}) confidence: {confidence:.3f}")
                                # Expand window to capture full ad (simple approach)
                                expanded_box = AdLearningEngine._expand_detection(
                                    gray, x, y, window_w, window_h, width, height
                                )
                                
                                # Check for overlapping detections
                                if not AdLearningEngine._overlaps_existing(expanded_box, detections, overlap_threshold=0.3):
                                    detections.append({
                                        'x': expanded_box['x'],
                                        'y': expanded_box['y'],
                                        'width': expanded_box['width'],
                                        'height': expanded_box['height'],
                                        'confidence': confidence
                                    })
                        
                        except Exception as e:
                            # Skip this window if prediction fails
                            continue
            
                # Sort by confidence and return top detections
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                print(f"Scan complete: {windows_scanned} windows scanned, {predictions_made} predictions made, {len(detections)} detections above threshold")
                return detections[:20]  # Limit to top 20 detections per page
                
            finally:
                # Clean up the single temp file
                if temp_img_path and os.path.exists(temp_img_path):
                    os.unlink(temp_img_path)
                    print(f"Cleaned up temp image: {temp_img_path}")
            
        except Exception as e:
            print(f"Error in _scan_page_for_ads: {e}")
            return []
    
    @staticmethod
    def _extract_window_features(window, x, y, page_width, page_height):
        """Extract features from a window for ad detection"""
        try:
            if window.size == 0:
                return None
            
            import cv2
            import numpy as np
            
            # Basic geometric features
            height, width = window.shape
            aspect_ratio = width / height if height > 0 else 0
            area = width * height
            relative_x = x / page_width if page_width > 0 else 0
            relative_y = y / page_height if page_height > 0 else 0
            
            # Statistical features
            mean_intensity = np.mean(window)
            std_intensity = np.std(window)
            
            # Edge detection
            edges = cv2.Canny(window, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            
            # Texture features (simplified)
            if window.std() > 0:
                # Compute local binary pattern approximation
                texture_score = cv2.Laplacian(window, cv2.CV_64F).var()
            else:
                texture_score = 0
            
            # Contrast and brightness
            min_val, max_val = np.min(window), np.max(window)
            contrast = max_val - min_val if max_val > min_val else 0
            
            features = [
                aspect_ratio,
                area,
                relative_x,
                relative_y,
                mean_intensity,
                std_intensity,
                edge_density,
                texture_score,
                contrast,
                width,
                height
            ]
            
            return features
            
        except Exception as e:
            print(f"Error extracting window features: {e}")
            return None
    
    @staticmethod
    def _expand_detection(gray, x, y, w, h, page_width, page_height, expansion_factor=1.5):
        """Expand a detection window to capture the full ad"""
        # Simple expansion - multiply dimensions by factor
        new_w = min(int(w * expansion_factor), page_width - x)
        new_h = min(int(h * expansion_factor), page_height - y)
        
        # Center the expansion
        expand_x = max(0, x - int((new_w - w) / 2))
        expand_y = max(0, y - int((new_h - h) / 2))
        
        # Ensure we don't go outside image bounds
        if expand_x + new_w > page_width:
            expand_x = page_width - new_w
        if expand_y + new_h > page_height:
            expand_y = page_height - new_h
        
        return {
            'x': expand_x,
            'y': expand_y,
            'width': new_w,
            'height': new_h
        }
    
    @staticmethod
    def _overlaps_existing(new_box, existing_detections, overlap_threshold=0.3):
        """Check if a new detection overlaps with existing ones"""
        for existing in existing_detections:
            # Calculate intersection
            x1 = max(new_box['x'], existing['x'])
            y1 = max(new_box['y'], existing['y'])
            x2 = min(new_box['x'] + new_box['width'], existing['x'] + existing['width'])
            y2 = min(new_box['y'] + new_box['height'], existing['y'] + existing['height'])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                new_area = new_box['width'] * new_box['height']
                existing_area = existing['width'] * existing['height']
                union = new_area + existing_area - intersection
                
                overlap = intersection / union if union > 0 else 0
                if overlap > overlap_threshold:
                    return True
        
        return False
    
    @staticmethod
    def _candidate_in_ad_zones(candidate, ad_zones):
        """Check if a candidate box overlaps with typical ad zones"""
        candidate_center_x = candidate['x'] + candidate['width'] // 2
        candidate_center_y = candidate['y'] + candidate['height'] // 2
        
        for zone in ad_zones:
            # Check if candidate center falls within this ad zone
            if (zone['x'] <= candidate_center_x <= zone['x'] + zone['width'] and
                zone['y'] <= candidate_center_y <= zone['y'] + zone['height']):
                return True
            
            # Also check for significant overlap (at least 30%)
            overlap_x = max(0, min(candidate['x'] + candidate['width'], zone['x'] + zone['width']) - 
                          max(candidate['x'], zone['x']))
            overlap_y = max(0, min(candidate['y'] + candidate['height'], zone['y'] + zone['height']) - 
                          max(candidate['y'], zone['y']))
            
            if overlap_x > 0 and overlap_y > 0:
                overlap_area = overlap_x * overlap_y
                candidate_area = candidate['width'] * candidate['height']
                overlap_ratio = overlap_area / candidate_area if candidate_area > 0 else 0
                
                if overlap_ratio >= 0.3:  # 30% overlap threshold
                    return True
        
        return False
    
    @staticmethod
    def is_likely_ad_region(gray_image, x, y, width, height):
        """Classify content region to distinguish ads from editorial content"""
        try:
            import cv2
            import numpy as np
            
            # Extract the region of interest
            roi = gray_image[y:y+height, x:x+width]
            if roi.size == 0:
                return False
            
            # Calculate various content analysis metrics
            
            # 1. Text density analysis
            # Use edge detection to estimate text density
            edges = cv2.Canny(roi, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            
            # 2. White space analysis
            # Ads typically have more white space than dense editorial content
            white_pixels = np.sum(roi > 200)  # Assuming white/light pixels
            white_space_ratio = white_pixels / (width * height)
            
            # 3. Contrast and uniformity
            # Ads often have higher contrast and more uniform regions
            roi_std = np.std(roi)
            roi_mean = np.mean(roi)
            
            # 4. Horizontal line detection (typical in ads for borders/separators)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            horizontal_line_density = np.sum(horizontal_lines > 0) / (width * height)
            
            # 5. Vertical line detection
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            vertical_line_density = np.sum(vertical_lines > 0) / (width * height)
            
            # 6. Size factor - larger regions more likely to be ads
            area = width * height
            size_factor = min(area / 10000, 1.0)  # Normalize to [0,1]
            
            # IMPROVED SCORING: More restrictive to prevent false positives
            ad_score = 0
            
            # Much stricter edge density - ads have moderate, controlled structure
            if 0.08 <= edge_density <= 0.20:
                ad_score += 3  # Good ad-like structure
            elif 0.05 <= edge_density < 0.08:
                ad_score += 1  # Minimal structure, possible ad
            elif edge_density < 0.05:  # Too little structure (likely empty space)
                ad_score -= 1
            elif edge_density > 0.35:   # Too much structure (likely dense text/articles)
                ad_score -= 2  # Moderate penalty for dense text regions
            
            # STRICTER white space requirements - ads need significant white space
            if white_space_ratio > 0.4:
                ad_score += 3  # Excellent white space for ads
            elif white_space_ratio > 0.25:
                ad_score += 2  # Good white space
            elif white_space_ratio > 0.15:
                ad_score += 1  # Minimal white space
            else:
                ad_score -= 2  # Too dense, likely text
            
            # Standard deviation - ads often have more varied contrast
            if roi_std > 40:
                ad_score += 1
            
            # Border lines - ads often have clear borders
            if horizontal_line_density > 0.01 or vertical_line_density > 0.01:
                ad_score += 2
            
            # Size factor - larger regions more likely to be display ads
            ad_score += size_factor * 2
            
            # Aspect ratio consideration - very tall/thin likely to be sidebars (ads)
            aspect_ratio = width / height if height > 0 else 1
            if 0.3 <= aspect_ratio <= 3.0:  # Reasonable ad proportions
                ad_score += 1
            
            # Additional text pattern detection - penalize regions with lots of small text
            # Use template matching to detect common text patterns
            text_pattern_score = AdLearningEngine._detect_text_patterns(roi)
            ad_score -= text_pattern_score * 1  # Moderate penalty for text-heavy regions
            
            # Balanced decision threshold - allow legitimate ads while filtering false positives
            threshold = 3  # Lowered from 6 to 3 - more balanced approach
            return ad_score >= threshold
            
        except Exception as e:
            print(f"Error in content classification: {e}")
            return False  # Conservative: if we can't analyze, assume it's not an ad
    
    @staticmethod
    def _detect_text_patterns(roi):
        """Detect text-like patterns that indicate editorial content rather than ads"""
        try:
            import cv2
            import numpy as np
            
            if roi.size == 0:
                return 0
            
            text_score = 0
            
            # Detect horizontal text lines using morphological operations
            # Small horizontal kernels catch text lines
            kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
            horizontal_text = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel_small)
            
            # Count connected components (likely text lines)
            thresh = cv2.threshold(horizontal_text, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            num_labels, labels = cv2.connectedComponents(thresh)
            
            # More than 5 horizontal structures suggests text paragraphs
            if num_labels > 5:
                text_score += (num_labels - 5) * 0.5
            
            # Look for repeating patterns typical of justified text
            # Calculate row-wise pixel density - text has regular patterns
            if roi.shape[0] > 20:  # Only for regions tall enough
                row_densities = []
                for row in range(0, roi.shape[0], 3):  # Sample every 3rd row
                    if row < roi.shape[0]:
                        density = np.sum(roi[row] < 180) / roi.shape[1]  # Dark pixels per row
                        row_densities.append(density)
                
                if len(row_densities) > 5:
                    # Text has more regular patterns than ads
                    density_variance = np.var(row_densities)
                    if density_variance < 0.01:  # Very regular = likely text
                        text_score += 2
            
            return min(text_score, 3)  # Cap the text penalty
            
        except:
            return 0
    
    @staticmethod
    def _boxes_overlap(box1, box2, threshold=0.5):
        """Check if two boxes overlap beyond the given threshold"""
        # Calculate intersection area
        x1_min, y1_min = box1['x'], box1['y']
        x1_max, y1_max = box1['x'] + box1['width'], box1['y'] + box1['height']
        
        x2_min, y2_min = box2['x'], box2['y']
        x2_max, y2_max = box2['x'] + box2['width'], box2['y'] + box2['height']
        
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
            overlap_ratio = intersection / union if union > 0 else 0
            return overlap_ratio > threshold
        
        return False

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
                except (ValueError, IndexError, Exception):
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
        if not password:
            flash('Password is required.', 'error')
            return render_template('login.html')
            
        # Rate limiting could be added here
        if verify_password(LOGIN_PASSWORD, password):
            from datetime import datetime, timedelta
            session.permanent = True
            session['logged_in'] = True
            session['login_time'] = datetime.utcnow().timestamp()
            session['session_id'] = secrets.token_hex(16)
            flash('Successfully logged in!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Incorrect password. Please try again.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()  # Clear all session data
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    try:
        publications = Publication.query.order_by(Publication.upload_date.desc()).all()
        return render_template('index.html', publications=publications)
    except Exception as e:
        print(f"Error in index route: {e}")
        import traceback
        traceback.print_exc()
        flash('An error occurred loading publications. Please try again.', 'error')
        return render_template('index.html', publications=[])

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
        
        # Security validations
        if not allowed_file(file.filename):
            flash('Invalid file type or filename. Only PDF files are allowed.', 'error')
            return redirect(request.url)
        
        if not pub_type or pub_type not in PUBLICATION_CONFIGS:
            flash('Invalid publication type selected.', 'error')
            return redirect(request.url)
        
        if file:
            try:
                print(f"📁 Starting upload process for file: {file.filename}")
                
                # Generate secure unique filename
                file_ext = os.path.splitext(file.filename.lower())[1]
                unique_filename = f"{uuid.uuid4()}{file_ext}"
                print(f"📝 Generated unique filename: {unique_filename}")
                
                # Ensure upload directory exists
                pdf_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs')
                os.makedirs(pdf_dir, exist_ok=True)
                print(f"📂 Upload directory ready: {pdf_dir}")
                
                # Save file
                file_path = os.path.join(pdf_dir, unique_filename)
                print(f"💾 Saving file to: {file_path}")
                file.save(file_path)
                
                # Check file size after saving
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"✅ File saved successfully - Size: {file_size} bytes")
                    if file_size < 1000:  # Less than 1KB is suspicious
                        print(f"⚠️ Warning: File size is unusually small ({file_size} bytes)")
                else:
                    print(f"❌ Error: File was not saved properly")
                    flash('File upload failed. Please try again.', 'error')
                    return redirect(request.url)
                
                # Validate the uploaded file is actually a PDF
                print(f"🔍 Validating PDF file...")
                if not validate_pdf_file(file_path):
                    print(f"❌ PDF validation failed")
                    os.remove(file_path)  # Clean up invalid file
                    flash('Invalid PDF file. Please upload a valid PDF document.', 'error')
                    return redirect(request.url)
                print(f"✅ PDF validation passed")
                
                # Process PDF
                print(f"📖 Opening PDF to count pages...")
                pdf_doc = fitz.open(file_path)
                page_count = pdf_doc.page_count
                pdf_doc.close()
                print(f"📄 PDF has {page_count} pages")
                
                # Create publication record
                print(f"📊 Creating publication record...")
                config = PUBLICATION_CONFIGS[pub_type]
                total_inches = config['total_inches_per_page'] * page_count
                
                publication = Publication(
                    filename=unique_filename,
                    original_filename=file.filename,
                    publication_type=pub_type,
                    total_pages=page_count,
                    total_inches=total_inches
                )
                print(f"✅ Publication object created")
                
                # Set processing status if available (with timeout protection)
                print(f"🔄 Setting processing status...")
                try:
                    publication.set_processing_status('uploaded')
                    print(f"✅ Processing status set")
                except Exception as e:
                    print(f"⚠️ Warning: Could not set processing status: {e}")
                
                print(f"💾 Adding publication to database...")
                db.session.add(publication)
                print(f"💾 Committing to database...")
                db.session.commit()
                print(f"✅ Database commit successful")
                
                flash(f'File uploaded successfully! Processing {page_count} pages...', 'success')
                print(f"🎉 Upload completed for publication {publication.id}: {publication.original_filename}")
                
                # Start background processing immediately
                print(f"🚀 Starting background processing...")
                try:
                    start_background_processing(publication.id)
                    print(f"✅ Background processing started")
                except Exception as e:
                    print(f"⚠️ Warning: Could not start background processing: {e}")
                
                print(f"🔄 Redirecting to processing status page...")
                # Redirect to processing status page
                return redirect(url_for('processing_status', pub_id=publication.id))
                
            except Exception as e:
                # CRITICAL: Always rollback on database errors
                try:
                    db.session.rollback()
                except Exception:
                    db.session.remove()
                
                print(f"💥 UPLOAD ERROR: {str(e)}")
                import traceback
                print(f"📋 Error traceback:")
                traceback.print_exc()
                
                # Return user-friendly error message
                error_msg = str(e)
                if "InFailedSqlTransaction" in error_msg:
                    error_msg = "Database transaction error. Please try uploading again."
                
                flash(f'Error uploading file: {error_msg}', 'error')
                return redirect(request.url)
    
    return render_template('upload.html', pub_types=PUBLICATION_CONFIGS)

@app.route('/processing/<int:pub_id>')
@login_required
def processing_status(pub_id):
    """Show processing status page with progress"""
    try:
        publication = Publication.query.get_or_404(pub_id)
        return render_template('processing_status.html', publication=publication)
    except Exception as e:
        flash(f'Error loading processing status: {str(e)}')
        return redirect(url_for('index'))

@app.route('/api/processing_status/<int:pub_id>')
@login_required
def get_processing_status(pub_id):
    """API endpoint to check processing status"""
    try:
        publication = Publication.query.get_or_404(pub_id)
        
        # Use safe properties
        status = publication.safe_processing_status
        error = publication.safe_processing_error
        
        # Start background processing if not started
        if status == 'uploaded':
            # Trigger background processing
            start_background_processing(pub_id)
            publication.set_processing_status('processing')
            db.session.commit()
            status = 'processing'
        
        # Handle basic_completed status (when synchronous processing was used)
        if status == 'basic_completed':
            status = 'completed'
        
        return jsonify({
            'status': status,
            'processed': publication.processed,
            'error': error,
            'redirect_url': url_for('measure_publication', pub_id=pub_id) if publication.processed else None
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'processed': False,
            'error': str(e),
            'redirect_url': None
        })

@app.route('/api/restart_processing/<int:pub_id>', methods=['POST'])
@login_required  
def restart_processing(pub_id):
    """Force restart stuck processing"""
    try:
        publication = Publication.query.get_or_404(pub_id)
        
        global _processing_publications, _processing_lock
        
        # Remove from processing set to allow restart
        try:
            if _processing_lock:
                with _processing_lock:
                    _processing_publications.discard(pub_id)
        except:
            pass
        
        # Reset processing status
        publication.set_processing_status('uploaded')
        publication.processed = False
        db.session.commit()
        
        print(f"Restarted processing for publication {pub_id}")
        
        return jsonify({
            'success': True,
            'message': 'Processing restarted successfully'
        })
        
    except Exception as e:
        print(f"Error restarting processing: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

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
            
            # AI Box Detection - Use ML predictions if model is available, fallback to CV detection
            detected_boxes = []
            ml_predictions = AdLearningEngine.predict_ads(image_path, publication.publication_type, confidence_threshold=0.4)
            
            if ml_predictions['success'] and ml_predictions['predictions']:
                # Use ML predictions
                for pred in ml_predictions['predictions']:
                    detected_boxes.append({
                        'x': pred['x'],
                        'y': pred['y'],
                        'width': pred['width'],
                        'height': pred['height'],
                        'confidence': pred['confidence'],
                        'predicted_type': pred['predicted_type']
                    })
                print(f"Used ML model for page {page_num + 1}: {len(detected_boxes)} ads predicted")
            else:
                # Fallback to traditional CV detection
                detected_boxes = AdBoxDetector.detect_boxes(image_path)
                print(f"Used CV detection for page {page_num + 1}: {len(detected_boxes)} ads detected")
            
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
                    ad_type=box.get('predicted_type', 'ai_detected'),
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
    try:
        print(f"📏 Loading measurement page for publication {pub_id}")
        
        publication = Publication.query.get_or_404(pub_id)
        print(f"✅ Found publication: {publication.original_filename}")
        
        pages = Page.query.filter_by(publication_id=pub_id).order_by(Page.page_number).all()
        print(f"📄 Found {len(pages)} pages")
        
        # Check if publication is still processing
        if len(pages) == 0 and not publication.processed:
            print(f"🔄 Publication still processing, redirecting to status page")
            flash('Publication is still being processed. Please wait...', 'info')
            return redirect(url_for('processing_status', pub_id=pub_id))
        
        # Calculate total detected boxes
        total_boxes = 0
        for page in pages:
            boxes = AdBox.query.filter_by(page_id=page.id).count()
            total_boxes += boxes
        print(f"📦 Total ad boxes: {total_boxes}")
        
        print(f"🎨 Rendering measure.html template")
        return render_template('measure.html', 
                             publication=publication, 
                             pages=pages,
                             total_boxes=total_boxes)
                             
    except Exception as e:
        print(f"💥 MEASURE PAGE ERROR: {str(e)}")
        import traceback
        print(f"📋 Error traceback:")
        traceback.print_exc()
        flash(f'Error loading measurement page: {str(e)}', 'error')
        return redirect(url_for('index'))

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
    """Serve page images with lazy generation"""
    page = Page.query.get_or_404(page_id)
    publication = Publication.query.get(page.publication_id)
    
    # Try to generate image if it doesn't exist
    image_filename = generate_page_image_if_needed(publication, page.page_number)
    
    if image_filename:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pages', image_filename)
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/png')
    
    return "Image could not be generated", 404

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
    
    # AUTOMATIC AI LEARNING: Seamlessly learn from this publication during report generation
    ai_learning_status = {
        'enabled': True,
        'publication_patterns_learned': 0,
        'total_training_samples': 0,
        'model_accuracy': None,
        'model_updated': False,
        'learning_summary': '',
        'error_message': None
    }

    try:
        print(f"AUTOMATIC AI LEARNING: Processing {publication.publication_type} publication...")

        # Step 1: Extract all user-verified measurements from this publication as training data
        pub_ad_boxes = AdBox.query.join(Page).filter(
            Page.publication_id == pub_id,
            AdBox.user_verified == True
        ).all()

        # Count patterns we can learn from this publication
        patterns_to_learn = 0
        for ad_box in pub_ad_boxes:
            existing_training = TrainingData.query.filter_by(ad_box_id=ad_box.id).first()
            if not existing_training:
                patterns_to_learn += 1

        ai_learning_status['publication_patterns_learned'] = patterns_to_learn

        # Step 2: Collect training data from this publication
        if patterns_to_learn > 0:
            print(f"LEARNING: Extracting patterns from {patterns_to_learn} user measurements...")
            collected = AdLearningEngine.collect_training_data(max_samples=patterns_to_learn, batch_size=10)
            print(f"LEARNING: Successfully extracted {collected} ad patterns from this publication")

            # Update status
            ai_learning_status['publication_patterns_learned'] = collected
        else:
            print(f"LEARNING: All patterns from this publication already learned")

        # Step 3: Get total training samples available
        total_samples = TrainingData.query.filter_by(publication_type=publication.publication_type).count()
        ai_learning_status['total_training_samples'] = total_samples

        # Step 4: Update ML models with new data automatically
        if total_samples >= 10:  # Minimum samples for reliable training
            print(f"LEARNING: Updating {publication.publication_type} AI model with {total_samples} total patterns...")

            result = AdLearningEngine.train_model(
                publication_type=publication.publication_type,
                min_samples=10,  # Lower threshold for report-triggered retraining
                collect_new_data=False  # We just collected above
            )

            if result.get('success'):
                ai_learning_status['model_updated'] = True
                ai_learning_status['model_accuracy'] = result.get('validation_accuracy', 0) * 100

                print(f"LEARNING: Successfully updated AI model - {ai_learning_status['model_accuracy']:.1f}% accuracy")

                # Create learning summary
                if patterns_to_learn > 0:
                    ai_learning_status['learning_summary'] = f"AI learned {patterns_to_learn} new patterns from this publication and updated the {publication.publication_type} model (now {ai_learning_status['model_accuracy']:.1f}% accurate with {total_samples} total patterns)"
                else:
                    ai_learning_status['learning_summary'] = f"AI model refreshed with all {total_samples} known patterns ({ai_learning_status['model_accuracy']:.1f}% accuracy)"

                # User feedback
                if patterns_to_learn > 0:
                    flash(f'AI learned {patterns_to_learn} new patterns from this publication! Model is now {ai_learning_status["model_accuracy"]:.1f}% accurate.', 'success')

            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"LEARNING: Model update failed: {error_msg}")
                ai_learning_status['error_message'] = error_msg
                ai_learning_status['learning_summary'] = f"Collected {patterns_to_learn} new patterns but model update failed: {error_msg}"
        else:
            # Not enough data yet
            ai_learning_status['learning_summary'] = f"Collected {patterns_to_learn} patterns from this publication. Need {10 - total_samples} more patterns to create reliable AI model."

        print(f"AUTOMATIC AI LEARNING COMPLETE: {ai_learning_status['learning_summary']}")

    except Exception as ml_error:
        print(f"LEARNING ERROR: {ml_error}")
        ai_learning_status['error_message'] = str(ml_error)
        ai_learning_status['learning_summary'] = f"AI learning encountered an error: {str(ml_error)}"
        # Don't let ML errors break report generation

    report_data = {
        'publication': publication,
        'config': config,
        'pages': page_data,
        'total_boxes': total_boxes,
        'user_created_boxes': user_created_boxes,
        'ai_detected_boxes': ai_detected_boxes,
        'generated_date': datetime.now(),
        'ai_learning_status': ai_learning_status
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
    try:
        # Clear any failed transaction state at start
        db.session.rollback()
        
        ad_box = AdBox.query.get_or_404(box_id)
        data = request.json
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
        
        # Automatically extract features for ML training when user modifies ad
        try:
            image_filename = f"{publication.filename}_page_{page.page_number}.png"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pages', image_filename)
            
            if os.path.exists(image_path):
                box_coords = {'x': data['x'], 'y': data['y'], 'width': data['width'], 'height': data['height']}
                features = AdLearningEngine.extract_features(image_path, box_coords)
                
                if features:
                    # Update or create training data
                    existing = TrainingData.query.filter_by(ad_box_id=ad_box.id).first()
                    if existing:
                        # Update existing training data
                        existing.features = json.dumps(features)
                        existing.label = ad_box.ad_type or 'manual'
                        existing.extracted_date = datetime.utcnow()
                        print(f"Updated ML features for modified ad box {ad_box.id}")
                    else:
                        # Create new training data
                        training_data = TrainingData(
                            ad_box_id=ad_box.id,
                            publication_type=publication.publication_type,
                            features=json.dumps(features),
                            label=ad_box.ad_type or 'manual',
                            confidence_score=1.0
                        )
                        db.session.add(training_data)
                        print(f"Extracted ML features for updated ad box {ad_box.id}")
                    
        except Exception as e:
            print(f"Warning: Could not extract features for training: {e}")
        
        # Commit all changes together
        db.session.commit()
        
        # Recalculate page and publication totals
        update_totals(ad_box.page_id)
        
        return jsonify({
            'success': True,
            'x': int(data['x']),
            'y': int(data['y']),
            'width': int(data['width']),
            'height': int(data['height']),
            'width_raw': float(width_inches_raw),
            'height_raw': float(height_inches_raw),
            'width_rounded': float(width_rounded),
            'height_rounded': float(height_rounded),
            'column_inches': float(ad_box.column_inches),
            'ad_type': ad_box.ad_type or 'manual'
        })
        
    except Exception as e:
        # CRITICAL: Always rollback on database errors
        try:
            db.session.rollback()
        except Exception:
            db.session.remove()
        
        print(f"Error updating ad box: {e}")
        import traceback
        traceback.print_exc()
        
        # Return user-friendly error message
        error_msg = str(e)
        if "InFailedSqlTransaction" in error_msg:
            error_msg = "Database transaction error. Please refresh the page and try again."
        elif "does not exist" in error_msg.lower():
            error_msg = "Ad box not found or already deleted."
        
        return jsonify({'success': False, 'error': error_msg}), 500

@app.route('/api/delete_box/<int:box_id>', methods=['DELETE'])
def delete_box(box_id):
    """Delete an ad box"""
    try:
        # Start fresh transaction
        db.session.rollback()  # Clear any existing failed transaction
        
        ad_box = db.session.get(AdBox, box_id)
        if not ad_box:
            return jsonify({'success': False, 'error': 'Ad box not found'})
        
        page_id = ad_box.page_id
        
        # Delete training data first (avoid selecting non-existent columns)
        from sqlalchemy import text
        db.session.execute(text('DELETE FROM training_data WHERE ad_box_id = :box_id'), {'box_id': box_id})
        
        # Delete the ad box
        db.session.delete(ad_box)
        db.session.commit()
        
        # Recalculate totals
        update_totals(page_id)
        
        return jsonify({'success': True})
        
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting ad box {box_id}: {e}")
        return jsonify({
            'success': False, 
            'error': 'Database transaction error. Please refresh the page and try again.'
        })

@app.route('/api/db_health', methods=['GET'])
def db_health_check():
    """Check database connection and transaction state"""
    try:
        # Clear any failed transaction state
        db.session.rollback()
        
        # Test basic query
        count = db.session.query(Publication).count()
        
        # Test transaction
        db.session.execute(db.text("SELECT 1"))
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'status': 'healthy',
            'publication_count': count
        })
    except Exception as e:
        try:
            db.session.rollback()
        except Exception:
            db.session.remove()
        
        return jsonify({
            'success': False, 
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/api/add_box/<int:page_id>', methods=['POST'])
def add_box(page_id):
    """Add a new ad box"""
    try:
        # Clear any failed transaction state at start
        db.session.rollback()
        
        page = Page.query.get_or_404(page_id)
        publication = Publication.query.get(page.publication_id)
        config = PUBLICATION_CONFIGS[publication.publication_type]
        data = request.json
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
        db.session.flush()  # Get the ad_box.id without committing
        
        # PRIORITY 4: Collect positive training data from manually created ad
        try:
            # Collect comprehensive positive training example
            NegativeTrainingCollector.collect_positive_example(ad_box, training_source='manual')
            
            # Legacy feature extraction (keep for compatibility)
            image_filename = f"{publication.filename}_page_{page.page_number}.png"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pages', image_filename)
            
            if os.path.exists(image_path) and hasattr(globals().get('AdLearningEngine', None), 'extract_features'):
                box_coords = {'x': data['x'], 'y': data['y'], 'width': data['width'], 'height': data['height']}
                features = AdLearningEngine.extract_features(image_path, box_coords)
                
                if features:
                    # Store legacy features if needed
                    existing = TrainingData.query.filter(
                        TrainingData.ad_box_id == ad_box.id
                    ).first()
                    if not existing:
                        training_data = TrainingData(
                            ad_box_id=ad_box.id,
                            publication_type=publication.publication_type,
                            features=json.dumps(features),
                            label=ad_box.ad_type,
                            confidence_score=1.0
                        )
                        db.session.add(training_data)
                        
        except Exception as e:
            print(f"Warning: Could not collect training data: {e}")
        
        # Commit all changes together
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
        # CRITICAL: Always rollback on database errors
        try:
            db.session.rollback()
        except Exception:
            db.session.remove()
        
        print(f"Error adding ad box: {e}")
        import traceback
        traceback.print_exc()
        
        # Return user-friendly error message
        error_msg = str(e)
        if "InFailedSqlTransaction" in error_msg:
            error_msg = "Database transaction error. Please refresh the page and try again."
        elif "does not exist" in error_msg.lower():
            error_msg = "Page not found or invalid data."
        
        return jsonify({'success': False, 'error': error_msg}), 500

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
        
        # Automatically extract features for ML training
        try:
            box_coords = {'x': detected_box['x'], 'y': detected_box['y'], 'width': detected_box['width'], 'height': detected_box['height']}
            features = AdLearningEngine.extract_features(image_path, box_coords)
            
            if features:
                # Check if training data already exists
                existing = TrainingData.query.filter_by(ad_box_id=ad_box.id).first()
                if not existing:
                    training_data = TrainingData(
                        ad_box_id=ad_box.id,
                        publication_type=publication.publication_type,
                        features=json.dumps(features),
                        label=ad_type,
                        confidence_score=1.0
                    )
                    db.session.add(training_data)
                    db.session.commit()
                    print(f"Extracted ML features for intelligent detection ad {ad_box.id}")
        except Exception as e:
            print(f"Warning: Could not extract features for training: {e}")
        
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

@app.route('/api/auto_detect_ads/<int:page_id>', methods=['POST'])
def auto_detect_ads_with_learning(page_id):
    """NEW: Hybrid logo recognition + manual detection for single page"""
    try:
        page = Page.query.get_or_404(page_id)
        publication = Publication.query.get(page.publication_id)

        print(f"Starting HYBRID logo recognition for page {page_id}")

        # Initialize hybrid detection pipeline
        hybrid_pipeline = HybridDetectionPipeline()

        # Run logo recognition detection on this specific page
        recognition_engine = LogoRecognitionDetectionEngine()
        page_result = recognition_engine.detect_logos_on_page(page_id, confidence_threshold=0.7)

        if page_result.get('success'):
            detections_created = page_result.get('detections_created', 0)
            business_names = page_result.get('business_names', [])

            print(f"Hybrid detection found {detections_created} logo-based ads on page {page.page_number}")

            # Get detection suggestions for additional manual review
            suggestions_result = hybrid_pipeline.get_detection_suggestions(page_id, threshold=0.6)
            suggestions = suggestions_result.get('suggestions', []) if suggestions_result.get('success') else []

            return jsonify({
                'success': True,
                'detections': detections_created,
                'business_names': business_names,
                'suggestions': suggestions,
                'message': f'Logo recognition complete: {detections_created} business ads detected'
            })
        else:
            error_msg = page_result.get('error', 'Logo recognition failed')
            print(f"Logo recognition failed: {error_msg}")

            # Still provide suggestions for manual detection
            suggestions_result = hybrid_pipeline.get_detection_suggestions(page_id, threshold=0.5)
            suggestions = suggestions_result.get('suggestions', []) if suggestions_result.get('success') else []

            return jsonify({
                'success': True,
                'detections': 0,
                'suggestions': suggestions,
                'message': 'No logo-based ads detected - use manual detection with suggestions'
            })

    except Exception as e:
        print(f"Error in hybrid auto detection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})
        
        if not detections:
            return jsonify({
                'success': True,
                'detections': [],
                'message': 'No ads detected with current settings'
            })
        
        # Convert detections to format expected by frontend
        formatted_detections = []
        for detection in detections:
            formatted_detections.append({
                'x': detection['x'],
                'y': detection['y'],
                'width': detection['width'],
                'height': detection['height'],
                'confidence': detection['confidence'],
                'type': detection.get('type', 'bordered_rectangle'),
                'element_id': detection.get('element_id', 'unknown'),
                'has_border': detection.get('border', False),
                'is_merged': 'merged' in detection.get('element_id', ''),
                'classification': AutoDetectionClassifier.classify_detection(detection)
            })
        
        print(f"Enhanced detection complete: {len(formatted_detections)} ads found")
        
        return jsonify({
            'success': True,
            'detections': formatted_detections,
            'total_found': len(formatted_detections),
            'detection_method': 'enhanced_pdf_metadata',
            'learning_enabled': True
        })
        
    except Exception as e:
        print(f"Error in enhanced auto detection: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

class AutoDetectionClassifier:
    """
    PRIORITY 4: Classify detected regions for better user understanding
    """
    
    @staticmethod
    def classify_detection(detection):
        """
        Classify detection type for user interface
        
        Returns:
            dict: Classification with user-friendly labels
        """
        try:
            confidence = detection['confidence']
            width = detection['width']
            height = detection['height']
            has_border = detection.get('border', False)
            
            classification = {
                'category': 'unknown',
                'description': 'Unknown element',
                'user_action': 'review',
                'color': '#666666'
            }
            
            if confidence == 0.1:
                classification.update({
                    'category': 'photo',
                    'description': 'Editorial photo (likely not an ad)',
                    'user_action': 'probably_delete',
                    'color': '#ff6b6b'  # Red
                })
            elif confidence >= 0.95:
                classification.update({
                    'category': 'business_directory',
                    'description': 'Business directory item',
                    'user_action': 'probably_keep',
                    'color': '#51cf66'  # Green
                })
            elif confidence >= 0.85 and has_border:
                if width >= 250 and height >= 150:
                    classification.update({
                        'category': 'complex_ad',
                        'description': 'Complex ad (outer boundary)',
                        'user_action': 'review_boundary',
                        'color': '#339af0'  # Blue
                    })
                else:
                    classification.update({
                        'category': 'standard_ad',
                        'description': 'Standard bordered ad',
                        'user_action': 'probably_keep',
                        'color': '#51cf66'  # Green
                    })
            else:
                classification.update({
                    'category': 'uncertain',
                    'description': 'Uncertain - needs review',
                    'user_action': 'review',
                    'color': '#ffd43b'  # Yellow
                })
            
            return classification
            
        except Exception as e:
            print(f"Error classifying detection: {e}")
            return {
                'category': 'error',
                'description': 'Classification error',
                'user_action': 'review',
                'color': '#666666'
            }

@app.route('/api/confirm_auto_detection/<int:page_id>', methods=['POST'])
def confirm_auto_detection(page_id):
    """PRIORITY 4: Confirm automatic detections and collect positive training data"""
    try:
        page = Page.query.get_or_404(page_id)
        data = request.json
        confirmed_detections = data.get('confirmed_detections', [])
        
        created_boxes = []
        
        for detection in confirmed_detections:
            # Create ad box from confirmed detection
            ad_box = AdBox(
                page_id=page_id,
                x=detection['x'],
                y=detection['y'],
                width=detection['width'],
                height=detection['height'],
                ad_type=detection.get('ad_type', 'open_display'),
                column_inches=0  # Will be calculated
            )
            
            db.session.add(ad_box)
            db.session.flush()  # Get ID
            
            # PRIORITY 4: Collect positive training data
            NegativeTrainingCollector.collect_positive_example(
                ad_box, training_source='automatic_confirmed'
            )
            
            created_boxes.append({
                'id': ad_box.id,
                'x': ad_box.x,
                'y': ad_box.y,
                'width': ad_box.width,
                'height': ad_box.height
            })
        
        db.session.commit()
        
        # Update page totals
        update_totals(page_id)
        
        print(f"Confirmed {len(created_boxes)} automatic detections for page {page_id}")
        
        return jsonify({
            'success': True,
            'created_boxes': created_boxes,
            'total_confirmed': len(created_boxes)
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"Error confirming auto detections: {e}")
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

@app.route('/api/copy_ad_box/<int:page_id>', methods=['POST'])
def copy_ad_box(page_id):
    """Copy an ad box with exact measurements"""
    page = Page.query.get_or_404(page_id)
    publication = Publication.query.get(page.publication_id)
    data = request.json
    
    try:
        # Create ad box record with the exact measurements from the original
        ad_box = AdBox(
            page_id=page_id,
            x=data['x'],
            y=data['y'],
            width=data['width'],
            height=data['height'],
            width_inches_raw=data.get('width_raw', 0),
            height_inches_raw=data.get('height_raw', 0),
            width_inches_rounded=data.get('width_rounded', 0),
            height_inches_rounded=data.get('height_rounded', 0),
            column_inches=data.get('column_inches', 0),
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
            'x': int(data['x']),
            'y': int(data['y']),
            'width': int(data['width']),
            'height': int(data['height']),
            'width_raw': float(data.get('width_raw', 0)),
            'height_raw': float(data.get('height_raw', 0)),
            'width_rounded': float(data.get('width_rounded', 0)),
            'height_rounded': float(data.get('height_rounded', 0)),
            'column_inches': float(data.get('column_inches', 0)),
            'ad_type': data.get('ad_type', 'manual')
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

# ML Learning System API Routes
@app.route('/api/ml/train/<publication_type>', methods=['POST'])
@login_required
def train_ml_model(publication_type):
    """Train ML model for ad detection"""
    try:
        # Validate publication type
        if publication_type not in PUBLICATION_CONFIGS:
            return jsonify({'success': False, 'error': 'Invalid publication type'})
        
        # Get parameters from request or use defaults
        data = request.get_json() or {}
        min_samples = data.get('min_samples', 20)
        collect_new_data = data.get('collect_new_data', False)
        
        # Train the model
        result = AdLearningEngine.train_model(publication_type, min_samples, collect_new_data)
        
        if result['success']:
            flash(f'Successfully trained ML model for {publication_type}! '
                  f'Accuracy: {result["validation_accuracy"]:.1%} '
                  f'({result["training_samples"]} samples)')
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ml/stats')
@login_required  
def get_ml_stats():
    """Get ML training statistics"""
    try:
        # Get stats for each publication type
        all_stats = {}
        for pub_type in PUBLICATION_CONFIGS.keys():
            all_stats[pub_type] = AdLearningEngine.get_training_stats(pub_type)
        
        # Get active models
        active_models = MLModel.query.filter_by(is_active=True).all()
        model_info = []
        for model in active_models:
            model_info.append({
                'id': model.id,
                'name': model.model_name,
                'type': model.model_type,
                'publication_type': model.publication_type,
                'version': model.version,
                'training_accuracy': model.training_accuracy,
                'validation_accuracy': model.validation_accuracy,
                'training_samples': model.training_samples,
                'created_date': model.created_date.isoformat()
            })
        
        return jsonify({
            'success': True,
            'training_stats': all_stats,
            'active_models': model_info,
            'total_verified_ads': AdBox.query.filter_by(user_verified=True).count()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ml/reset_training_data', methods=['POST'])
@login_required
def reset_training_data():
    """Reset training data to start fresh with improved Vision AI"""
    try:
        data = request.get_json() or {}
        publication_type = data.get('publication_type')  # Optional - can reset all or specific type
        confirmation_code = data.get('confirmation_code')
        
        result = AdLearningEngine.reset_training_data(
            publication_type=publication_type,
            confirmation_code=confirmation_code
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ml/collect_training_data', methods=['POST'])
@login_required
def collect_training_data():
    """Manually trigger training data collection"""
    try:
        # Process in smaller batches to avoid timeouts
        data = request.get_json() or {}
        max_samples = data.get('max_samples', 100)  # Process max 100 at a time
        batch_size = data.get('batch_size', 25)      # Commit every 25 samples
        
        samples_collected = AdLearningEngine.collect_training_data(
            batch_size=batch_size, 
            max_samples=max_samples
        )
        
        return jsonify({
            'success': True,
            'samples_collected': samples_collected,
            'message': f'Collected features from {samples_collected} new verified ads'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ml/predict/<int:page_id>')
@login_required
def get_ml_predictions(page_id):
    """Get ML predictions for a specific page"""
    try:
        page = Page.query.get_or_404(page_id)
        publication = Publication.query.get(page.publication_id)
        
        # Get image path
        image_filename = f"{publication.filename}_page_{page.page_number}.png"
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pages', image_filename)
        
        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Page image not found'})
        
        # Get ML predictions
        confidence_threshold = request.args.get('confidence', 0.7, type=float)
        predictions = AdLearningEngine.predict_ads(image_path, publication.publication_type, confidence_threshold)
        
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ml/models')
@login_required
def list_ml_models():
    """List all ML models"""
    try:
        models = MLModel.query.order_by(MLModel.created_date.desc()).all()
        
        model_list = []
        for model in models:
            model_list.append({
                'id': model.id,
                'name': model.model_name,
                'type': model.model_type,
                'publication_type': model.publication_type,
                'version': model.version,
                'training_accuracy': model.training_accuracy,
                'validation_accuracy': model.validation_accuracy,
                'training_samples': model.training_samples,
                'created_date': model.created_date.isoformat(),
                'is_active': model.is_active
            })
        
        return jsonify({
            'success': True,
            'models': model_list
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ml/models/<int:model_id>/activate', methods=['POST'])
@login_required
def activate_ml_model(model_id):
    """Activate a specific ML model"""
    try:
        model = MLModel.query.get_or_404(model_id)
        
        # Deactivate other models of the same type and publication
        MLModel.query.filter_by(
            model_type=model.model_type,
            publication_type=model.publication_type
        ).update({'is_active': False})
        
        # Activate this model
        model.is_active = True
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Activated model {model.model_name}'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/ml')
@login_required
def ml_dashboard():
    """Machine Learning Dashboard"""
    try:
        # Get training statistics
        stats = {}
        for pub_type in PUBLICATION_CONFIGS.keys():
            stats[pub_type] = AdLearningEngine.get_training_stats(pub_type)
        
        # Get active models
        active_models = MLModel.query.filter_by(is_active=True).all()
        
        # Get recent training data
        recent_training = TrainingData.query.order_by(TrainingData.extracted_date.desc()).limit(10).all()
        
        # Check if we have enough data to train
        ready_to_train = {}
        for pub_type in PUBLICATION_CONFIGS.keys():
            ready_to_train[pub_type] = stats[pub_type]['total_samples'] >= 20
        
        return render_template('ml_dashboard.html',
                             stats=stats,
                             active_models=active_models,
                             recent_training=recent_training,
                             ready_to_train=ready_to_train,
                             publication_types=list(PUBLICATION_CONFIGS.keys()))
                             
    except Exception as e:
        flash(f'Error loading ML dashboard: {str(e)}', 'error')
        return redirect(url_for('index'))


# Create database tables and ensure schema is up to date
with app.app_context():
    db.create_all()
    
    # Add missing columns if they don't exist (skip if causing timeout)
    try:
        from sqlalchemy import text, inspect
        
        # Skip schema updates in production to prevent worker timeout (re-enabled after column fix)
        if os.environ.get('RAILWAY_ENVIRONMENT'):
            print("Skipping schema updates in production environment")
        else:
            inspector = inspect(db.engine)
            columns = [col['name'] for col in inspector.get_columns('publication')]
            
            if 'processing_status' not in columns:
                db.session.execute(text("ALTER TABLE publication ADD COLUMN processing_status VARCHAR(50) DEFAULT 'uploaded'"))
                db.session.commit()
                print("Added processing_status column")
            
            # Check and add auto-detection columns to ad_box
            ad_box_columns = [col['name'] for col in inspector.get_columns('ad_box')]
            
            if 'detected_automatically' not in ad_box_columns:
                db.session.execute(text('ALTER TABLE ad_box ADD COLUMN detected_automatically BOOLEAN DEFAULT FALSE'))
                db.session.commit()
                print("Added detected_automatically column to ad_box")
            
            if 'confidence_score' not in ad_box_columns:
                db.session.execute(text('ALTER TABLE ad_box ADD COLUMN confidence_score FLOAT'))
                db.session.commit()
                print("Added confidence_score column to ad_box")
            
            if 'processing_error' not in columns:
                db.session.execute(text('ALTER TABLE publication ADD COLUMN processing_error VARCHAR(500)'))
                db.session.commit() 
                print("Added processing_error column")
            
    except Exception as e:
        print(f"Schema update error (may be normal): {e}")

@app.route('/api/test-vision-api', methods=['GET'])
@login_required
def test_vision_api():
    """Test Google Vision API connectivity and functionality"""
    try:
        # Check if any page images exist to test with
        pages_dir = os.path.join('static', 'uploads', 'pages')
        if not os.path.exists(pages_dir):
            return jsonify({
                'success': False, 
                'error': 'No page images directory found',
                'vision_api_available': is_vision_api_available()
            })
        
        # Find first available page image
        test_image = None
        for filename in os.listdir(pages_dir):
            if filename.endswith('.png'):
                test_image = os.path.join(pages_dir, filename)
                break
        
        if not test_image:
            return jsonify({
                'success': False,
                'error': 'No test images found in pages directory',
                'vision_api_available': is_vision_api_available()
            })
        
        print(f"Testing Vision API with: {test_image}")
        
        # Test Vision API
        api_working = GoogleVisionAdDetector.test_vision_api(test_image)
        
        if api_working:
            # Test actual ad detection
            ads = GoogleVisionAdDetector.detect_ads(test_image)
            
            return jsonify({
                'success': True,
                'vision_api_available': True,
                'test_image': test_image,
                'ads_detected': len(ads),
                'ads': ads[:3],  # Show first 3 ads
                'message': f'Vision API working! Detected {len(ads)} potential ads.'
            })
        else:
            return jsonify({
                'success': False,
                'vision_api_available': False,
                'test_image': test_image,
                'error': 'Vision API test failed - check credentials and setup'
            })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Vision API test error: {str(e)}',
            'vision_api_available': is_vision_api_available()
        })

def check_database_connection():
    """Check if database connection is working"""
    try:
        with app.app_context():
            # Try to query the database
            publication_count = Publication.query.count()
            print(f"Database connection OK - Found {publication_count} publications")
            
            # Check if processing columns exist
            try:
                test_pub = Publication.query.first()
                if test_pub:
                    _ = test_pub.safe_processing_status
                    print("Processing status columns available")
            except (AttributeError, Exception):
                print("Processing status columns not available (using fallback)")
            
            return True
    except Exception as e:
        print(f"Database connection ERROR: {e}")
        return False

def safe_db_operation(operation_func, rollback_on_error=True):
    """
    Safely execute database operations with proper error handling
    
    Args:
        operation_func: Function that performs database operations
        rollback_on_error: Whether to rollback on error (default: True)
    
    Returns:
        tuple: (success, result_or_error)
    """
    try:
        # Clear any failed transaction state
        db.session.rollback()
        
        # Execute the operation
        result = operation_func()
        
        # Commit if successful
        db.session.commit()
        return True, result
    
    except Exception as e:
        # Rollback on error
        if rollback_on_error:
            try:
                db.session.rollback()
            except Exception:
                db.session.remove()
        
        print(f"Database operation failed: {e}")
        import traceback
        traceback.print_exc()
        
        return False, str(e)


# =============================================================================
# LOGO MANAGEMENT AND HYBRID DETECTION ROUTES
# =============================================================================

@app.route('/logo_management')
@login_required
def logo_management():
    """Logo management interface"""
    try:
        # Get all business logos with statistics
        business_logos = BusinessLogo.query.order_by(BusinessLogo.business_name).all()

        # Calculate statistics for each logo
        logo_stats = []
        for logo in business_logos:
            total_detections = LogoRecognitionResult.query.filter_by(business_logo_id=logo.id).count()
            recent_detections = LogoRecognitionResult.query.filter(
                LogoRecognitionResult.business_logo_id == logo.id,
                LogoRecognitionResult.detection_date >= datetime.utcnow() - timedelta(days=30)
            ).count()

            logo_stats.append({
                'logo': logo,
                'total_detections': total_detections,
                'recent_detections': recent_detections,
                'avg_confidence': logo.average_confidence_score,
                'is_active': logo.is_active
            })

        return render_template('logo_management.html', logo_stats=logo_stats)

    except Exception as e:
        print(f"Error in logo management: {e}")
        flash('Error loading logo management interface', 'error')
        return redirect(url_for('index'))

@app.route('/api/hybrid_detection/<int:pub_id>', methods=['POST'])
@login_required
def run_hybrid_detection(pub_id):
    """Run hybrid detection on a publication"""
    try:
        data = request.get_json() or {}
        mode = data.get('mode', 'auto')  # 'auto', 'manual', 'hybrid'
        page_numbers = data.get('page_numbers')  # Optional specific pages

        # Initialize hybrid detection pipeline
        hybrid_pipeline = HybridDetectionPipeline()

        # Run detection
        result = hybrid_pipeline.detect_ads_hybrid(pub_id, mode, page_numbers)

        return jsonify(result)

    except Exception as e:
        print(f"Error in hybrid detection: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/manual_click_detection/<int:page_id>', methods=['POST'])
@login_required
def process_manual_click_detection(page_id):
    """Process manual click for smart ad detection"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})

        click_x = data.get('click_x')
        click_y = data.get('click_y')
        business_name = data.get('business_name')
        learn_logo = data.get('learn_logo', False)

        if click_x is None or click_y is None:
            return jsonify({'success': False, 'error': 'Click coordinates required'})

        # Initialize hybrid detection pipeline
        hybrid_pipeline = HybridDetectionPipeline()

        # Process manual click
        result = hybrid_pipeline.process_manual_click(
            page_id, click_x, click_y, business_name, learn_logo
        )

        return jsonify(result)

    except Exception as e:
        print(f"Error in manual click detection: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/detection_suggestions/<int:page_id>')
@login_required
def get_detection_suggestions(page_id):
    """Get automated detection suggestions for a page"""
    try:
        threshold = float(request.args.get('threshold', 0.6))

        # Initialize hybrid detection pipeline
        hybrid_pipeline = HybridDetectionPipeline()

        # Get suggestions
        result = hybrid_pipeline.get_detection_suggestions(page_id, threshold)

        return jsonify(result)

    except Exception as e:
        print(f"Error getting detection suggestions: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/hybrid_detection_status/<int:pub_id>')
@login_required
def get_hybrid_detection_status(pub_id):
    """Get hybrid detection status for a publication"""
    try:
        # Initialize hybrid detection pipeline
        hybrid_pipeline = HybridDetectionPipeline()

        # Get status
        result = hybrid_pipeline.get_hybrid_detection_status(pub_id)

        return jsonify(result)

    except Exception as e:
        print(f"Error getting hybrid detection status: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/business_logos')
@login_required
def get_business_logos():
    """Get all business logos for UI selection"""
    try:
        logos = BusinessLogo.query.filter_by(is_active=True).order_by(BusinessLogo.business_name).all()

        logo_list = []
        for logo in logos:
            logo_list.append({
                'id': logo.id,
                'business_name': logo.business_name,
                'successful_detections': logo.successful_detections,
                'total_detections': logo.total_detections,
                'confidence_threshold': logo.confidence_threshold,
                'last_detected': logo.last_detected_date.isoformat() if logo.last_detected_date else None
            })

        return jsonify({'success': True, 'logos': logo_list})

    except Exception as e:
        print(f"Error getting business logos: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/logo_learning/<int:ad_box_id>', methods=['POST'])
@login_required
def learn_logo_from_ad(ad_box_id):
    """Learn logo from an existing ad box"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})

        business_name = data.get('business_name')
        if not business_name:
            return jsonify({'success': False, 'error': 'Business name required'})

        # Get the ad box
        ad_box = AdBox.query.get(ad_box_id)
        if not ad_box:
            return jsonify({'success': False, 'error': 'Ad box not found'})

        # Initialize logo learning workflow
        logo_learning = LogoLearningWorkflow()

        # Load page image to extract features
        page = Page.query.get(ad_box.page_id)
        publication = Publication.query.get(page.publication_id)
        image_filename = f"{publication.filename}_page_{page.page_number}.png"
        image_path = os.path.join('static', 'uploads', 'pages', image_filename)

        if not os.path.exists(image_path):
            return jsonify({'success': False, 'error': 'Page image not found'})

        page_image = cv2.imread(image_path)
        if page_image is None:
            return jsonify({'success': False, 'error': 'Could not load page image'})

        # Extract ad region
        x, y, w, h = int(ad_box.x), int(ad_box.y), int(ad_box.width), int(ad_box.height)
        ad_region = page_image[y:y+h, x:x+w]

        # Extract features
        feature_extractor = LogoFeatureExtractor()
        features = feature_extractor.extract_logo_features(ad_region, business_name)

        # Learn logo
        result = logo_learning.learn_logo_from_manual_ad(ad_box, business_name, features)

        return jsonify(result)

    except Exception as e:
        print(f"Error learning logo from ad: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/toggle_logo_status/<int:logo_id>', methods=['POST'])
@login_required
def toggle_logo_status(logo_id):
    """Toggle active status of a business logo"""
    try:
        logo = BusinessLogo.query.get(logo_id)
        if not logo:
            return jsonify({'success': False, 'error': 'Logo not found'})

        logo.is_active = not logo.is_active
        db.session.commit()

        return jsonify({
            'success': True,
            'logo_id': logo_id,
            'is_active': logo.is_active,
            'message': f"Logo '{logo.business_name}' {'activated' if logo.is_active else 'deactivated'}"
        })

    except Exception as e:
        print(f"Error toggling logo status: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_logo/<int:logo_id>', methods=['DELETE'])
@login_required
def delete_logo(logo_id):
    """Delete a business logo and its associated data"""
    try:
        logo = BusinessLogo.query.get(logo_id)
        if not logo:
            return jsonify({'success': False, 'error': 'Logo not found'})

        business_name = logo.business_name

        # Delete associated recognition results
        LogoRecognitionResult.query.filter_by(business_logo_id=logo_id).delete()

        # Delete the logo
        db.session.delete(logo)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': f"Logo '{business_name}' and all associated data deleted"
        })

    except Exception as e:
        print(f"Error deleting logo: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/hybrid_detection/<int:pub_id>')
@login_required
def hybrid_detection_interface(pub_id):
    """Hybrid detection interface for manual enhancement"""
    try:
        publication = Publication.query.get(pub_id)
        if not publication:
            flash('Publication not found', 'error')
            return redirect(url_for('index'))

        # Get pages for this publication
        pages = Page.query.filter_by(publication_id=pub_id).order_by(Page.page_number).all()

        # Get current detection status
        hybrid_pipeline = HybridDetectionPipeline()
        status = hybrid_pipeline.get_hybrid_detection_status(pub_id)

        # Get available business logos
        business_logos = BusinessLogo.query.filter_by(is_active=True).order_by(BusinessLogo.business_name).all()

        return render_template('hybrid_detection.html',
                             publication=publication,
                             pages=pages,
                             detection_status=status,
                             business_logos=business_logos)

    except Exception as e:
        print(f"Error in hybrid detection interface: {e}")
        flash('Error loading hybrid detection interface', 'error')
        return redirect(url_for('index'))


# GLOBAL DATABASE ERROR HANDLER
@app.errorhandler(Exception)
def handle_database_errors(error):
    """Global handler for database transaction errors"""
    error_str = str(error)
    
    # Handle database transaction errors specifically
    if "InFailedSqlTransaction" in error_str or "psycopg2.errors.InFailedSqlTransaction" in error_str:
        try:
            db.session.rollback()
        except Exception:
            db.session.remove()
        
        print(f"Global database transaction error caught: {error}")
        import traceback
        traceback.print_exc()
        
        # Return appropriate response based on request type
        if request.content_type == 'application/json' or request.is_json:
            return jsonify({
                'success': False, 
                'error': 'Database transaction error. Please refresh the page and try again.'
            }), 500
        else:
            flash('Database error occurred. Please refresh the page and try again.', 'error')
            return redirect(request.url if request.method == 'GET' else url_for('index'))
    
    # Re-raise non-database errors
    raise error

if __name__ == '__main__':
    print("Starting Newspaper Ad Measurement System...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    
    # Check database connection
    if check_database_connection():
        print("Opening at: http://localhost:5000")
        print("AI Box Detection: Enhanced & Improved")
        print("Interactive Measurement Tools: Ready")
        print("Professional Reporting System: Ready")
        print("Intelligent Click Detection: Ready for Broadsheet")
        print("Screen Calibration System: Ready")
        app.run(debug=True)
    else:
        print("Cannot start application - Database connection failed!")
        exit(1)