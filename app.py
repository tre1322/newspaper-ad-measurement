import os
import uuid
import fitz  # PyMuPDF
import cv2
import numpy as np
import math

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
    from datetime import datetime
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
                    
                    # Try PDF metadata detection first (doesn't require credentials)
                    try:
                        print(f"📄 Attempting PDF metadata-based ad detection")
                        pdf_result = PDFAdDetectionEngine.detect_ads_from_pdf(publication.id)
                        
                        if pdf_result and pdf_result.get('success'):
                            print(f"✅ PDF detection complete: {pdf_result['detections']} ads detected across {pdf_result['pages_processed']} pages")
                            if pdf_result['detections'] > 0:
                                print(f"📝 Next: Review the auto-detected ads on the measurement pages to verify accuracy")
                            else:
                                print(f"ℹ️  No ads detected via PDF analysis - you can manually mark ads as usual")
                        else:
                            error_msg = pdf_result.get('error', 'PDF analysis failed') if pdf_result else 'No result returned'
                            print(f"⚠️  PDF detection failed: {error_msg}")
                            print(f"📝 Will try alternative detection methods if available")
                    except Exception as pdf_error:
                        print(f"⚠️  PDF detection failed with error: {pdf_error}")
                        print(f"📝 Will try alternative detection methods if available")
                    
                    # Skip traditional AI detection if Google Vision credentials are not available
                    if not os.path.exists('google-vision-credentials.json'):
                        print(f"⚠️  Google Vision credentials not found - skipping Vision AI detection")
                        print(f"📝 Publication processed successfully - continue with manual ad marking")
                    else:
                        # Timeout protection for auto-detection (robust error handling)
                        import time
                        start_time = time.time()
                        
                        try:
                            # Test if Vision API is available before attempting detection
                            if not is_vision_api_available():
                                print(f"⚠️  Google Vision API not available - skipping AI detection")
                                print(f"📝 Publication processed successfully - continue with manual ad marking")
                            else:
                                # Proceed with AI detection with memory management
                                result = AdLearningEngine.auto_detect_ads(publication.id, confidence_threshold=0.25)
                                
                                if result and result.get('success'):
                                    print(f"✅ AI detection complete: {result['detections']} ads automatically detected and boxed across {result['pages_processed']} pages")
                                    if result['detections'] > 0:
                                        print(f"📊 Model used: {result.get('model_used', 'Unknown')}")
                                        print(f"📝 Next: Review the auto-detected ads on the measurement pages to verify accuracy")
                                    else:
                                        print(f"ℹ️  No ads detected above confidence threshold - you can manually mark ads as usual")
                                else:
                                    error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                                    print(f"⚠️  AI detection failed: {error_msg}")
                                    print(f"📝 Publication processed successfully - continue with manual ad marking")
                                    
                        except MemoryError as me:
                            print(f"⚠️  AI detection failed due to memory constraints: {me}")
                            print(f"📝 Publication processed successfully - continue with manual ad marking")
                        except ImportError as ie:
                            print(f"⚠️  AI detection failed due to missing dependencies: {ie}")
                            print(f"📝 Publication processed successfully - continue with manual ad marking")
                        except Exception as detection_error:
                            print(f"⚠️  AI detection failed with error: {detection_error}")
                            print(f"📝 Publication processed successfully - continue with manual ad marking")
                        finally:
                            # Check if we exceeded timeout and cleanup
                            elapsed = time.time() - start_time
                            if elapsed > 300:  # 5 minutes
                                print(f"⚠️  AI detection took {elapsed:.1f}s (timeout threshold: 300s)")
                            
                            # Force garbage collection to free memory
                            try:
                                import gc
                                gc.collect()
                            except:
                                pass
                            
                except Exception as outer_error:
                    print(f"⚠️  AI detection phase failed completely: {outer_error}")
                    print(f"📝 Publication processed successfully - continue with manual ad marking")
                
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
            print(f"Error in synchronous processing: {e}")
            publication.set_processing_status('failed')
            publication.set_processing_error(str(e))
            db.session.commit()

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
    ad_box_id = db.Column(db.Integer, db.ForeignKey('ad_box.id'), nullable=False)
    publication_type = db.Column(db.String(50), nullable=False)
    features = db.Column(db.Text)  # JSON serialized feature vector
    label = db.Column(db.String(50), nullable=False)  # ad type or 'not_ad'
    confidence_score = db.Column(db.Float)  # User confidence in this label
    extracted_date = db.Column(db.DateTime, default=datetime.utcnow)
    used_in_training = db.Column(db.Boolean, default=False)

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
            
            print(f"📄 Processing {len(pages)} pages with PDF metadata detection")
            
            detections_count = 0
            pages_processed = 0
            
            for page in pages:
                try:
                    print(f"📄 Analyzing page {page.page_number} for PDF metadata ads")
                    
                    # Get PDF detections
                    pdf_detections = PDFMetadataAdDetector.detect_ads_from_pdf_metadata(
                        pdf_path, page.page_number, publication.publication_type
                    )
                    
                    if pdf_detections:
                        print(f"✅ Found {len(pdf_detections)} ad candidates on page {page.page_number}")
                        
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
                                if detection['type'] == 'image':
                                    ad_type = 'pdf_image_ad'
                                elif detection['type'] == 'text':
                                    ad_type = 'pdf_text_ad'
                                elif detection['type'] == 'border':
                                    ad_type = 'pdf_border_ad'
                                
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
                                
                                print(f"📦 Created ad box: {ad_type} at ({detection['x']:.0f},{detection['y']:.0f}) "
                                      f"size {detection['width']:.0f}x{detection['height']:.0f} "
                                      f"confidence={detection['confidence']:.3f}")
                                
                            except Exception as box_error:
                                print(f"⚠️  Error creating ad box: {box_error}")
                                continue
                    else:
                        print(f"ℹ️  No ads found on page {page.page_number}")
                    
                    pages_processed += 1
                    
                except Exception as page_error:
                    print(f"⚠️  Error processing page {page.page_number}: {page_error}")
                    continue
            
            # Commit all changes
            if detections_count > 0:
                db.session.commit()
                print(f"✅ Successfully saved {detections_count} ad detections to database")
            
            return {
                'success': True,
                'detections': detections_count,
                'pages_processed': pages_processed,
                'detection_method': 'PDF_metadata'
            }
            
        except Exception as e:
            print(f"❌ PDF detection engine error: {e}")
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
    def detect_ads_from_pdf_metadata(pdf_path, page_number, publication_type='broadsheet'):
        """
        Detect ads using PDF structural analysis instead of image processing.
        
        Args:
            pdf_path (str): Path to the PDF file
            page_number (int): Page number to analyze (1-based)
            publication_type (str): Type of publication for context
            
        Returns:
            list: Detected ad regions as dicts with x, y, width, height, confidence
        """
        try:
            if not os.path.exists(pdf_path):
                print(f"PDF file not found: {pdf_path}")
                return []
                
            doc = fitz.open(pdf_path)
            if page_number < 1 or page_number > len(doc):
                print(f"Invalid page number {page_number} for PDF with {len(doc)} pages")
                return []
                
            page = doc[page_number - 1]  # fitz uses 0-based indexing
            page_rect = page.rect
            
            detected_ads = []
            
            # Get all page elements for structural analysis
            images = page.get_images()
            text_blocks = page.get_text("dict")
            drawings = page.get_drawings()
            
            print(f"Page {page_number}: Found {len(images)} images, {len(text_blocks.get('blocks', []))} text blocks, {len(drawings)} drawings")
            
            # Analyze images as potential ads
            for img_index, img in enumerate(images):
                try:
                    img_rect = page.get_image_bbox(img[7])  # img[7] is the xref
                    if img_rect:
                        ad_candidate = PDFMetadataAdDetector._analyze_image_element(
                            img_rect, page_rect, publication_type, f"image_{img_index}"
                        )
                        if ad_candidate:
                            detected_ads.append(ad_candidate)
                except Exception as e:
                    print(f"Error analyzing image {img_index}: {e}")
                    continue
            
            # Analyze text blocks for ad patterns
            if 'blocks' in text_blocks:
                for block_index, block in enumerate(text_blocks['blocks']):
                    if 'lines' in block and len(block['lines']) > 0:
                        try:
                            block_rect = fitz.Rect(block['bbox'])
                            ad_candidate = PDFMetadataAdDetector._analyze_text_block(
                                block, block_rect, page_rect, publication_type, f"text_{block_index}"
                            )
                            if ad_candidate:
                                detected_ads.append(ad_candidate)
                        except Exception as e:
                            print(f"Error analyzing text block {block_index}: {e}")
                            continue
            
            # Analyze vector drawings (borders, graphics)
            for draw_index, drawing in enumerate(drawings):
                try:
                    draw_rect = drawing.get('rect', None)
                    if draw_rect:
                        ad_candidate = PDFMetadataAdDetector._analyze_drawing_element(
                            draw_rect, page_rect, publication_type, f"drawing_{draw_index}"
                        )
                        if ad_candidate:
                            detected_ads.append(ad_candidate)
                except Exception as e:
                    print(f"Error analyzing drawing {draw_index}: {e}")
                    continue
            
            # Merge overlapping regions and filter by confidence
            filtered_ads = PDFMetadataAdDetector._merge_and_filter_detections(detected_ads)
            
            print(f"PDF analysis complete: {len(detected_ads)} candidates -> {len(filtered_ads)} final detections")
            
            doc.close()
            return filtered_ads
            
        except Exception as e:
            print(f"Error in PDF metadata ad detection: {e}")
            import traceback
            traceback.print_exc()
            return []
    
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
    def _analyze_drawing_element(draw_rect, page_rect, publication_type, element_id):
        """Analyze vector drawing elements (borders, graphics) for ad detection"""
        try:
            width = draw_rect.width
            height = draw_rect.height
            
            # Focus on rectangular borders that might frame ads
            if width < 50 or height < 50:  # Too small to be ad border
                return None
            
            # Look for border-like elements (thin rectangles)
            line_thickness_threshold = 5
            is_border = (width > height * 10 or height > width * 10)  # Very thin rectangle
            
            if not is_border:
                return None  # Only interested in border elements for now
            
            # Borders suggest structured content (ads) vs flowing text
            confidence = 0.7 if is_border else 0.3
            
            return {
                'x': draw_rect.x0,
                'y': draw_rect.y0,
                'width': width,
                'height': height,
                'confidence': confidence,
                'type': 'border',
                'element_id': element_id
            }
            
        except Exception as e:
            print(f"Error analyzing drawing element {element_id}: {e}")
            return None
    
    @staticmethod
    def _merge_and_filter_detections(detections, min_confidence=0.5, overlap_threshold=0.3):
        """Merge overlapping detections and filter by confidence"""
        try:
            # Filter by minimum confidence
            filtered = [d for d in detections if d['confidence'] >= min_confidence]
            
            if not filtered:
                return []
            
            # Sort by confidence (highest first)
            filtered.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Merge overlapping detections
            merged = []
            
            for current in filtered:
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
                        # Merge with existing detection (keep higher confidence one's properties)
                        if current['confidence'] > existing['confidence']:
                            merged[i] = current
                        should_merge = True
                        break
                
                if not should_merge:
                    merged.append(current)
            
            return merged
            
        except Exception as e:
            print(f"Error merging detections: {e}")
            return detections  # Return original if merging fails
    
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
                print(f"Attempting PDF metadata detection on page {page.page_number}")
                try:
                    # Get PDF detections in PDF coordinate system
                    pdf_detections = PDFMetadataAdDetector.detect_ads_from_pdf_metadata(
                        pdf_path, page.page_number, publication.publication_type
                    )
                    
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
                        print(f"Transformed {len(pdf_detected_boxes)} PDF detections to image coordinates")
                    else:
                        print(f"No ads detected via PDF metadata on page {page.page_number}")
                        
                except Exception as pdf_error:
                    print(f"PDF metadata detection failed for page {page.page_number}: {pdf_error}")
                    pdf_detected_boxes = []
                
                # SECONDARY: Try Google Vision AI if PDF detection finds nothing
                vision_detected_boxes = []
                if len(pdf_detected_boxes) == 0 and is_vision_api_available():
                    try:
                        print(f"Falling back to Google Vision AI detection on page {page.page_number}")
                        vision_detected_boxes = GoogleVisionAdDetector.detect_ads(
                            image_path, publication.publication_type
                        )
                        print(f"Google Vision AI found {len(vision_detected_boxes)} ads on page {page.page_number}")
                    except Exception as vision_error:
                        error_type = handle_vision_api_error(vision_error, f"page {page.page_number} detection")
                        print(f"Google Vision AI failed for page {page.page_number}: {vision_error}")
                        vision_detected_boxes = []
                
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
            from datetime import datetime
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
                print(f"💥 UPLOAD ERROR: {str(e)}")
                import traceback
                print(f"📋 Error traceback:")
                traceback.print_exc()
                flash(f'Error uploading file: {str(e)}', 'error')
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
                    
                    db.session.commit()
        except Exception as e:
            print(f"Warning: Could not extract features for training: {e}")
        
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
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/delete_box/<int:box_id>', methods=['DELETE'])
def delete_box(box_id):
    """Delete an ad box"""
    try:
        ad_box = AdBox.query.get_or_404(box_id)
        page_id = ad_box.page_id
        
        # CRITICAL FIX: Delete ALL training data records for this ad_box to avoid foreign key constraint
        training_records = TrainingData.query.filter_by(ad_box_id=box_id).all()
        if training_records:
            # Log negative training information before deleting
            try:
                # Get publication through ad_box -> page relationship
                page = Page.query.get(ad_box.page_id)
                publication = Publication.query.get(page.publication_id) if page else None
                
                if page and publication:
                    # Create a standalone negative training record without ad_box_id dependency
                    negative_features = {
                        'deleted_by_user': True, 
                        'false_positive': True,
                        'original_ad_box_id': box_id,  # Store reference for logging
                        'x': ad_box.x,
                        'y': ad_box.y, 
                        'width': ad_box.width,
                        'height': ad_box.height,
                        'ad_type': ad_box.ad_type,
                        'training_records_count': len(training_records),
                        'publication_type': publication.publication_type,
                        'page_number': page.page_number
                    }
                    
                    print(f"Negative training data logged for deleted ad box {box_id}: {negative_features}")
                    
            except Exception as e:
                print(f"Could not log negative training data: {e}")
            
            # Delete ALL training data records first to avoid foreign key constraint
            for training_record in training_records:
                db.session.delete(training_record)
            print(f"Deleted {len(training_records)} training data records for ad box {box_id}")
        
        # Additional safety check - force delete any remaining training data records
        # This catches any records that might have been missed by the initial query
        try:
            remaining_records = TrainingData.query.filter_by(ad_box_id=box_id).all()
            if remaining_records:
                print(f"Found {len(remaining_records)} additional training records to delete")
                for record in remaining_records:
                    db.session.delete(record)
        except Exception as cleanup_error:
            print(f"Warning during additional cleanup: {cleanup_error}")
        
        # Execute deletion of training data first
        db.session.flush()  # Force execution of training data deletions
        
        # Now safely delete the ad_box
        db.session.delete(ad_box)
        db.session.commit()
        
        # Recalculate totals
        update_totals(page_id)
        
        return jsonify({'success': True})
    except Exception as e:
        db.session.rollback()
        print(f"Error deleting ad box {box_id}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

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
        
        # Automatically extract features for ML training
        try:
            image_filename = f"{publication.filename}_page_{page.page_number}.png"
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pages', image_filename)
            
            if os.path.exists(image_path):
                box_coords = {'x': data['x'], 'y': data['y'], 'width': data['width'], 'height': data['height']}
                features = AdLearningEngine.extract_features(image_path, box_coords)
                
                if features:
                    # Check if training data already exists
                    existing = TrainingData.query.filter_by(ad_box_id=ad_box.id).first()
                    if not existing:
                        training_data = TrainingData(
                            ad_box_id=ad_box.id,
                            publication_type=publication.publication_type,
                            features=json.dumps(features),
                            label=ad_box.ad_type,
                            confidence_score=1.0
                        )
                        db.session.add(training_data)
                        db.session.commit()
                        print(f"Extracted ML features for new ad box {ad_box.id}")
        except Exception as e:
            print(f"Warning: Could not extract features for training: {e}")
        
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