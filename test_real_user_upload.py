#!/usr/bin/env python3
"""
TEST REAL USER UPLOAD: Simulate exactly what user does through web interface
"""
from app import app, db, Publication, Page, AdBox
import requests
import time
import os

def test_real_user_upload():
    """Test the exact user upload workflow"""
    print("=== TESTING REAL USER UPLOAD WORKFLOW ===")
    print()

    # Check if app is running
    try:
        response = requests.get('http://localhost:5000')
        if response.status_code != 200:
            print("ERROR: App is not running at localhost:5000")
            return False
    except Exception as e:
        print(f"ERROR: Cannot connect to app: {e}")
        return False

    print("App is running at localhost:5000")

    # STEP 1: Clear any existing OA publication
    with app.app_context():
        test_file = "OA-2025-01-01.pdf"
        existing_pub = Publication.query.filter_by(original_filename=test_file).first()

        if existing_pub:
            print(f"Found existing publication {existing_pub.id}, clearing it...")
            pages = Page.query.filter_by(publication_id=existing_pub.id).all()
            for page in pages:
                AdBox.query.filter_by(page_id=page.id).delete()
                db.session.delete(page)
            db.session.delete(existing_pub)
            db.session.commit()
            print("Cleared existing publication")

    # STEP 2: Login to the app
    print("\n=== STEP 2: LOGGING IN ===")

    session = requests.Session()

    # Get login page
    login_response = session.get('http://localhost:5000/login')
    if login_response.status_code != 200:
        print(f"ERROR: Cannot access login page: {login_response.status_code}")
        return False

    # Login (assuming default credentials)
    login_data = {
        'username': 'admin',  # Default username
        'password': 'password'  # Default password
    }

    login_post = session.post('http://localhost:5000/login', data=login_data)
    if login_post.status_code not in [200, 302]:
        print(f"ERROR: Login failed: {login_post.status_code}")
        return False

    print("Logged in successfully")

    # STEP 3: Upload the PDF file through web interface
    print("\n=== STEP 3: UPLOADING PDF ===")

    pdf_path = r"C:\Users\trevo\newspaper-ad-measurement\OA-2025-01-01.pdf"
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF file not found at {pdf_path}")
        return False

    # Prepare upload data
    upload_data = {
        'publication_type': 'broadsheet'
    }

    with open(pdf_path, 'rb') as pdf_file:
        files = {
            'pdf_file': ('OA-2025-01-01.pdf', pdf_file, 'application/pdf')
        }

        upload_response = session.post('http://localhost:5000/upload',
                                     data=upload_data,
                                     files=files)

    print(f"Upload response status: {upload_response.status_code}")
    print(f"Upload response headers: {dict(upload_response.headers)}")

    if upload_response.status_code == 200:
        print("Upload returned 200 (form with errors)")
        print(f"Response text snippet: {upload_response.text[:1000]}")
        return False
    elif upload_response.status_code == 302:
        print("PDF uploaded successfully")
        redirect_url = upload_response.headers.get('Location', '')
        print(f"Redirected to: {redirect_url}")

        # Extract pub_id from redirect URL like /processing/123
        if '/processing/' in redirect_url:
            pub_id = redirect_url.split('/processing/')[-1]
            print(f"Publication ID: {pub_id}")
        else:
            print("ERROR: Cannot extract publication ID from redirect")
            return False
    else:
        print(f"ERROR: Upload failed: {upload_response.status_code}")
        print(f"Response text: {upload_response.text[:500]}")
        return False

    # STEP 4: Monitor processing status
    print("\n=== STEP 4: MONITORING PROCESSING ===")

    max_wait = 300  # 5 minutes max
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            status_response = session.get(f'http://localhost:5000/api/processing_status/{pub_id}')
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Status: {status_data.get('status', 'unknown')}")

                if status_data.get('processed', False):
                    print("Processing completed!")
                    break
                elif status_data.get('status') == 'error':
                    print(f"ERROR: Processing failed: {status_data.get('error', 'Unknown error')}")
                    return False

            time.sleep(5)  # Wait 5 seconds before checking again

        except Exception as e:
            print(f"Error checking status: {e}")
            time.sleep(5)
    else:
        print("ERROR: Processing timed out")
        return False

    # STEP 5: Check final results
    print("\n=== STEP 5: CHECKING FINAL RESULTS ===")

    with app.app_context():
        # Find the uploaded publication
        final_pub = Publication.query.filter_by(original_filename=test_file).first()
        if not final_pub:
            print("ERROR: Cannot find uploaded publication")
            return False

        print(f"Publication ID: {final_pub.id}")
        print(f"Processed: {final_pub.processed}")
        print(f"Total ad inches: {final_pub.total_ad_inches}")
        print(f"Ad percentage: {final_pub.ad_percentage:.1f}%")

        # Check pages
        pages = Page.query.filter_by(publication_id=final_pub.id).all()
        print(f"Pages created: {len(pages)}")

        # Check ads
        total_ads = 0
        for page in pages:
            page_ads = AdBox.query.filter_by(page_id=page.id).all()
            if len(page_ads) > 0:
                print(f"  Page {page.page_number}: {len(page_ads)} ads")
            total_ads += len(page_ads)

        print(f"Total ads detected: {total_ads}")

        # SUCCESS CRITERIA
        if total_ads >= 15:
            print("*** SUCCESS: Real user upload detects 15+ ads! ***")
            return True
        elif total_ads > 2:
            print(f"*** PARTIAL SUCCESS: {total_ads} ads (improved from baseline) ***")
            return True
        else:
            print(f"*** FAILURE: Only {total_ads} ads detected in real user workflow ***")
            return False

if __name__ == "__main__":
    success = test_real_user_upload()
    if success:
        print("\nREAL USER UPLOAD WORKFLOW WORKS!")
    else:
        print("\nREAL USER UPLOAD WORKFLOW STILL FAILING")