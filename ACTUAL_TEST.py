#!/usr/bin/env python3
"""
ACTUAL TEST: Upload OA-2025-01-01.pdf through web interface and verify results
NO MORE CLAIMS - ACTUAL PROOF
"""
import requests
import time
import os
from app import app, db, Publication, Page, AdBox

def actual_web_test():
    """Actually test the web interface like a real user"""
    print("=== ACTUAL WEB INTERFACE TEST ===")

    # Test 1: Can we access the web interface?
    try:
        response = requests.get('http://localhost:5000', timeout=10)
        print(f"Web interface accessible: {response.status_code}")
        if response.status_code not in [200, 302]:
            print(f"ERROR: Web interface not working: {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: Cannot connect to web interface: {e}")
        return False

    # Test 2: Can we login?
    session = requests.Session()

    # Get login page
    try:
        login_page = session.get('http://localhost:5000/login', timeout=10)
        print(f"Login page accessible: {login_page.status_code}")
    except Exception as e:
        print(f"ERROR: Cannot access login page: {e}")
        return False

    # Login (password only, no username)
    login_data = {
        'password': 'CCCitizen56101!'
    }

    try:
        login_response = session.post('http://localhost:5000/login', data=login_data, timeout=10)
        print(f"Login attempt: {login_response.status_code}")

        if login_response.status_code == 302:
            print("Login successful (302 redirect)")
        elif login_response.status_code == 200:
            if 'dashboard' in login_response.text.lower() or 'upload' in login_response.text.lower():
                print("Login successful (200 with dashboard)")
            else:
                print("ERROR: Login failed - still on login page")
                return False
        else:
            print(f"ERROR: Login failed: {login_response.status_code}")
            return False

    except Exception as e:
        print(f"ERROR: Login failed: {e}")
        return False

    # Test 3: Can we access upload page?
    try:
        upload_page = session.get('http://localhost:5000/upload', timeout=10)
        print(f"Upload page accessible: {upload_page.status_code}")
        if upload_page.status_code != 200:
            print("ERROR: Cannot access upload page")
            return False
    except Exception as e:
        print(f"ERROR: Cannot access upload page: {e}")
        return False

    # Test 4: Can we upload the OA file?
    pdf_path = "C:\\Users\\trevo\\newspaper-ad-measurement\\OA-2025-01-01.pdf"
    if not os.path.exists(pdf_path):
        print(f"ERROR: Test file not found: {pdf_path}")
        return False

    print(f"Uploading file: {pdf_path}")

    # Clear existing publications first
    with app.app_context():
        existing = Publication.query.filter_by(original_filename="OA-2025-01-01.pdf").first()
        if existing:
            pages = Page.query.filter_by(publication_id=existing.id).all()
            for page in pages:
                AdBox.query.filter_by(page_id=page.id).delete()
                db.session.delete(page)
            db.session.delete(existing)
            db.session.commit()
            print("Cleared existing publication")

    upload_data = {
        'publication_type': 'broadsheet'
    }

    try:
        with open(pdf_path, 'rb') as pdf_file:
            files = {
                'pdf_file': ('OA-2025-01-01.pdf', pdf_file, 'application/pdf')
            }

            upload_response = session.post('http://localhost:5000/upload',
                                         data=upload_data,
                                         files=files,
                                         timeout=60)  # Long timeout for processing

        print(f"Upload response: {upload_response.status_code}")

        if upload_response.status_code == 302:
            redirect_url = upload_response.headers.get('Location', '')
            print(f"Upload successful - redirected to: {redirect_url}")

            if '/processing/' in redirect_url:
                pub_id = redirect_url.split('/processing/')[-1]
                print(f"Publication ID: {pub_id}")

                # Wait for processing to complete
                print("Waiting for processing to complete...")
                time.sleep(10)  # Give it time to process

                return actual_verify_results(pub_id)
            else:
                print("ERROR: Unexpected redirect URL")
                return False

        elif upload_response.status_code == 200:
            # Check if this is a processing page (success) or error page
            response_text = upload_response.text
            if 'Processing' in response_text and 'title>Processing' in response_text:
                print("Upload successful - showing processing page")

                # Get publication ID from database (latest publication with this filename)
                with app.app_context():
                    latest_pub = Publication.query.filter_by(original_filename="OA-2025-01-01.pdf").order_by(Publication.id.desc()).first()
                    if latest_pub:
                        pub_id = latest_pub.id
                        print(f"Publication ID: {pub_id}")
                        print(f"Processed status: {latest_pub.processed}")

                        # Wait for processing to complete if needed
                        if not latest_pub.processed:
                            print("Waiting for processing to complete...")
                            time.sleep(10)

                        return actual_verify_results(pub_id)
                    else:
                        print("ERROR: Could not find publication in database")
                        return False
            else:
                print("ERROR: Upload returned 200 (probably form errors)")
                print(f"Response contains: {response_text[:500]}")
                return False
        else:
            print(f"ERROR: Upload failed: {upload_response.status_code}")
            return False

    except Exception as e:
        print(f"ERROR: Upload failed: {e}")
        return False

def actual_verify_results(pub_id):
    """Verify the actual results in the database"""
    print(f"\\n=== VERIFYING RESULTS FOR PUBLICATION {pub_id} ===")

    with app.app_context():
        try:
            # Get publication
            publication = Publication.query.get(int(pub_id))
            if not publication:
                print(f"ERROR: Publication {pub_id} not found")
                return False

            print(f"Publication: {publication.original_filename}")
            print(f"Processed: {publication.processed}")
            print(f"Total ad inches: {publication.total_ad_inches}")

            # Get pages
            pages = Page.query.filter_by(publication_id=publication.id).all()
            print(f"Pages: {len(pages)}")

            # Count ads
            total_ads = 0
            ads_by_page = {}

            for page in pages:
                page_ads = AdBox.query.filter_by(page_id=page.id).all()
                if len(page_ads) > 0:
                    ads_by_page[page.page_number] = len(page_ads)
                    total_ads += len(page_ads)

            print(f"Ads by page: {ads_by_page}")
            print(f"TOTAL ADS: {total_ads}")

            # SUCCESS CRITERIA
            if total_ads >= 15:
                print(f"\\n*** SUCCESS! {total_ads} ads detected ***")
                print("The web interface actually works!")
                return True
            else:
                print(f"\\n*** FAILURE: Only {total_ads} ads detected ***")
                return False

        except Exception as e:
            print(f"ERROR verifying results: {e}")
            return False

if __name__ == "__main__":
    success = actual_web_test()
    if success:
        print("\\nWEB INTERFACE ACTUALLY WORKS!")
    else:
        print("\\nWEB INTERFACE STILL BROKEN")