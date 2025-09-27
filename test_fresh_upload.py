#!/usr/bin/env python3
"""
Test fresh upload to verify NewspaperDomainDetector is working in web interface
"""
import requests
import time
import os
from app import app, db, Publication, Page, AdBox

def test_fresh_upload():
    """Upload fresh file and check the actual ad box coordinates"""
    print("=== TESTING FRESH UPLOAD WITH NEWSPAPER DOMAIN DETECTOR ===")

    # Clear ALL existing publications first
    with app.app_context():
        publications = Publication.query.filter_by(original_filename="OA-2025-01-01.pdf").all()
        for pub in publications:
            print(f"Deleting publication {pub.id}")
            pages = Page.query.filter_by(publication_id=pub.id).all()
            for page in pages:
                AdBox.query.filter_by(page_id=page.id).delete()
                db.session.delete(page)
            db.session.delete(pub)
        db.session.commit()
        print("Cleared all existing publications")

    # Upload fresh file through web interface
    session = requests.Session()

    # Login
    login_data = {'password': 'CCCitizen56101!'}
    login_response = session.post('http://localhost:5000/login', data=login_data)
    print(f"Login status: {login_response.status_code}")

    # Upload
    pdf_path = "C:\\Users\\trevo\\newspaper-ad-measurement\\OA-2025-01-01.pdf"
    upload_data = {'publication_type': 'broadsheet'}

    with open(pdf_path, 'rb') as pdf_file:
        files = {'pdf_file': ('OA-2025-01-01.pdf', pdf_file, 'application/pdf')}
        upload_response = session.post('http://localhost:5000/upload',
                                     data=upload_data,
                                     files=files,
                                     timeout=120)

    print(f"Upload response: {upload_response.status_code}")

    # Wait for processing
    time.sleep(15)

    # Get latest publication and check ad coordinates
    with app.app_context():
        latest_pub = Publication.query.filter_by(original_filename="OA-2025-01-01.pdf").order_by(Publication.id.desc()).first()
        if not latest_pub:
            print("ERROR: No publication found")
            return False

        print(f"\nPublication ID: {latest_pub.id}")
        print(f"Processed: {latest_pub.processed}")

        # Get pages with ads
        pages = Page.query.filter_by(publication_id=latest_pub.id).all()
        print(f"Pages: {len(pages)}")

        # Check actual ad box coordinates and sizes for key pages
        print("\n=== CHECKING AD BOX COORDINATES ===")

        for page in [4, 6, 9]:  # Business directory and classified pages
            page_record = Page.query.filter_by(publication_id=latest_pub.id, page_number=page).first()
            if page_record:
                ads = AdBox.query.filter_by(page_id=page_record.id).all()
                print(f"\nPage {page}: {len(ads)} ads")

                # Show first 5 ads with their coordinates and sizes
                for i, ad in enumerate(ads[:5]):
                    print(f"  Ad {i+1}: ({ad.x}, {ad.y}) size: {ad.width}x{ad.height} type: {ad.ad_type}")

                    # Check if ad box is reasonable size (not tiny random spots)
                    if ad.width < 30 or ad.height < 15:
                        print(f"    WARNING: Ad {i+1} is very small ({ad.width}x{ad.height}) - might be random spot")
                    elif ad.width > 200 and ad.height > 50:
                        print(f"    GOOD: Ad {i+1} has reasonable business ad size")

        # Test if business directory ads are on the right pages
        business_ads = []
        classified_ads = []

        for page in pages:
            ads = AdBox.query.filter_by(page_id=page.id).all()
            for ad in ads:
                if ad.ad_type == 'business_directory':
                    business_ads.append((page.page_number, ad))
                elif ad.ad_type == 'classified_ad':
                    classified_ads.append((page.page_number, ad))

        print(f"\n=== AD TYPE DISTRIBUTION ===")
        print(f"Business directory ads: {len(business_ads)}")
        print(f"Classified ads: {len(classified_ads)}")

        # Check if business ads are on expected pages (4, 6)
        business_pages = set(page_num for page_num, ad in business_ads)
        classified_pages = set(page_num for page_num, ad in classified_ads)

        print(f"Business ads found on pages: {sorted(business_pages)}")
        print(f"Classified ads found on pages: {sorted(classified_pages)}")

        # Success criteria
        success = True
        if 4 not in business_pages or 6 not in business_pages:
            print("ERROR: Business directory ads not found on pages 4 and 6")
            success = False

        if 9 not in classified_pages:
            print("ERROR: Classified ads not found on page 9")
            success = False

        total_ads = len(business_ads) + len(classified_ads)
        if total_ads < 15:
            print(f"ERROR: Only {total_ads} ads found, need 15+")
            success = False

        if success:
            print(f"\nSUCCESS: {total_ads} properly detected business ads on correct pages")
            return True
        else:
            print(f"\nFAILURE: Ad detection not working correctly")
            return False

if __name__ == "__main__":
    success = test_fresh_upload()
    if success:
        print("\nNEWSPAPER DOMAIN DETECTOR WORKING IN WEB INTERFACE!")
    else:
        print("\nWEB INTERFACE STILL USING OLD BROKEN DETECTOR")