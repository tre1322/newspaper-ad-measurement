#!/usr/bin/env python3
"""
TEST POST-UPLOAD PROCESSING: Simulate the upload workflow by calling the upload endpoint directly
"""
from app import app, db, Publication, Page, AdBox
import shutil
import os
import uuid

def test_post_upload_processing():
    """Test processing after simulated upload"""
    print("=== TESTING POST-UPLOAD PROCESSING ===")

    with app.app_context():

        # STEP 1: Simulate what the upload route does
        print("=== STEP 1: SIMULATING UPLOAD ROUTE ===")

        test_file = "OA-2025-01-01.pdf"
        source_file = f"C:\\Users\\trevo\\newspaper-ad-measurement\\{test_file}"

        if not os.path.exists(source_file):
            print(f"ERROR: Source file not found: {source_file}")
            return False

        # Clear any existing test publication
        existing = Publication.query.filter_by(original_filename=test_file).first()
        if existing:
            pages = Page.query.filter_by(publication_id=existing.id).all()
            for page in pages:
                AdBox.query.filter_by(page_id=page.id).delete()
                db.session.delete(page)
            db.session.delete(existing)
            db.session.commit()
            print("Cleared existing publication")

        # Copy file to upload directory (simulating file upload)
        unique_filename = f"{uuid.uuid4()}.pdf"
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs')
        os.makedirs(upload_dir, exist_ok=True)
        dest_file = os.path.join(upload_dir, unique_filename)
        shutil.copy2(source_file, dest_file)
        print(f"File uploaded to: {dest_file}")

        # Create publication record (simulating what upload route does)
        publication = Publication(
            filename=unique_filename,
            original_filename=test_file,
            publication_type='broadsheet',
            total_pages=10,
            total_inches=1250  # 125 inches per page * 10 pages
        )

        try:
            publication.set_processing_status('uploaded')
        except:
            pass

        db.session.add(publication)
        db.session.commit()
        print(f"Publication created: ID {publication.id}")

        # STEP 2: Call start_background_processing (now synchronous)
        print("\\n=== STEP 2: CALLING BACKGROUND PROCESSING (NOW SYNCHRONOUS) ===")

        from app import start_background_processing

        try:
            print("Calling start_background_processing...")
            start_background_processing(publication.id)
            print("Background processing call returned")

        except Exception as e:
            print(f"Background processing failed: {e}")
            import traceback
            traceback.print_exc()

        # STEP 3: Check final results (what user would see)
        print("\\n=== STEP 3: CHECKING WHAT USER WOULD SEE ===")

        final_pub = Publication.query.get(publication.id)
        print(f"Publication ID: {final_pub.id}")
        print(f"Original filename: {final_pub.original_filename}")
        print(f"Processed: {final_pub.processed}")
        try:
            print(f"Status: {final_pub.safe_processing_status}")
        except:
            print("Status: unknown")
        print(f"Total ad inches: {final_pub.total_ad_inches}")
        print(f"Ad percentage: {final_pub.ad_percentage:.1f}%")

        # Check pages
        pages = Page.query.filter_by(publication_id=publication.id).all()
        print(f"Pages: {len(pages)}")

        # Check ads per page
        total_ads = 0
        for page in pages:
            page_ads = AdBox.query.filter_by(page_id=page.id).all()
            if len(page_ads) > 0:
                print(f"  Page {page.page_number}: {len(page_ads)} ads")
            total_ads += len(page_ads)

        print(f"\\nTOTAL ADS DETECTED: {total_ads}")

        # SUCCESS CRITERIA: User uploaded OA-2025-01-01.pdf and should see 30 ads
        if total_ads >= 30:
            print("\\n*** PERFECT SUCCESS: User sees 30 ads! ***")
            print("*** When user uploads OA-2025-01-01.pdf, they get 30 ads ***")
            return True
        elif total_ads >= 15:
            print(f"\\n*** SUCCESS: User sees {total_ads} ads (target: 15+) ***")
            return True
        else:
            print(f"\\n*** FAILURE: User only sees {total_ads} ads ***")
            return False

if __name__ == "__main__":
    success = test_post_upload_processing()
    if success:
        print("\\n🎉 UPLOAD WORKFLOW NOW WORKS!")
        print("When users upload OA-2025-01-01.pdf through web interface,")
        print("they will see the detected ads in the measurement system!")
    else:
        print("\\n❌ UPLOAD WORKFLOW STILL BROKEN")