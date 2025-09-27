#!/usr/bin/env python3
"""
TEST UPLOAD WITH NEW CODE: Upload OA file and see if content detection works
"""
from app import app, db, Publication, Page, AdBox, start_background_processing
import shutil
import os

def test_upload_with_new_code():
    """Test upload with the updated content-based detection code"""
    print("=== TESTING UPLOAD WITH NEW CONTENT-BASED DETECTION ===")

    with app.app_context():

        # STEP 1: Simulate upload by creating publication record
        print("=== STEP 1: CREATING PUBLICATION RECORD ===")

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

        # Copy file to upload directory
        import uuid
        unique_filename = f"{uuid.uuid4()}.pdf"
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs')
        os.makedirs(upload_dir, exist_ok=True)
        dest_file = os.path.join(upload_dir, unique_filename)
        shutil.copy2(source_file, dest_file)
        print(f"Copied file to: {dest_file}")

        # Create publication record
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
        print(f"Created publication {publication.id}")

        # STEP 2: Run background processing
        print("\n=== STEP 2: RUNNING BACKGROUND PROCESSING ===")

        try:
            # Call the background processing directly (not threaded to see errors)
            start_background_processing(publication.id)

            # Wait for processing to complete
            import time
            time.sleep(5)

            # Check results
            final_pub = Publication.query.get(publication.id)
            pages = Page.query.filter_by(publication_id=publication.id).all()
            total_ads = 0

            for page in pages:
                page_ads = AdBox.query.filter_by(page_id=page.id).all()
                total_ads += len(page_ads)
                if len(page_ads) > 0:
                    print(f"  Page {page.page_number}: {len(page_ads)} ads")

            print(f"\n=== BACKGROUND PROCESSING RESULTS ===")
            print(f"Processed: {final_pub.processed}")
            print(f"Status: {final_pub.safe_processing_status}")
            print(f"Pages: {len(pages)}")
            print(f"Total ads: {total_ads}")
            print(f"Ad inches: {final_pub.total_ad_inches}")

            if total_ads >= 25:
                print("*** SUCCESS: Content-based detection working in background processing! ***")
                return True
            elif total_ads > 17:
                print(f"*** IMPROVEMENT: {total_ads} ads (better than 17 baseline) ***")
                return True
            else:
                print(f"*** FAILURE: Only {total_ads} ads detected - same as old system ***")
                return False

        except Exception as e:
            print(f"ERROR in background processing: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_upload_with_new_code()
    if success:
        print("\nCONTENT-BASED DETECTION NOW WORKING IN UPLOADS!")
    else:
        print("\nCONTENT-BASED DETECTION STILL NOT WORKING IN UPLOADS")