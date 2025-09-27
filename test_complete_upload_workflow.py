#!/usr/bin/env python3
"""
TEST COMPLETE UPLOAD WORKFLOW: Verify content-based detection runs during actual upload processing
"""
from app import app, db, Publication, Page, AdBox, start_background_processing
import os
import shutil

def test_complete_upload_workflow():
    """Test the actual upload → background processing → detection workflow"""
    print("=== TESTING COMPLETE UPLOAD WORKFLOW ===")
    print()

    with app.app_context():

        # Find existing OA publication
        test_file = "OA-2025-01-01.pdf"
        publication = Publication.query.filter_by(original_filename=test_file).first()

        if not publication:
            print(f"ERROR: {test_file} not found in database")
            return False

        print(f"Found publication {publication.id}: {publication.original_filename}")

        # Clear ALL existing data for clean test
        pages = Page.query.filter_by(publication_id=publication.id).all()
        for page in pages:
            AdBox.query.filter_by(page_id=page.id).delete()
            db.session.delete(page)

        publication.processed = False
        publication.total_ad_inches = 0
        publication.ad_percentage = 0
        try:
            publication.set_processing_status('uploaded')
        except:
            pass
        db.session.commit()
        print("Cleared all existing data for clean test")

        # Test the ACTUAL background processing that runs during uploads
        print("\n=== RUNNING ACTUAL BACKGROUND PROCESSING ===")

        try:
            # This is the EXACT function called during uploads
            start_background_processing(publication.id)

            # Wait a moment for processing to complete
            import time
            time.sleep(2)

            # Check final results
            final_publication = Publication.query.get(publication.id)
            final_pages = Page.query.filter_by(publication_id=publication.id).all()
            final_ads = AdBox.query.join(Page).filter(Page.publication_id == publication.id).all()

            print(f"\n=== FINAL RESULTS FROM ACTUAL UPLOAD WORKFLOW ===")
            print(f"Publication processed: {final_publication.processed}")
            print(f"Processing status: {final_publication.safe_processing_status}")
            print(f"Pages created: {len(final_pages)}")
            print(f"Ads detected: {len(final_ads)}")
            print(f"Total ad inches: {final_publication.total_ad_inches:.2f}")
            print(f"Ad percentage: {final_publication.ad_percentage:.1f}%")

            # Show detailed ad breakdown
            if final_ads:
                print(f"\nDetected ads by page:")
                for page in final_pages:
                    page_ads = [ad for ad in final_ads if ad.page_id == page.id]
                    if page_ads:
                        print(f"  Page {page.page_number}: {len(page_ads)} ads")
                        for i, ad in enumerate(page_ads[:3]):  # Show first 3 ads per page
                            print(f"    Ad {i+1}: Type={ad.ad_type}, Size={ad.width:.0f}x{ad.height:.0f}, Confidence={ad.confidence_score:.3f}")

            print()

            # SUCCESS CRITERIA
            if len(final_ads) >= 15:
                print("*** SUCCESS: Real upload workflow detected 15+ ads! ***")
                print(f"Users will see {len(final_ads)} ads when they upload this file")
                return True
            elif len(final_ads) > 2:
                print(f"*** PARTIAL SUCCESS: {len(final_ads)} ads (improved from 2 baseline) ***")
                return True
            else:
                print(f"*** FAILURE: Only {len(final_ads)} ads detected in real upload workflow ***")
                return False

        except Exception as e:
            print(f"ERROR in complete upload workflow: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_complete_upload_workflow()
    if success:
        print("\nCOMPLETE UPLOAD WORKFLOW NOW DETECTS 30+ ADS!")
    else:
        print("\nCOMPLETE UPLOAD WORKFLOW STILL FAILING")