#!/usr/bin/env python3
"""
EMERGENCY TEST: Simulate the actual upload process and verify detection
"""

import os
import sys
from app import app, db, Publication, Page, AdBox
from app import start_background_processing, HybridDetectionPipeline

def test_upload_detection():
    """Test the actual upload detection process"""

    with app.app_context():
        print("=" * 80)
        print("EMERGENCY UPLOAD DETECTION TEST")
        print("=" * 80)

        # Get the most recent publication (simulating an upload)
        publication = Publication.query.order_by(Publication.id.desc()).first()
        if not publication:
            print("ERROR: No publications found - need to test with actual upload")
            return False

        print(f"Testing with publication: {publication.original_filename}")
        print(f"Publication ID: {publication.id}")
        print(f"Pages: {publication.total_pages}")

        # Check current AdBox count BEFORE any new detection
        before_count = AdBox.query.join(Page).filter(
            Page.publication_id == publication.id
        ).count()
        print(f"AdBoxes BEFORE new detection: {before_count}")

        # Test the hybrid detection directly (what should run during upload)
        print("\nTesting HybridDetectionPipeline directly...")
        try:
            hybrid_pipeline = HybridDetectionPipeline()
            result = hybrid_pipeline.detect_ads_hybrid(publication.id, mode='auto')

            print(f"Direct hybrid test result: {result.get('success', False)}")
            print(f"Total detections: {result.get('total_detections', 0)}")
            print(f"Logo detections: {result.get('logo_detections', 0)}")
            print(f"Pages processed: {result.get('pages_processed', 0)}")

        except Exception as e:
            print(f"CRITICAL ERROR in hybrid detection: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Check AdBox count AFTER detection
        after_count = AdBox.query.join(Page).filter(
            Page.publication_id == publication.id
        ).count()
        print(f"AdBoxes AFTER new detection: {after_count}")
        print(f"NEW AdBoxes created: {after_count - before_count}")

        # Check if any recent AdBoxes were created
        recent_ads = AdBox.query.join(Page).filter(
            Page.publication_id == publication.id
        ).order_by(AdBox.id.desc()).limit(5).all()

        print("\nRecent AdBoxes for this publication:")
        for ad in recent_ads:
            print(f"  AdBox {ad.id}: {ad.ad_type}, auto={ad.detected_automatically}, conf={ad.confidence_score}")

        # Test if the pages exist and have images
        print(f"\nChecking page images...")
        pages = Page.query.filter_by(publication_id=publication.id).all()
        print(f"Pages in database: {len(pages)}")

        pages_with_images = 0
        for page in pages[:3]:  # Check first 3 pages
            image_filename = f"{publication.filename}_page_{page.page_number}.png"
            image_path = os.path.join('static', 'uploads', 'pages', image_filename)
            if os.path.exists(image_path):
                pages_with_images += 1
                print(f"  Page {page.page_number}: Image exists")
            else:
                print(f"  Page {page.page_number}: Image MISSING at {image_path}")

        print(f"Pages with images: {pages_with_images}/{len(pages[:3])}")

        # Test background processing function call
        print(f"\nTesting background processing trigger...")
        try:
            # This is what gets called during upload
            start_background_processing(publication.id)
            print("Background processing trigger completed")
        except Exception as e:
            print(f"ERROR in background processing: {e}")

        # Final check
        final_count = AdBox.query.join(Page).filter(
            Page.publication_id == publication.id
        ).count()
        print(f"\nFINAL AdBox count: {final_count}")

        if final_count > before_count:
            print("SUCCESS: SUCCESS: New ads were detected and saved!")
            return True
        else:
            print("ERROR: PROBLEM: No new ads detected")
            return False

def check_detection_requirements():
    """Check if all detection requirements are met"""
    print("\n" + "=" * 80)
    print("CHECKING DETECTION REQUIREMENTS")
    print("=" * 80)

    with app.app_context():
        # Check 1: Publications exist
        pub_count = Publication.query.count()
        print(f"Publications in database: {pub_count}")

        # Check 2: Pages exist
        page_count = Page.query.count()
        print(f"Pages in database: {page_count}")

        # Check 3: Business logos exist
        from app import BusinessLogo
        logo_count = BusinessLogo.query.filter_by(is_active=True).count()
        print(f"Active business logos: {logo_count}")

        # Check 4: Page images exist
        pages = Page.query.limit(5).all()
        image_count = 0
        for page in pages:
            pub = Publication.query.get(page.publication_id)
            image_filename = f"{pub.filename}_page_{page.page_number}.png"
            image_path = os.path.join('static', 'uploads', 'pages', image_filename)
            if os.path.exists(image_path):
                image_count += 1

        print(f"Page images available: {image_count}/{len(pages)}")

        # Check 5: Hybrid system imports
        try:
            pipeline = HybridDetectionPipeline()
            print("SUCCESS: HybridDetectionPipeline imports successfully")
        except Exception as e:
            print(f"ERROR: HybridDetectionPipeline import error: {e}")

        return pub_count > 0 and page_count > 0

def main():
    print("EMERGENCY UPLOAD DETECTION VERIFICATION")
    print("=" * 80)

    # Check requirements first
    if not check_detection_requirements():
        print("ERROR: CRITICAL: Missing basic requirements for detection")
        return

    # Test the detection
    success = test_upload_detection()

    print("\n" + "=" * 80)
    print("EMERGENCY TEST SUMMARY")
    print("=" * 80)

    if success:
        print("SUCCESS: DETECTION IS WORKING!")
        print("- Hybrid detection pipeline functional")
        print("- Ads being detected and saved")
        print("- Upload process should work correctly")
    else:
        print("ERROR: DETECTION IS BROKEN!")
        print("- Hybrid detection not creating new AdBoxes")
        print("- Need to rollback or fix immediately")
        print("\nRECOMMEND:")
        print("1. Check for errors in hybrid detection")
        print("2. Verify logo recognition is finding ads")
        print("3. Consider temporary rollback to old system")

if __name__ == "__main__":
    main()