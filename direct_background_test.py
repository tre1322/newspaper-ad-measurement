#!/usr/bin/env python3
"""
DIRECT BACKGROUND TEST: Call background processing directly on publication 74
"""
from app import app, db, Publication, Page, AdBox, start_background_processing

def direct_background_test():
    """Test background processing directly"""
    print("=== DIRECT BACKGROUND PROCESSING TEST ===")

    with app.app_context():

        # Get publication 73 (most recent)
        publication = Publication.query.get(73)
        if not publication:
            print("ERROR: Publication 73 not found")
            return False

        print(f"Testing publication {publication.id}: {publication.original_filename}")
        print(f"Current status: processed={publication.processed}")

        # Clear existing ads
        pages = Page.query.filter_by(publication_id=publication.id).all()
        for page in pages:
            AdBox.query.filter_by(page_id=page.id).delete()
        db.session.commit()
        print(f"Cleared existing ads from {len(pages)} pages")

        # Reset publication status
        publication.processed = False
        try:
            publication.set_processing_status('uploaded')
        except:
            pass
        db.session.commit()
        print("Reset publication status")

        # Call background processing directly
        print("=== CALLING BACKGROUND PROCESSING DIRECTLY ===")
        try:
            start_background_processing(publication.id)
            print("Background processing call completed")
        except Exception as e:
            print(f"Background processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        # Check results
        print("=== CHECKING RESULTS ===")
        pub = Publication.query.get(publication.id)
        print(f"Processed: {pub.processed}")
        try:
            print(f"Status: {pub.safe_processing_status}")
        except:
            print("Status: unknown")

        total_ads = 0
        for page in pages:
            page_ads = AdBox.query.filter_by(page_id=page.id).all()
            total_ads += len(page_ads)
            if len(page_ads) > 0:
                print(f"  Page {page.page_number}: {len(page_ads)} ads")

        print(f"Total ads: {total_ads}")
        print(f"Ad inches: {pub.total_ad_inches}")

        if total_ads >= 15:
            print("*** SUCCESS: Background processing detects 15+ ads! ***")
            return True
        else:
            print(f"*** FAILURE: Only {total_ads} ads detected ***")
            return False

if __name__ == "__main__":
    success = direct_background_test()
    if success:
        print("\nBACKGROUND PROCESSING WORKS DIRECTLY!")
    else:
        print("\nBACKGROUND PROCESSING STILL FAILING")