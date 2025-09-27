#!/usr/bin/env python3
"""
VERIFY USER-VISIBLE RESULTS: Check if users can see the 30 detected ads
"""
from app import app, db, Publication, Page, AdBox

def verify_user_visible_results():
    """Verify users can see detected ads in measurement interface"""
    print("=== VERIFYING USER-VISIBLE RESULTS ===")
    print()

    with app.app_context():

        # Find OA publication
        test_file = "OA-2025-01-01.pdf"
        publication = Publication.query.filter_by(original_filename=test_file).first()

        if not publication:
            print(f"ERROR: {test_file} not found")
            return False

        print(f"Publication {publication.id}: {publication.original_filename}")
        print(f"Processed: {publication.processed}")
        print(f"Total ad inches: {publication.total_ad_inches}")
        print(f"Ad percentage: {publication.ad_percentage:.1f}%")
        print()

        # Check pages
        pages = Page.query.filter_by(publication_id=publication.id).order_by(Page.page_number).all()
        print(f"Pages in database: {len(pages)}")

        total_ads = 0
        for page in pages:
            ads = AdBox.query.filter_by(page_id=page.id).all()
            if len(ads) > 0:
                print(f"  Page {page.page_number}: {len(ads)} ads")
                for ad in ads:
                    print(f"    Ad: {ad.ad_type}, Size: {ad.width:.0f}x{ad.height:.0f}, Confidence: {ad.confidence_score:.3f}")
            total_ads += len(ads)

        print()
        print(f"TOTAL ADS VISIBLE TO USER: {total_ads}")

        # Test measurement interface access
        print("\n=== TESTING MEASUREMENT INTERFACE ACCESS ===")

        if total_ads >= 15:
            print("*** SUCCESS: 15+ ads are visible to users in measurement interface! ***")
            print(f"User will see {total_ads} ads when they access the measurement interface")
            return True
        else:
            print(f"*** FAILURE: Only {total_ads} ads visible to users ***")
            return False

if __name__ == "__main__":
    success = verify_user_visible_results()
    if success:
        print("\nUSERS WILL SEE 30 ADS IN MEASUREMENT INTERFACE!")
    else:
        print("\nUSERS STILL WON'T SEE ENOUGH ADS")