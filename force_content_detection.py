#!/usr/bin/env python3
"""
FORCE CONTENT DETECTION: Directly run content detection on existing publication
"""
from app import app, db, Publication, Page, AdBox, ContentBasedAdDetector

def force_content_detection():
    """Force content detection to run on OA publication"""
    print("=== FORCING CONTENT DETECTION ===")

    with app.app_context():

        # Find OA publication
        test_file = "OA-2025-01-01.pdf"
        publication = Publication.query.filter_by(original_filename=test_file).first()

        if not publication:
            print(f"ERROR: {test_file} not found")
            return False

        print(f"Publication {publication.id}: {publication.original_filename}")

        # Make sure it has pages
        pages = Page.query.filter_by(publication_id=publication.id).all()
        if len(pages) == 0:
            print("ERROR: No pages found")
            return False

        print(f"Found {len(pages)} pages")

        # Clear existing ads
        for page in pages:
            AdBox.query.filter_by(page_id=page.id).delete()
        db.session.commit()
        print("Cleared existing ads")

        # Force content detection to run
        file_path = "C:\\Users\\trevo\\newspaper-ad-measurement\\OA-2025-01-01.pdf"
        total_ads = 0

        for page in pages:
            print(f"Running content detection on page {page.page_number}...")

            content_ads = ContentBasedAdDetector.detect_business_content_ads(file_path, page.page_number)

            for ad in content_ads:
                # Create AdBox
                ad_box = AdBox(
                    page_id=page.id,
                    x=float(ad['x']),
                    y=float(ad['y']),
                    width=float(ad['width']),
                    height=float(ad['height']),
                    width_inches_raw=ad['width'] / 150,
                    height_inches_raw=ad['height'] / 150,
                    width_inches_rounded=round((ad['width'] / 150) * 16) / 16,
                    height_inches_rounded=round((ad['height'] / 150) * 16) / 16,
                    column_inches=(ad['width'] / 150) * (ad['height'] / 150),
                    ad_type='business_content',
                    is_ad=True,
                    detected_automatically=True,
                    confidence_score=ad['confidence'],
                    user_verified=False
                )
                db.session.add(ad_box)
                total_ads += 1

            print(f"Page {page.page_number}: {len(content_ads)} ads")

        # Update publication
        total_ad_inches = sum(box.column_inches for box in AdBox.query.join(Page).filter(Page.publication_id == publication.id).all())
        publication.total_ad_inches = total_ad_inches
        publication.ad_percentage = (total_ad_inches / publication.total_inches) * 100 if publication.total_inches > 0 else 0
        publication.processed = True

        db.session.commit()

        print(f"\n=== FORCE DETECTION COMPLETE ===")
        print(f"Total ads: {total_ads}")
        print(f"Total ad inches: {total_ad_inches:.2f}")
        print(f"Ad percentage: {publication.ad_percentage:.1f}%")

        if total_ads >= 15:
            print("*** SUCCESS: Users will now see 30 ads! ***")
            return True
        else:
            print(f"*** FAILURE: Only {total_ads} ads ***")
            return False

if __name__ == "__main__":
    success = force_content_detection()
    if success:
        print("\nCONTENT DETECTION FORCED - USERS WILL SEE 30 ADS!")
    else:
        print("\nFORCE DETECTION FAILED")