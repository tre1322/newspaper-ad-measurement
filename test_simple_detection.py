#!/usr/bin/env python3
"""
Test the simple bordered ad detection system
"""

import os
from app import app, db, Publication, Page, AdBox, SimpleAdDetector

def test_simple_detection():
    """Test simple bordered ad detection"""

    with app.app_context():
        print("TESTING SIMPLE BORDERED AD DETECTION")
        print("=" * 50)

        # Get test publication
        publication = Publication.query.order_by(Publication.id.desc()).first()

        print(f"Testing publication: {publication.original_filename}")

        # Count ads before
        before_count = AdBox.query.join(Page).filter(
            Page.publication_id == publication.id
        ).count()
        print(f"AdBoxes before simple detection: {before_count}")

        # Test simple detection
        try:
            result = SimpleAdDetector.detect_bordered_ads(publication.id)

            print(f"Simple detection result: {result}")

            if result and result.get('success'):
                print(f"SUCCESS: {result.get('detections', 0)} bordered ads detected")
                print(f"Pages processed: {result.get('pages_processed', 0)}")
            else:
                print(f"Failed: {result.get('error', 'Unknown') if result else 'No result'}")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

        # Count ads after
        after_count = AdBox.query.join(Page).filter(
            Page.publication_id == publication.id
        ).count()
        print(f"AdBoxes after simple detection: {after_count}")
        print(f"New ads created: {after_count - before_count}")

        # Show recent bordered ads
        recent_ads = AdBox.query.join(Page).filter(
            Page.publication_id == publication.id,
            AdBox.ad_type == 'bordered_ad'
        ).order_by(AdBox.id.desc()).limit(10).all()

        print(f"\nRecent bordered ads:")
        for ad in recent_ads:
            print(f"  AdBox {ad.id}: {ad.width}x{ad.height} pixels, confidence={ad.confidence_score:.2f}")

        return after_count > before_count

if __name__ == "__main__":
    success = test_simple_detection()
    if success:
        print("\nSUCCESS: Simple detection created new ads!")
    else:
        print("\nERROR: Simple detection found no new ads!")