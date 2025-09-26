#!/usr/bin/env python3
"""
Test the rollback PDF detection system
"""

import os
from app import app, db, Publication, Page, AdBox, PDFAdDetectionEngine

def test_rollback_detection():
    """Test the rollback PDF detection"""

    with app.app_context():
        print("TESTING ROLLBACK PDF DETECTION")
        print("=" * 50)

        # Get test publication
        publication = Publication.query.order_by(Publication.id.desc()).first()

        # Count ads before
        before_count = AdBox.query.join(Page).filter(
            Page.publication_id == publication.id
        ).count()
        print(f"AdBoxes before rollback detection: {before_count}")

        # Test PDF detection directly
        try:
            result = PDFAdDetectionEngine.detect_ads_from_pdf(publication.id)

            print(f"PDF detection result: {result}")

            if result and result.get('success'):
                print(f"SUCCESS: {result.get('detections', 0)} ads detected")
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
        print(f"AdBoxes after rollback detection: {after_count}")
        print(f"New ads created: {after_count - before_count}")

        return after_count > before_count

if __name__ == "__main__":
    success = test_rollback_detection()
    if success:
        print("\nSUCCESS: Rollback detection is working!")
    else:
        print("\nERROR: Rollback detection also not working!")