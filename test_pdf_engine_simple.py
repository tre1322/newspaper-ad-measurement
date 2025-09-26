#!/usr/bin/env python3
"""
Simple test to verify PDF detection engine works without Unicode issues
"""

import os
import sys
from app import app, db, Publication, Page, AdBox

def test_pdf_engine_simple():
    """Test PDF engine with minimal logging to avoid Unicode issues"""

    with app.app_context():
        print("=" * 60)
        print("SIMPLE PDF ENGINE TEST")
        print("=" * 60)

        # Get the most recent publication
        publication = Publication.query.order_by(Publication.id.desc()).first()
        if not publication:
            print("ERROR: No publications found")
            return

        print(f"Testing publication {publication.id}: {publication.original_filename}")

        # Check current AdBox count for this publication BEFORE detection
        before_count = db.session.query(AdBox).join(Page).filter(
            Page.publication_id == publication.id
        ).count()
        print(f"AdBox records BEFORE detection: {before_count}")

        # Test the PDF detection engine directly
        try:
            from app import PDFAdDetectionEngine

            # Call the detection engine and catch any errors
            print("Calling PDFAdDetectionEngine.detect_ads_from_pdf...")
            result = PDFAdDetectionEngine.detect_ads_from_pdf(publication.id)
            print(f"Detection result: {result}")

            # Check AdBox count AFTER detection
            after_count = db.session.query(AdBox).join(Page).filter(
                Page.publication_id == publication.id
            ).count()
            print(f"AdBox records AFTER detection: {after_count}")
            print(f"NEW AdBox records created: {after_count - before_count}")

            if result and result.get('success'):
                print(f"SUCCESS: Detection reported {result.get('detections', 0)} ads")
                if after_count > before_count:
                    print("SUCCESS: AdBox records were actually created in database")
                else:
                    print("PROBLEM: No new AdBox records despite successful detection")
            else:
                print(f"FAILED: Detection failed with error: {result.get('error', 'Unknown') if result else 'No result'}")

        except Exception as e:
            print(f"EXCEPTION in PDF engine: {e}")
            # Show the core error without Unicode
            import traceback
            print("Exception details:")
            traceback.print_exc()

        print("=" * 60)

if __name__ == "__main__":
    test_pdf_engine_simple()