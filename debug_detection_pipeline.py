#!/usr/bin/env python3
"""
Standalone test to debug the AI detection pipeline
Tests each component separately to find where the failure occurs
"""

import os
import sys
import fitz
from sqlalchemy import create_engine, text
from app import app, db, Publication, Page, AdBox, PDFAdDetectionEngine
from pdf_structure_analyzer import PDFStructureAdDetector

def debug_detection_pipeline():
    """Debug the complete detection pipeline step by step"""

    with app.app_context():
        print("=" * 80)
        print("DETECTION PIPELINE DEBUG - STEP BY STEP ANALYSIS")
        print("=" * 80)

        # Step 1: Find the most recent publication
        print("\nSTEP 1: Finding recent publications...")
        publications = Publication.query.order_by(Publication.id.desc()).limit(5).all()

        if not publications:
            print("ERROR: No publications found in database!")
            return

        for i, pub in enumerate(publications):
            print(f"   {i+1}. Publication {pub.id}: {pub.original_filename} ({pub.publication_type}) - {pub.total_pages} pages")

        # Use the most recent publication
        publication = publications[0]
        print(f"\nUsing publication {publication.id}: {publication.original_filename}")

        # Step 2: Check file existence
        print(f"\nSTEP 2: Checking file system...")
        pdf_path = os.path.join('static', 'uploads', 'pdfs', publication.filename)
        print(f"   PDF path: {pdf_path}")
        print(f"   PDF exists: {os.path.exists(pdf_path)}")

        if not os.path.exists(pdf_path):
            print("ERROR: PDF file not found!")
            return

        # Step 3: Check PDF can be opened
        print(f"\nSTEP 3: Testing PDF access...")
        try:
            doc = fitz.open(pdf_path)
            print(f"   PDF opened successfully")
            print(f"   Pages in PDF: {len(doc)}")
            doc.close()
        except Exception as e:
            print(f"   ERROR opening PDF: {e}")
            return

        # Step 4: Check database pages
        print(f"\nSTEP 4: Checking database pages...")
        pages = Page.query.filter_by(publication_id=publication.id).all()
        print(f"   Pages in database: {len(pages)}")

        for page in pages[:3]:  # Show first 3 pages
            print(f"      Page {page.page_number}: {page.width_pixels}x{page.height_pixels} pixels")

        if not pages:
            print("ERROR: No pages found in database!")
            return

        # Step 5: Test PDF structure detection on first page
        print(f"\nSTEP 5: Testing PDF structure detection on page 1...")
        try:
            structure_ads = PDFStructureAdDetector.detect_ads_from_pdf_structure(
                pdf_path, 1, publication.publication_type
            )
            print(f"   Structure detector found: {len(structure_ads)} ads")

            if structure_ads:
                for i, ad in enumerate(structure_ads):
                    print(f"      Ad {i+1}: {ad}")
            else:
                print("   No ads found by structure detector - investigating...")

                # Deep dive into structure
                doc = fitz.open(pdf_path)
                page = doc[0]

                print(f"      Page dimensions: {page.rect.width} x {page.rect.height}")

                # Check text blocks
                text_dict = page.get_text("dict")
                text_blocks = text_dict.get("blocks", [])
                print(f"      Raw text blocks: {len(text_blocks)}")

                # Check images
                images = page.get_images()
                print(f"      Images: {len(images)}")

                # Check drawings
                drawings = page.get_drawings()
                print(f"      Drawings: {len(drawings)}")

                doc.close()

        except Exception as e:
            print(f"   ERROR in structure detection: {e}")
            import traceback
            traceback.print_exc()

        # Step 6: Test complete PDF detection engine
        print(f"\nSTEP 6: Testing complete PDF detection engine...")
        try:
            result = PDFAdDetectionEngine.detect_ads_from_pdf(publication.id)
            print(f"   PDF Engine result: {result}")

            if result.get('success'):
                print(f"   PDF Engine successful: {result['detections']} ads detected")
            else:
                print(f"   PDF Engine failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"   ERROR in PDF engine: {e}")
            import traceback
            traceback.print_exc()

        # Step 7: Check current AdBox records
        print(f"\nSTEP 7: Checking current AdBox records...")
        current_adboxes = db.session.query(AdBox).join(Page).filter(
            Page.publication_id == publication.id
        ).all()

        print(f"   Current AdBox records for this publication: {len(current_adboxes)}")

        for i, adbox in enumerate(current_adboxes[:5]):  # Show first 5
            print(f"      AdBox {i+1}: Page {adbox.page_id}, pos=({adbox.x:.1f},{adbox.y:.1f}), "
                  f"size={adbox.width:.1f}x{adbox.height:.1f}, type={adbox.ad_type}, "
                  f"auto={adbox.detected_automatically}, confidence={adbox.confidence_score}")

        # Step 8: Test one manual detection
        print(f"\nSTEP 8: Running manual detection test...")
        test_page = pages[0]

        # Check if page image exists
        image_filename = f"{publication.filename}_page_{test_page.page_number}.png"
        image_path = os.path.join('static', 'uploads', 'pages', image_filename)
        print(f"   Page image path: {image_path}")
        print(f"   Page image exists: {os.path.exists(image_path)}")

        if os.path.exists(image_path):
            print(f"   Page image found")

            # Try to load the image
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                print(f"   Image loaded: {img.shape}")
            else:
                print(f"   Failed to load image")
        else:
            print(f"   Page image not found")

        print(f"\nDETECTION PIPELINE DEBUG COMPLETE")
        print("=" * 80)

if __name__ == "__main__":
    debug_detection_pipeline()