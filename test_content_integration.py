#!/usr/bin/env python3
"""
DIRECT INTEGRATION TEST: Run content detection in the exact upload context
"""
from app import app, db, Publication, Page, AdBox, ContentBasedAdDetector, PUBLICATION_CONFIGS
import os
import fitz

def test_content_integration():
    """Test content detection in upload context"""
    print("=== TESTING CONTENT DETECTION INTEGRATION ===")
    print()

    with app.app_context():

        # Find OA publication
        test_file = "OA-2025-01-01.pdf"
        publication = Publication.query.filter_by(original_filename=test_file).first()

        if not publication:
            print(f"ERROR: {test_file} not found")
            return False

        print(f"Testing publication {publication.id}: {publication.original_filename}")

        # Clear existing data
        pages = Page.query.filter_by(publication_id=publication.id).all()
        for page in pages:
            AdBox.query.filter_by(page_id=page.id).delete()
            db.session.delete(page)
        db.session.commit()

        # Simulate EXACT upload processing with content detection
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs', publication.filename)
        print(f"PDF path: {file_path}")

        if not os.path.exists(file_path):
            print(f"ERROR: PDF file not found at {file_path}")
            return False

        try:
            pdf_doc = fitz.open(file_path)
            config = PUBLICATION_CONFIGS[publication.publication_type]
            total_detected_boxes = 0

            print(f"Processing {pdf_doc.page_count} pages...")

            for page_num in range(pdf_doc.page_count):
                page = pdf_doc.load_page(page_num)

                # Convert page to image (simulate upload processing)
                mat = fitz.Matrix(2.0, 2.0)
                pix = page.get_pixmap(matrix=mat)

                # Create page record (simulate upload processing)
                page_record = Page(
                    publication_id=publication.id,
                    page_number=page_num + 1,
                    width_pixels=pix.width,
                    height_pixels=pix.height,
                    total_page_inches=config['total_inches_per_page']
                )
                page_record.pixels_per_inch = pix.width / config['width_units']

                db.session.add(page_record)
                db.session.flush()  # Get page ID

                # CONTENT-BASED DETECTION (the new system)
                print(f"Running content detection on page {page_num + 1}...")
                content_ads = ContentBasedAdDetector.detect_business_content_ads(file_path, page_num + 1)

                print(f"Content detection found {len(content_ads)} ads on page {page_num + 1}")

                # Create AdBox records (simulate upload processing)
                for ad in content_ads:
                    # Calculate measurements
                    dpi = page_record.pixels_per_inch or 150
                    width_inches_raw = ad['width'] / dpi
                    height_inches_raw = ad['height'] / dpi
                    column_inches = width_inches_raw * height_inches_raw

                    # Create AdBox
                    ad_box = AdBox(
                        page_id=page_record.id,
                        x=float(ad['x']),
                        y=float(ad['y']),
                        width=float(ad['width']),
                        height=float(ad['height']),
                        width_inches_raw=width_inches_raw,
                        height_inches_raw=height_inches_raw,
                        width_inches_rounded=round(width_inches_raw * 16) / 16,
                        height_inches_rounded=round(height_inches_raw * 16) / 16,
                        column_inches=column_inches,
                        ad_type='business_content',
                        is_ad=True,
                        detected_automatically=True,
                        confidence_score=ad['confidence'],
                        user_verified=False
                    )

                    db.session.add(ad_box)
                    total_detected_boxes += 1

            pdf_doc.close()

            # Update publication totals (simulate upload processing)
            total_ad_inches = sum(box.column_inches for box in AdBox.query.join(Page).filter(Page.publication_id == publication.id).all())
            publication.total_ad_inches = total_ad_inches
            publication.ad_percentage = (total_ad_inches / publication.total_inches) * 100 if publication.total_inches > 0 else 0
            publication.processed = True

            db.session.commit()

            print(f"\n=== INTEGRATION TEST RESULTS ===")
            print(f"Total ads detected: {total_detected_boxes}")
            print(f"Total ad inches: {total_ad_inches:.2f}")
            print(f"Ad percentage: {publication.ad_percentage:.1f}%")

            if total_detected_boxes >= 15:
                print("*** SUCCESS: Content-based detection integrated successfully! ***")
                return True
            else:
                print(f"*** FAILURE: Only {total_detected_boxes} ads detected ***")
                return False

        except Exception as e:
            print(f"ERROR during integration test: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_content_integration()
    if success:
        print("\nCONTENT-BASED DETECTION SUCCESSFULLY INTEGRATED!")
    else:
        print("\nINTEGRATION FAILED")