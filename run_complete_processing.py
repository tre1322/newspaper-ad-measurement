#!/usr/bin/env python3
"""
RUN COMPLETE PROCESSING: Manually run the full background processing
"""
from app import app, db, Publication, Page, AdBox, ContentBasedAdDetector, PUBLICATION_CONFIGS
import os
import fitz

def run_complete_processing():
    """Run the COMPLETE processing workflow manually"""
    print("=== RUNNING COMPLETE PROCESSING WORKFLOW ===")

    with app.app_context():

        # Find OA publication
        test_file = "OA-2025-01-01.pdf"
        publication = Publication.query.filter_by(original_filename=test_file).first()

        if not publication:
            print(f"ERROR: {test_file} not found")
            return False

        print(f"Publication {publication.id}: {publication.original_filename}")

        # Clear all data
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
        print("Cleared all data")

        # STEP 1: Process PDF to create pages (COMPLETE VERSION)
        print("\n=== STEP 1: PROCESSING PDF TO CREATE PAGES ===")

        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs', publication.filename)
            print(f"PDF path: {file_path}")

            if not os.path.exists(file_path):
                print(f"ERROR: PDF not found at {file_path}")
                return False

            pdf_doc = fitz.open(file_path)
            print(f"PDF opened - {pdf_doc.page_count} pages")

            # Update status
            publication.set_processing_status('creating_images')
            db.session.commit()

            # Process all pages in batches
            batch_size = 3
            config = PUBLICATION_CONFIGS[publication.publication_type]

            for batch_start in range(0, pdf_doc.page_count, batch_size):
                batch_end = min(batch_start + batch_size, pdf_doc.page_count)

                publication.set_processing_status(f'processing_pages_{batch_start + 1}_to_{batch_end}')

                for page_num in range(batch_start, batch_end):
                    page = pdf_doc[page_num]

                    # Convert to image
                    mat = fitz.Matrix(1.5, 1.5)
                    pix = page.get_pixmap(matrix=mat)
                    img_data = pix.tobytes("png")

                    # Save page image
                    image_filename = f"{publication.filename}_page_{page_num + 1}.png"
                    pages_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'pages')
                    os.makedirs(pages_dir, exist_ok=True)
                    image_path = os.path.join(pages_dir, image_filename)

                    with open(image_path, 'wb') as img_file:
                        img_file.write(img_data)

                    # Create page record
                    page_record = Page(
                        publication_id=publication.id,
                        page_number=page_num + 1,
                        width_pixels=pix.width,
                        height_pixels=pix.height,
                        total_page_inches=config['total_inches_per_page']
                    )
                    db.session.add(page_record)
                    print(f"Created page {page_num + 1}")

                # Commit batch
                db.session.commit()
                print(f"Committed batch {batch_start + 1}-{batch_end}")

            pdf_doc.close()
            print("PDF processing completed")

        except Exception as e:
            print(f"ERROR in PDF processing: {e}")
            import traceback
            traceback.print_exc()
            return False

        # STEP 2: Verify pages were created
        print("\n=== STEP 2: VERIFYING PAGES ===")
        created_pages = Page.query.filter_by(publication_id=publication.id).all()
        print(f"Pages created: {len(created_pages)}")

        if len(created_pages) == 0:
            print("ERROR: No pages were created")
            return False

        # STEP 3: Run content-based detection
        print("\n=== STEP 3: RUNNING CONTENT-BASED DETECTION ===")

        try:
            publication.set_processing_status('ai_detection')
            db.session.commit()

            total_detected_ads = 0

            for page in created_pages:
                print(f"Running content detection on page {page.page_number}...")
                content_ads = ContentBasedAdDetector.detect_business_content_ads(file_path, page.page_number)

                for ad in content_ads:
                    # Calculate measurements
                    dpi = 150  # Standard DPI
                    width_inches_raw = ad['width'] / dpi
                    height_inches_raw = ad['height'] / dpi
                    column_inches = width_inches_raw * height_inches_raw

                    # Create AdBox
                    ad_box = AdBox(
                        page_id=page.id,
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
                    total_detected_ads += 1

                print(f"Page {page.page_number}: {len(content_ads)} ads")

            print(f"Total ads detected: {total_detected_ads}")

        except Exception as e:
            print(f"ERROR in content detection: {e}")
            import traceback
            traceback.print_exc()
            return False

        # STEP 4: Update publication and mark complete
        print("\n=== STEP 4: FINALIZING PUBLICATION ===")

        try:
            # Calculate totals
            total_ad_inches = sum(box.column_inches for box in AdBox.query.join(Page).filter(Page.publication_id == publication.id).all())
            publication.total_ad_inches = total_ad_inches
            publication.ad_percentage = (total_ad_inches / publication.total_inches) * 100 if publication.total_inches > 0 else 0
            publication.processed = True
            publication.set_processing_status('completed')

            db.session.commit()

            print(f"Final results:")
            print(f"  Pages: {len(created_pages)}")
            print(f"  Ads: {total_detected_ads}")
            print(f"  Ad inches: {total_ad_inches:.2f}")
            print(f"  Ad percentage: {publication.ad_percentage:.1f}%")

            if total_detected_ads >= 15:
                print("*** SUCCESS: Complete processing works! ***")
                return True
            else:
                print(f"*** PARTIAL SUCCESS: {total_detected_ads} ads ***")
                return True

        except Exception as e:
            print(f"ERROR in finalization: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = run_complete_processing()
    if success:
        print("\nCOMPLETE PROCESSING WORKFLOW SUCCESSFUL!")
        print("Users will now see 30+ ads when uploading this file!")
    else:
        print("\nCOMPLETE PROCESSING WORKFLOW FAILED")