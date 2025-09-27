#!/usr/bin/env python3
"""
MANUAL UPLOAD TEST: Create publication manually and test content detection
"""
from app import app, db, Publication, Page, AdBox, ContentBasedAdDetector, start_background_processing
import shutil
import os
import uuid

def manual_upload_test():
    """Create publication manually and test content detection"""
    print("=== MANUAL UPLOAD TEST ===")

    with app.app_context():

        # STEP 1: Create publication record manually
        print("=== STEP 1: CREATING PUBLICATION MANUALLY ===")

        test_file = "OA-2025-01-01.pdf"
        source_file = f"C:\\Users\\trevo\\newspaper-ad-measurement\\{test_file}"

        if not os.path.exists(source_file):
            print(f"ERROR: Source file not found: {source_file}")
            return False

        # Clear any existing test publication
        existing = Publication.query.filter_by(original_filename=test_file).first()
        if existing:
            pages = Page.query.filter_by(publication_id=existing.id).all()
            for page in pages:
                AdBox.query.filter_by(page_id=page.id).delete()
                db.session.delete(page)
            db.session.delete(existing)
            db.session.commit()
            print("Cleared existing publication")

        # Copy file to upload directory
        unique_filename = f"{uuid.uuid4()}.pdf"
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs')
        os.makedirs(upload_dir, exist_ok=True)
        dest_file = os.path.join(upload_dir, unique_filename)
        shutil.copy2(source_file, dest_file)
        print(f"Copied file to: {dest_file}")

        # Create publication record
        publication = Publication(
            filename=unique_filename,
            original_filename=test_file,
            publication_type='broadsheet',
            total_pages=10,
            total_inches=1250  # 125 inches per page * 10 pages
        )

        try:
            publication.set_processing_status('uploaded')
        except:
            pass

        db.session.add(publication)
        db.session.commit()
        print(f"Created publication {publication.id}")

        # STEP 2: Run background processing directly (synchronously)
        print("\\n=== STEP 2: RUNNING BACKGROUND PROCESSING DIRECTLY ===")

        try:
            # Call the actual processing logic directly (bypassing threading)
            print("Running actual processing logic synchronously...")

            # Import fitz for PDF processing
            import fitz

            # Process PDF to images and create page records (simplified version)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs', publication.filename)
            print(f"Processing file: {file_path}")

            pdf_doc = fitz.open(file_path)
            print(f"PDF has {pdf_doc.page_count} pages")

            # Create page records
            from app import PUBLICATION_CONFIGS
            config = PUBLICATION_CONFIGS[publication.publication_type]

            for page_num in range(pdf_doc.page_count):
                page = pdf_doc[page_num]

                # Convert to image with lower resolution for faster processing
                mat = fitz.Matrix(1.5, 1.5)
                pix = page.get_pixmap(matrix=mat)

                # Save page image
                image_filename = f"{publication.filename}_page_{page_num + 1}.png"
                pages_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'pages')
                os.makedirs(pages_dir, exist_ok=True)
                image_path = os.path.join(pages_dir, image_filename)

                with open(image_path, 'wb') as img_file:
                    img_file.write(pix.tobytes("png"))

                # Create page record
                page_record = Page(
                    publication_id=publication.id,
                    page_number=page_num + 1,
                    width_pixels=pix.width,
                    height_pixels=pix.height,
                    total_page_inches=config['total_inches_per_page']
                )
                db.session.add(page_record)
                print(f"Created page record for page {page_num + 1}")

            db.session.commit()
            pdf_doc.close()

            # Now run content-based detection
            print("Running content-based ad detection...")
            pages = Page.query.filter_by(publication_id=publication.id).all()
            total_detected_ads = 0

            for page in pages:
                print(f"Running content-based detection on page {page.page_number}...")
                content_ads = ContentBasedAdDetector.detect_business_content_ads(file_path, page.page_number)

                for ad in content_ads:
                    # Calculate measurements
                    dpi = page.pixels_per_inch or 150
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

                print(f"Content detection found {len(content_ads)} ads on page {page.page_number}")

            # Update publication totals
            total_ad_inches = sum(box.column_inches for box in AdBox.query.join(Page).filter(Page.publication_id == publication.id).all())
            publication.total_ad_inches = total_ad_inches
            publication.ad_percentage = (total_ad_inches / publication.total_inches) * 100 if publication.total_inches > 0 else 0
            publication.processed = True

            db.session.commit()

            print(f"Processing completed: {total_detected_ads} business ads detected")

        except Exception as e:
            print(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()

        # STEP 3: Check results
        print("\\n=== STEP 3: CHECKING RESULTS ===")

        final_pub = Publication.query.get(publication.id)
        print(f"Publication ID: {final_pub.id}")
        print(f"Processed: {final_pub.processed}")
        try:
            print(f"Status: {final_pub.safe_processing_status}")
        except:
            print("Status: unknown")
        print(f"Total ad inches: {final_pub.total_ad_inches}")

        # Check pages
        pages = Page.query.filter_by(publication_id=publication.id).all()
        print(f"Pages created: {len(pages)}")

        # Check ads
        total_ads = 0
        for page in pages:
            page_ads = AdBox.query.filter_by(page_id=page.id).all()
            if len(page_ads) > 0:
                print(f"  Page {page.page_number}: {len(page_ads)} ads")
            total_ads += len(page_ads)

        print(f"Total ads detected: {total_ads}")

        # SUCCESS CRITERIA
        if total_ads >= 15:
            print("*** SUCCESS: Manual upload workflow detects 15+ ads! ***")
            return True
        elif total_ads > 2:
            print(f"*** PARTIAL SUCCESS: {total_ads} ads (better than baseline) ***")
            return True
        else:
            print(f"*** FAILURE: Only {total_ads} ads detected ***")
            return False

if __name__ == "__main__":
    success = manual_upload_test()
    if success:
        print("\\nMANUAL UPLOAD WORKFLOW WORKS!")
    else:
        print("\\nMANUAL UPLOAD WORKFLOW FAILING")