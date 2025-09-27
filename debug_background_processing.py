#!/usr/bin/env python3
"""
DEBUG BACKGROUND PROCESSING: Find exactly where it fails
"""
from app import app, db, Publication, Page, PUBLICATION_CONFIGS
import os
import fitz

def debug_background_processing():
    """Debug the background processing step by step"""
    print("=== DEBUGGING BACKGROUND PROCESSING ===")

    with app.app_context():

        # Find OA publication
        test_file = "OA-2025-01-01.pdf"
        publication = Publication.query.filter_by(original_filename=test_file).first()

        if not publication:
            print(f"ERROR: {test_file} not found")
            return False

        print(f"Publication {publication.id}: {publication.original_filename}")

        # Clear existing pages
        pages = Page.query.filter_by(publication_id=publication.id).all()
        for page in pages:
            db.session.delete(page)
        db.session.commit()
        print("Cleared existing pages")

        # Test the EXACT background processing logic
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs', publication.filename)
        print(f"PDF path: {file_path}")

        if not os.path.exists(file_path):
            print(f"ERROR: PDF not found at {file_path}")
            return False

        try:
            pdf_doc = fitz.open(file_path)
            print(f"PDF opened successfully - {pdf_doc.page_count} pages")

            # Process each page (EXACT same logic as background processing)
            batch_size = 3  # Same as background processing
            config = PUBLICATION_CONFIGS[publication.publication_type]

            for batch_start in range(0, pdf_doc.page_count, batch_size):
                batch_end = min(batch_start + batch_size, pdf_doc.page_count)

                print(f"\nProcessing batch {batch_start + 1}-{batch_end}")

                for page_num in range(batch_start, batch_end):
                    print(f"  Processing page {page_num + 1}...")

                    try:
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

                        print(f"    Saved image: {image_path}")

                        # Create page record
                        page_record = Page(
                            publication_id=publication.id,
                            page_number=page_num + 1,
                            width_pixels=pix.width,
                            height_pixels=pix.height,
                            total_page_inches=config['total_inches_per_page']
                        )
                        db.session.add(page_record)
                        print(f"    Created page record for page {page_num + 1}")

                    except Exception as page_error:
                        print(f"    ERROR processing page {page_num + 1}: {page_error}")
                        import traceback
                        traceback.print_exc()
                        return False

                # Commit the batch
                try:
                    db.session.commit()
                    print(f"  Committed batch {batch_start + 1}-{batch_end}")
                except Exception as commit_error:
                    print(f"  ERROR committing batch: {commit_error}")
                    return False

            pdf_doc.close()
            print("\nPDF processing completed successfully")

            # Verify all pages were created
            final_pages = Page.query.filter_by(publication_id=publication.id).all()
            print(f"Final page count: {len(final_pages)}")

            if len(final_pages) == pdf_doc.page_count:
                print("SUCCESS: All pages created correctly")
                return True
            else:
                print(f"FAILURE: Expected {pdf_doc.page_count} pages, got {len(final_pages)}")
                return False

        except Exception as e:
            print(f"ERROR in background processing debug: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = debug_background_processing()
    if success:
        print("\nBACKGROUND PROCESSING WORKS!")
    else:
        print("\nBACKGROUND PROCESSING FAILED")