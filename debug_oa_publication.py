#!/usr/bin/env python3
"""
DEBUG OA PUBLICATION: Check what happened with publication 30
"""
from app import app, db, Publication, Page, AdBox, ContentBasedAdDetector
import os

def debug_oa_publication():
    """Debug the specific OA publication that was processed"""
    print("=== DEBUGGING OA PUBLICATION 30 ===")

    with app.app_context():

        # Get the specific publication
        publication = Publication.query.get(30)
        if not publication:
            print("ERROR: Publication 30 not found")
            return False

        print(f"Publication {publication.id}: {publication.original_filename}")
        print(f"Processed: {publication.processed}")
        print(f"Status: {publication.safe_processing_status}")

        # Check the file path
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pdfs', publication.filename)
        print(f"File path: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")

        # Check pages
        pages = Page.query.filter_by(publication_id=publication.id).all()
        print(f"Pages: {len(pages)}")

        # Check ads
        total_ads = 0
        for page in pages:
            page_ads = AdBox.query.filter_by(page_id=page.id).all()
            total_ads += len(page_ads)
            if len(page_ads) > 0:
                print(f"  Page {page.page_number}: {len(page_ads)} ads")

        print(f"Total ads in database: {total_ads}")

        # Now test content-based detection on this file to compare
        print("\n=== TESTING CONTENT DETECTION ON THIS FILE ===")

        total_content_ads = 0
        for page_num in range(1, 11):  # 10 pages
            try:
                content_ads = ContentBasedAdDetector.detect_business_content_ads(file_path, page_num)
                print(f"Page {page_num}: {len(content_ads)} content ads")
                total_content_ads += len(content_ads)
            except Exception as e:
                print(f"Page {page_num}: ERROR - {e}")

        print(f"Total content ads possible: {total_content_ads}")
        print(f"Difference: {total_content_ads - total_ads} ads missing")

        # The issue might be that the wrong file is being used
        # Let's check if the original file exists
        original_file = "C:\\Users\\trevo\\newspaper-ad-measurement\\OA-2025-01-01.pdf"
        print(f"\nOriginal file exists: {os.path.exists(original_file)}")

        if os.path.exists(original_file):
            print("Testing content detection on original file:")
            total_original_ads = 0
            for page_num in range(1, 11):
                try:
                    content_ads = ContentBasedAdDetector.detect_business_content_ads(original_file, page_num)
                    print(f"Page {page_num}: {len(content_ads)} ads")
                    total_original_ads += len(content_ads)
                except Exception as e:
                    print(f"Page {page_num}: ERROR - {e}")

            print(f"Total ads on original file: {total_original_ads}")

        return True

if __name__ == "__main__":
    debug_oa_publication()