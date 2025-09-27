#!/usr/bin/env python3
"""
CHECK RECENT UPLOADS: See what actually happened with recent uploads
"""
from app import app, db, Publication, Page, AdBox

def check_recent_uploads():
    """Check recent uploads and their processing status"""
    print("=== CHECKING RECENT UPLOADS ===")

    with app.app_context():

        # Get all publications, sorted by most recent
        publications = Publication.query.order_by(Publication.id.desc()).limit(10).all()

        print(f"Found {len(publications)} recent publications:")
        print()

        for pub in publications:
            print(f"Publication {pub.id}:")
            print(f"  Filename: {pub.original_filename}")
            print(f"  Type: {pub.publication_type}")
            print(f"  Processed: {pub.processed}")

            # Check processing status if available
            try:
                print(f"  Status: {pub.safe_processing_status}")
            except:
                print(f"  Status: N/A")

            # Check pages
            pages = Page.query.filter_by(publication_id=pub.id).all()
            print(f"  Pages: {len(pages)}")

            # Check ads
            total_ads = 0
            for page in pages:
                page_ads = AdBox.query.filter_by(page_id=page.id).all()
                total_ads += len(page_ads)

            print(f"  Ads: {total_ads}")
            print(f"  Total ad inches: {pub.total_ad_inches}")
            print()

        # Focus on OA files
        oa_pubs = Publication.query.filter_by(original_filename="OA-2025-01-01.pdf").all()

        if oa_pubs:
            print("=== OA-2025-01-01.pdf PUBLICATIONS ===")
            for pub in oa_pubs:
                print(f"OA Publication {pub.id}:")
                print(f"  Processed: {pub.processed}")

                pages = Page.query.filter_by(publication_id=pub.id).all()
                print(f"  Pages: {len(pages)}")

                total_ads = 0
                for page in pages:
                    page_ads = AdBox.query.filter_by(page_id=page.id).all()
                    total_ads += len(page_ads)

                print(f"  Ads: {total_ads}")
                print(f"  Ad inches: {pub.total_ad_inches}")

                # Show page-by-page breakdown
                if pages:
                    print("  Page breakdown:")
                    for page in pages:
                        page_ads = AdBox.query.filter_by(page_id=page.id).all()
                        if len(page_ads) > 0:
                            print(f"    Page {page.page_number}: {len(page_ads)} ads")

                print()

        return True

if __name__ == "__main__":
    check_recent_uploads()