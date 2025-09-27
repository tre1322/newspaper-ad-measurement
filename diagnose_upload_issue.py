#!/usr/bin/env python3
"""
Diagnose why uploads aren't working - check if publications are created without pages
"""
from flask import Flask
from app import app, db, Publication, Page

def diagnose_issue():
    """Check the current state of publications and identify the problem"""

    with app.app_context():
        print("=== UPLOAD ISSUE DIAGNOSIS ===")
        print()

        # Get all publications
        publications = Publication.query.all()
        print(f"Total publications in database: {len(publications)}")
        print()

        if publications:
            print("Recent publications:")
            for pub in publications[-5:]:  # Last 5 publications
                pages = Page.query.filter_by(publication_id=pub.id).all()
                print(f"Publication {pub.id}: {pub.original_filename}")
                print(f"  - Type: {pub.publication_type}")
                print(f"  - Total Pages: {pub.total_pages}")
                print(f"  - Pages in DB: {len(pages)}")
                print(f"  - Processed: {pub.processed}")

                if hasattr(pub, 'processing_status'):
                    print(f"  - Processing Status: {pub.processing_status}")

                print()

        # Check if there are publications without pages
        pubs_without_pages = []
        for pub in publications:
            pages = Page.query.filter_by(publication_id=pub.id).all()
            if len(pages) == 0:
                pubs_without_pages.append(pub)

        if pubs_without_pages:
            print(f"PROBLEM FOUND: {len(pubs_without_pages)} publications have no pages!")
            print("This means uploads are creating publication records but background processing is failing.")
            print()
            print("Publications without pages:")
            for pub in pubs_without_pages[-3:]:  # Show last 3
                print(f"  - {pub.id}: {pub.original_filename}")
            print()
            print("SOLUTION: The background processing that creates page records is failing.")
            print("This could be due to:")
            print("1. PDF processing errors")
            print("2. File permission issues")
            print("3. Path resolution problems")
            print("4. Threading issues in background processing")
        else:
            print("All publications have pages - upload processing is working correctly.")

        print()
        print("=== CLASSIFICATION TEST ===")

        # Find a publication with pages to test classification
        pubs_with_pages = [pub for pub in publications if len(Page.query.filter_by(publication_id=pub.id).all()) > 0]

        if pubs_with_pages:
            test_pub = pubs_with_pages[0]
            print(f"Testing classification with publication {test_pub.id}: {test_pub.original_filename}")

            # Import here to avoid circular imports
            from app import SimpleAdDetector

            try:
                result = SimpleAdDetector.detect_regions_for_classification(test_pub.id)
                if result['success']:
                    print("Classification system is working!")
                    print(f"Created classification session: {result.get('session_id')}")
                else:
                    print(f"Classification failed: {result['error']}")
            except Exception as e:
                print(f"Classification error: {e}")
        else:
            print("Cannot test classification - no publications with pages found")

if __name__ == "__main__":
    diagnose_issue()