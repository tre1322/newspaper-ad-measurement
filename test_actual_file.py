#!/usr/bin/env python3
"""
ACCOUNTABILITY TEST: Test actual OA-2025-01-01.pdf file to prove AI learning works
"""
from app import app, db, SimpleAdDetector, SimpleAdLearner, Publication, Page, AdBox
import os
import fitz

def test_actual_file():
    """Test the exact file the user mentioned"""
    print("=== ACCOUNTABILITY TEST: OA-2025-01-01.pdf ===")
    print()

    with app.app_context():

        # Find the OA-2025-01-01.pdf publication
        test_file = "OA-2025-01-01.pdf"
        publication = Publication.query.filter_by(original_filename=test_file).first()

        if not publication:
            print(f"ERROR: {test_file} not found in database")
            print("Available publications:")
            pubs = Publication.query.all()
            for pub in pubs:
                print(f"  - {pub.original_filename}")
            return False

        print(f"Found publication {publication.id}: {publication.original_filename}")

        # Check if it has pages
        pages = Page.query.filter_by(publication_id=publication.id).all()
        print(f"Pages in database: {len(pages)}")

        if len(pages) == 0:
            print("ERROR: No pages found - need to process this file first")
            return False

        # Test 1: Check current detection count
        print("\n=== CURRENT DETECTION COUNT ===")
        current_ads = AdBox.query.join(Page).filter(Page.publication_id == publication.id).all()
        print(f"Current ads detected: {len(current_ads)}")

        # Clear existing detections to test fresh
        for ad in current_ads:
            db.session.delete(ad)
        db.session.commit()
        print("Cleared existing detections for fresh test")

        # Test 2: Run detection WITHOUT learning filter first
        print("\n=== TEST 1: DETECTION WITHOUT LEARNING ===")

        # Temporarily disable learning by checking if we can modify the detection
        # Let's run the basic detection to see baseline
        try:
            result = SimpleAdDetector.detect_bordered_ads(publication.id)
            if result['success']:
                baseline_count = result['detections']
                print(f"BASELINE DETECTION: {baseline_count} ads found")
            else:
                print(f"BASELINE DETECTION FAILED: {result['error']}")
                return False
        except Exception as e:
            print(f"BASELINE DETECTION ERROR: {e}")
            return False

        # Test 3: Add manual corrections (simulate user fixing bad detections)
        print("\n=== TEST 2: ADDING MANUAL CORRECTIONS ===")

        # Get first page for testing
        test_page = pages[0]

        # Add positive training examples (real ads user would add)
        training_corrections = [
            # Large display ads (real ads user would manually add)
            {'x': 50, 'y': 100, 'width': 300, 'height': 200, 'is_ad': True, 'type': 'added'},
            {'x': 400, 'y': 150, 'width': 250, 'height': 180, 'is_ad': True, 'type': 'added'},
            {'x': 100, 'y': 400, 'width': 280, 'height': 150, 'is_ad': True, 'type': 'added'},
            {'x': 450, 'y': 500, 'width': 200, 'height': 160, 'is_ad': True, 'type': 'added'},
            {'x': 50, 'y': 700, 'width': 320, 'height': 180, 'is_ad': True, 'type': 'added'},

            # Small non-ads (things user would delete)
            {'x': 10, 'y': 10, 'width': 40, 'height': 15, 'is_ad': False, 'type': 'deleted'},
            {'x': 600, 'y': 50, 'width': 60, 'height': 25, 'is_ad': False, 'type': 'deleted'},
            {'x': 20, 'y': 900, 'width': 50, 'height': 20, 'is_ad': False, 'type': 'deleted'},
        ]

        corrections_saved = 0
        for correction in training_corrections:
            success = SimpleAdLearner.save_user_correction(
                publication_id=publication.id,
                page_id=test_page.id,
                x=correction['x'],
                y=correction['y'],
                width=correction['width'],
                height=correction['height'],
                is_ad=correction['is_ad'],
                correction_type=correction['type'],
                publication_type=publication.publication_type
            )
            if success:
                corrections_saved += 1

        print(f"Added {corrections_saved} training corrections")

        # Train model on corrections
        print("\n=== TEST 3: TRAINING MODEL ===")
        model, accuracy = SimpleAdLearner.train_model()
        if model is None:
            print("ERROR: Model training failed")
            return False

        print(f"Model trained with accuracy: {accuracy:.3f}")

        # Clear detections again for final test
        current_ads = AdBox.query.join(Page).filter(Page.publication_id == publication.id).all()
        for ad in current_ads:
            db.session.delete(ad)
        db.session.commit()

        # Test 4: Run detection WITH learning filter
        print("\n=== TEST 4: DETECTION WITH LEARNING ===")

        try:
            result = SimpleAdDetector.detect_bordered_ads(publication.id)
            if result['success']:
                learned_count = result['detections']
                print(f"LEARNED DETECTION: {learned_count} ads found")
                print(f"IMPROVEMENT: {learned_count - baseline_count} additional ads")

                if learned_count > baseline_count:
                    print("SUCCESS: AI learning improved detection!")
                    return True
                else:
                    print("FAILURE: AI learning did not improve detection")
                    return False
            else:
                print(f"LEARNED DETECTION FAILED: {result['error']}")
                return False
        except Exception as e:
            print(f"LEARNED DETECTION ERROR: {e}")
            return False

if __name__ == "__main__":
    success = test_actual_file()
    if success:
        print("\n*** AI LEARNING ACTUALLY WORKS ***")
    else:
        print("\n*** AI LEARNING FAILED - SYSTEM NEEDS REDESIGN ***")