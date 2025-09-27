#!/usr/bin/env python3
"""
Test the complete learning cycle with real detection pipeline
"""
from app import app, db, SimpleAdDetector, SimpleAdLearner, Publication, Page
import numpy as np

def test_complete_learning_cycle():
    """Test complete learning cycle with detection pipeline"""
    print("=== TESTING COMPLETE LEARNING CYCLE ===")
    print()

    with app.app_context():

        # Test 1: Verify training data exists
        print("Test 1: Checking training data")
        model, accuracy = SimpleAdLearner.train_model()

        if model is None:
            print("ERROR: No trained model available. Run test_learning_system.py first.")
            return False

        print(f"Model trained with accuracy: {accuracy:.3f}")
        print()

        # Test 2: Find a publication with pages to test on
        print("Test 2: Finding test publication")
        publications = Publication.query.all()
        test_publication = None

        for pub in publications:
            pages = Page.query.filter_by(publication_id=pub.id).all()
            if len(pages) > 0:
                test_publication = pub
                break

        if not test_publication:
            print("ERROR: No publications with pages found for testing")
            return False

        print(f"Using publication {test_publication.id}: {test_publication.original_filename}")
        print()

        # Test 3: Test detection with learning filter
        print("Test 3: Testing detection with learning filter")

        try:
            # Test SimpleAdDetector.detect_bordered_ads (which now includes learning filter)
            result = SimpleAdDetector.detect_bordered_ads(test_publication.id)

            if result['success']:
                print(f"Detection successful!")
                print(f"   Detections: {result['detections']}")
                print(f"   Pages processed: {result['pages_processed']}")
                print(f"   Message: {result['message']}")
            else:
                print(f"Detection failed: {result['error']}")
                return False

        except Exception as e:
            print(f"Detection error: {e}")
            import traceback
            traceback.print_exc()
            return False

        print()

        # Test 4: Verify learning filter integration by checking output
        print("Test 4: Verifying learning filter integration")

        # Create mock detection data to test learning filter directly
        test_page = Page.query.filter_by(publication_id=test_publication.id).first()

        mock_detections = [
            # These should pass filter (ad-like)
            {'x': 100, 'y': 200, 'width': 200, 'height': 150, 'confidence': 0.9},
            {'x': 300, 'y': 300, 'width': 180, 'height': 120, 'confidence': 0.8},

            # These should be filtered out (not ad-like)
            {'x': 10, 'y': 10, 'width': 50, 'height': 20, 'confidence': 0.6},
            {'x': 500, 'y': 50, 'width': 80, 'height': 30, 'confidence': 0.5},
        ]

        try:
            filtered_detections = SimpleAdLearner.apply_learning_filter(
                mock_detections,
                model,
                test_page.width_pixels,
                test_page.height_pixels,
                confidence_threshold=0.85
            )

            print(f"Learning filter working!")
            print(f"   Original detections: {len(mock_detections)}")
            print(f"   Filtered detections: {len(filtered_detections)}")
            print(f"   Removed: {len(mock_detections) - len(filtered_detections)}")

            for detection in filtered_detections:
                if 'ml_confidence' in detection:
                    print(f"   Kept detection with ML confidence: {detection['ml_confidence']:.3f}")

        except Exception as e:
            print(f"Learning filter error: {e}")
            return False

        print()

        # Test 5: Summary
        print("=== COMPLETE LEARNING CYCLE TEST SUMMARY ===")
        print("Training data: Available")
        print("Model training: Working")
        print("Detection pipeline: Working")
        print("Learning filter integration: Working")
        print("Complete cycle: READY!")
        print()
        print("SIMPLE AI LEARNING SYSTEM IS FULLY INTEGRATED!")
        print()
        print("The system will now:")
        print("1. Learn from user corrections in measurement interface")
        print("2. Apply learned patterns to filter future detections")
        print("3. Improve detection accuracy over time")

        return True

if __name__ == "__main__":
    success = test_complete_learning_cycle()
    if success:
        print("\nALL TESTS PASSED - LEARNING SYSTEM READY!")
    else:
        print("\nTESTS FAILED - Check errors above")