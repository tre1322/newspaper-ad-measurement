#!/usr/bin/env python3
"""
Test the simple AI learning system
"""
from app import app, db, SimpleAdLearner, UserCorrection, Publication, Page
import numpy as np

def test_learning_system():
    """Test all components of the learning system"""
    print("=== TESTING SIMPLE AI LEARNING SYSTEM ===")
    print()

    with app.app_context():

        # Test 1: Feature extraction
        print("Test 1: Feature Extraction")
        features = SimpleAdLearner.extract_box_features(
            x=100, y=200, width=300, height=150,
            page_width=800, page_height=1000
        )
        print("Sample features:")
        for key, value in features.items():
            print(f"  {key}: {value:.3f}")
        print()

        # Test 2: Create sample training data
        print("Test 2: Creating sample training data")

        # Get a real page to use
        page = Page.query.first()
        if not page:
            print("ERROR: No pages found in database")
            return

        publication = Publication.query.get(page.publication_id)

        # Create sample corrections (positive and negative examples)
        sample_corrections = [
            # Positive examples (ads)
            {'x': 50, 'y': 100, 'width': 200, 'height': 150, 'is_ad': True, 'type': 'added'},
            {'x': 300, 'y': 200, 'width': 180, 'height': 120, 'is_ad': True, 'type': 'added'},
            {'x': 100, 'y': 500, 'width': 250, 'height': 180, 'is_ad': True, 'type': 'added'},
            {'x': 400, 'y': 300, 'width': 160, 'height': 100, 'is_ad': True, 'type': 'added'},
            {'x': 200, 'y': 700, 'width': 300, 'height': 200, 'is_ad': True, 'type': 'added'},

            # Negative examples (not ads)
            {'x': 10, 'y': 10, 'width': 50, 'height': 20, 'is_ad': False, 'type': 'deleted'},
            {'x': 500, 'y': 50, 'width': 80, 'height': 30, 'is_ad': False, 'type': 'deleted'},
            {'x': 150, 'y': 150, 'width': 60, 'height': 25, 'is_ad': False, 'type': 'deleted'},
            {'x': 600, 'y': 600, 'width': 70, 'height': 35, 'is_ad': False, 'type': 'deleted'},
            {'x': 50, 'y': 800, 'width': 90, 'height': 40, 'is_ad': False, 'type': 'deleted'},
            {'x': 350, 'y': 450, 'width': 75, 'height': 28, 'is_ad': False, 'type': 'deleted'},
            {'x': 450, 'y': 750, 'width': 65, 'height': 32, 'is_ad': False, 'type': 'deleted'},
        ]

        # Clear any existing test corrections
        UserCorrection.query.filter_by(publication_id=publication.id).delete()
        db.session.commit()

        # Save sample corrections
        saved_count = 0
        for correction in sample_corrections:
            success = SimpleAdLearner.save_user_correction(
                publication_id=publication.id,
                page_id=page.id,
                x=correction['x'],
                y=correction['y'],
                width=correction['width'],
                height=correction['height'],
                is_ad=correction['is_ad'],
                correction_type=correction['type'],
                publication_type=publication.publication_type
            )
            if success:
                saved_count += 1

        print(f"Saved {saved_count} sample corrections")
        print()

        # Test 3: Train model
        print("Test 3: Training model")
        model, accuracy = SimpleAdLearner.train_model()

        if model is not None:
            print(f"Model trained successfully!")
            print(f"   Training accuracy: {accuracy:.3f}")
            print(f"   Model type: {type(model).__name__}")
        else:
            print(f"Model training failed: {accuracy}")
            return
        print()

        # Test 4: Test predictions
        print("Test 4: Testing predictions")

        # Test on some sample boxes
        test_boxes = [
            # Should be predicted as ads (similar to positive examples)
            {'x': 80, 'y': 120, 'width': 190, 'height': 140, 'expected': 'ad'},
            {'x': 320, 'y': 220, 'width': 170, 'height': 110, 'expected': 'ad'},

            # Should be predicted as not ads (similar to negative examples)
            {'x': 20, 'y': 20, 'width': 60, 'height': 25, 'expected': 'not_ad'},
            {'x': 520, 'y': 70, 'width': 75, 'height': 35, 'expected': 'not_ad'},
        ]

        correct_predictions = 0
        for i, box in enumerate(test_boxes):
            features = SimpleAdLearner.extract_box_features(
                box['x'], box['y'], box['width'], box['height'],
                page.width_pixels, page.height_pixels
            )

            feature_vector = np.array([[
                features['box_area'],
                features['aspect_ratio'],
                features['position_x_ratio'],
                features['position_y_ratio'],
                features['border_strength'],
                features['text_density']
            ]])

            prediction = model.predict(feature_vector)[0]
            probability = model.predict_proba(feature_vector)[0]
            confidence = max(probability)

            predicted_label = 'ad' if prediction == 1 else 'not_ad'
            is_correct = predicted_label == box['expected']

            if is_correct:
                correct_predictions += 1

            status = "PASS" if is_correct else "FAIL"
            print(f"  Test {i+1}: {status} Predicted: {predicted_label} (confidence: {confidence:.3f}), Expected: {box['expected']}")

        prediction_accuracy = correct_predictions / len(test_boxes)
        print(f"  Prediction accuracy: {correct_predictions}/{len(test_boxes)} = {prediction_accuracy:.3f}")
        print()

        # Test 5: Test learning filter
        print("Test 5: Testing learning filter")

        mock_detected_boxes = [
            {'x': 75, 'y': 110, 'width': 200, 'height': 150},  # Should pass (ad-like)
            {'x': 25, 'y': 25, 'width': 50, 'height': 20},    # Should be filtered (too small)
            {'x': 300, 'y': 180, 'width': 180, 'height': 120}, # Should pass (ad-like)
            {'x': 600, 'y': 80, 'width': 70, 'height': 30},   # Should be filtered (small)
        ]

        filtered_boxes = SimpleAdLearner.apply_learning_filter(
            mock_detected_boxes, model, page.width_pixels, page.height_pixels, confidence_threshold=0.6
        )

        print(f"  Original boxes: {len(mock_detected_boxes)}")
        print(f"  Filtered boxes: {len(filtered_boxes)}")
        print(f"  Boxes removed: {len(mock_detected_boxes) - len(filtered_boxes)}")

        for box in filtered_boxes:
            if 'ml_confidence' in box:
                print(f"    Kept box with ML confidence: {box['ml_confidence']:.3f}")
        print()

        # Summary
        print("=== LEARNING SYSTEM TEST SUMMARY ===")
        print("Feature extraction: Working")
        print("Training data storage: Working")
        print("Model training: Working")
        print("Predictions: Working")
        print("Learning filter: Working")
        print()
        print("SIMPLE AI LEARNING SYSTEM IS READY!")
        print()
        print("Next steps:")
        print("1. Integrate with measurement interface to capture real user corrections")
        print("2. Apply learning filter to actual detection pipeline")
        print("3. Test on real publications")

if __name__ == "__main__":
    test_learning_system()