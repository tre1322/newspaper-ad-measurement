#!/usr/bin/env python3
"""
Test the full workflow: upload → manual marking → generate report → re-upload → verify detection

This will help identify exactly where the AI detection is failing.
"""

import os
import sys
from app import app, db, Publication, Page, AdBox, MLModel, TrainingData, AdLearningEngine

def test_full_workflow():
    """Test the complete AI learning and detection workflow"""

    with app.app_context():
        print("=== FULL WORKFLOW TEST ===")
        print()

        # Step 1: Find a recent publication with manual ads marked
        print("1. Finding publication with manually marked ads...")
        pub_with_ads = db.session.query(Publication).join(Page).join(AdBox).filter(
            AdBox.user_verified == True
        ).order_by(Publication.id.desc()).first()

        if not pub_with_ads:
            print("[ERROR] No publications found with manually marked ads")
            return

        print(f"[OK] Found publication: {pub_with_ads.original_filename} (ID: {pub_with_ads.id})")
        print(f"   Publication type: {pub_with_ads.publication_type}")

        # Count manual ads
        manual_ads = db.session.query(AdBox).join(Page).filter(
            Page.publication_id == pub_with_ads.id,
            AdBox.user_verified == True
        ).count()
        print(f"   Manual ads marked: {manual_ads}")

        # Step 2: Check if training data was created from this publication
        print()
        print("2. Checking if training data was extracted...")

        training_data_count = db.session.query(TrainingData).join(AdBox).join(Page).filter(
            Page.publication_id == pub_with_ads.id
        ).count()

        if training_data_count > 0:
            print(f"[OK] Training data extracted: {training_data_count} samples")
        else:
            print(f"[ERROR] No training data found for this publication")
            print("   This suggests the AI learning step failed during report generation")
            return

        # Step 3: Check if ML model exists and is active
        print()
        print("3. Checking ML model availability...")

        active_model = MLModel.query.filter_by(
            publication_type=pub_with_ads.publication_type,
            model_type='ad_detector',
            is_active=True
        ).first()

        if active_model:
            print(f"[OK] Active ML model found for {pub_with_ads.publication_type}")
            print(f"   Created: {active_model.created_date}")
        else:
            print(f"[ERROR] No active ML model found for {pub_with_ads.publication_type}")
            available_models = MLModel.query.filter_by(publication_type=pub_with_ads.publication_type).all()
            print(f"   Available models: {len(available_models)} (all inactive)")
            return

        # Step 4: Test the auto-detection function directly
        print()
        print("4. Testing auto-detection on the same publication...")

        try:
            result = AdLearningEngine.auto_detect_ads(pub_with_ads.id, confidence_threshold=0.3)

            if result and result.get('success'):
                print(f"[OK] Auto-detection successful!")
                print(f"   Ads detected: {result.get('detections', 0)}")
                print(f"   Pages processed: {result.get('pages_processed', 0)}")
                print(f"   Model used: {result.get('model_used', 'unknown')}")

                if result.get('detections', 0) == 0:
                    print()
                    print("[ISSUE] ISSUE IDENTIFIED: Auto-detection runs but finds 0 ads")
                    print("   This suggests:")
                    print("   - Confidence threshold might be too high")
                    print("   - Model might not be well-trained")
                    print("   - Feature extraction might be failing")

            else:
                error = result.get('error', 'Unknown error') if result else 'No result returned'
                print(f"[ERROR] Auto-detection failed: {error}")

        except Exception as e:
            print(f"[ERROR] Auto-detection crashed: {e}")
            import traceback
            traceback.print_exc()

        # Step 5: Test with very low confidence threshold
        print()
        print("5. Testing with very low confidence threshold (0.1)...")

        try:
            result = AdLearningEngine.auto_detect_ads(pub_with_ads.id, confidence_threshold=0.1)

            if result and result.get('success'):
                detections = result.get('detections', 0)
                print(f"[OK] Low-confidence detection: {detections} ads found")

                if detections > 0:
                    print("[SUCCESS] The model works, but normal confidence threshold is too high")
                else:
                    print("[ISSUE] DEEPER ISSUE: Even low confidence finds nothing")

        except Exception as e:
            print(f"[ERROR] Low-confidence detection failed: {e}")

        # Step 6: Check training data quality
        print()
        print("6. Analyzing training data quality...")

        total_training = TrainingData.query.filter_by(
            publication_type=pub_with_ads.publication_type
        ).count()

        recent_training = TrainingData.query.filter_by(
            publication_type=pub_with_ads.publication_type
        ).order_by(TrainingData.extracted_date.desc()).limit(10).all()

        print(f"   Total training samples: {total_training}")
        print(f"   Recent samples preview:")
        for sample in recent_training[:3]:
            print(f"     - Size: {sample.box_width}x{sample.box_height}, Features: {len(sample.features) if sample.features else 0}")

        print()
        print("=== WORKFLOW TEST COMPLETE ===")

        if result and result.get('success') and result.get('detections', 0) > 0:
            print("[SUCCESS] CONCLUSION: The system is working correctly!")
        else:
            print("[ISSUE] CONCLUSION: Issue identified - see analysis above")

if __name__ == "__main__":
    test_full_workflow()