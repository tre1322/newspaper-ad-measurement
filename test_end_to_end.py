#!/usr/bin/env python3
"""
Test complete end-to-end workflow with the fix applied
"""

import os
import sys
from app import app, db, Publication, Page, AdBox, MLModel, TrainingData, AdLearningEngine

def test_end_to_end_with_fix():
    """Test the complete workflow with confidence threshold fix"""

    with app.app_context():
        print("=== END-TO-END TEST WITH CONFIDENCE THRESHOLD FIX ===")
        print()

        # Find the most recent publication to test against
        recent_pub = Publication.query.order_by(Publication.id.desc()).first()

        if not recent_pub:
            print("[ERROR] No publications found")
            return

        print(f"[TEST] Testing on publication: {recent_pub.original_filename} (ID: {recent_pub.id})")
        print(f"       Publication type: {recent_pub.publication_type}")

        # Test the auto-detection with the new confidence threshold (0.1)
        print()
        print("[TEST] Running auto-detection with confidence threshold 0.1...")

        try:
            result = AdLearningEngine.auto_detect_ads(recent_pub.id, confidence_threshold=0.1)

            if result and result.get('success'):
                detections = result.get('detections', 0)
                pages_processed = result.get('pages_processed', 0)

                print(f"[SUCCESS] Auto-detection successful!")
                print(f"          Ads detected: {detections}")
                print(f"          Pages processed: {pages_processed}")

                if detections > 0:
                    print("[SUCCESS] ISSUE FIXED: The system now detects ads on re-upload!")

                    # Test with higher confidence to show the contrast
                    print()
                    print("[TEST] Comparing with higher confidence threshold (0.3)...")
                    result_high = AdLearningEngine.auto_detect_ads(recent_pub.id, confidence_threshold=0.3)
                    high_detections = result_high.get('detections', 0) if result_high and result_high.get('success') else 0

                    print(f"          High confidence (0.3): {high_detections} ads")
                    print(f"          Low confidence (0.1): {detections} ads")
                    print(f"          Improvement: +{detections - high_detections} ads detected")

                else:
                    print("[ISSUE] Still no ads detected - deeper investigation needed")

            else:
                error = result.get('error', 'Unknown error') if result else 'No result returned'
                print(f"[ERROR] Auto-detection failed: {error}")

        except Exception as e:
            print(f"[ERROR] Auto-detection crashed: {e}")

        print()
        print("=== TEST COMPLETE ===")

if __name__ == "__main__":
    test_end_to_end_with_fix()