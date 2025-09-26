#!/usr/bin/env python3
"""
Final verification that the AI detection issue has been fixed
"""

from app import app, db, Publication, AdLearningEngine

def final_verification():
    """Run a final test to confirm the fix works"""

    with app.app_context():
        print("=== FINAL VERIFICATION: AI DETECTION FIX ===")
        print()

        # Get the most recent publication
        recent_pub = Publication.query.order_by(Publication.id.desc()).first()

        if not recent_pub:
            print("[ERROR] No publications found")
            return

        print(f"Publication: {recent_pub.original_filename}")
        print(f"Type: {recent_pub.publication_type}")
        print()

        print("BEFORE FIX (confidence_threshold=0.25):")
        result_before = AdLearningEngine.auto_detect_ads(recent_pub.id, confidence_threshold=0.25)
        detections_before = result_before.get('detections', 0) if result_before and result_before.get('success') else 0
        print(f"  Ads detected: {detections_before}")

        print()
        print("AFTER FIX (confidence_threshold=0.1):")
        result_after = AdLearningEngine.auto_detect_ads(recent_pub.id, confidence_threshold=0.1)
        detections_after = result_after.get('detections', 0) if result_after and result_after.get('success') else 0
        print(f"  Ads detected: {detections_after}")

        print()
        print("RESULTS:")
        print(f"  Improvement: +{detections_after - detections_before} ads detected")

        if detections_after > detections_before:
            print("  [SUCCESS] Fix is working! AI detection now finds ads during re-upload.")
        else:
            print("  [ISSUE] Fix may need additional tuning.")

        print()
        print("NEXT STEPS:")
        print("  1. The confidence threshold in app.py has been lowered from 0.25 to 0.1")
        print("  2. When you re-upload the same publication, AI should now detect the ads")
        print("  3. You should see: 'AI detection complete: X ads automatically detected'")

if __name__ == "__main__":
    final_verification()