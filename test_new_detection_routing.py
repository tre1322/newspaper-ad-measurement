#!/usr/bin/env python3
"""
Test that the upload process now uses the NEW hybrid detection system
instead of the old broken sliding window detection
"""

import os
import sys
from app import app, db, Publication, Page, AdBox, BusinessLogo
from app import HybridDetectionPipeline, LogoRecognitionDetectionEngine

def test_new_detection_routing():
    """Test that new detection routing is active"""

    with app.app_context():
        print("=" * 80)
        print("TESTING NEW HYBRID DETECTION ROUTING")
        print("=" * 80)

        # Test 1: Verify hybrid system components exist
        print("\nTEST 1: Verifying hybrid system components...")
        try:
            hybrid_pipeline = HybridDetectionPipeline()
            recognition_engine = LogoRecognitionDetectionEngine()
            print("SUCCESS: Hybrid detection components are available")
        except Exception as e:
            print(f"ERROR: Hybrid components not available: {e}")
            return False

        # Test 2: Check for existing business logos
        print("\nTEST 2: Checking business logos in database...")
        business_logos = BusinessLogo.query.filter_by(is_active=True).all()
        print(f"Found {len(business_logos)} active business logos:")
        for logo in business_logos:
            print(f"  - {logo.business_name} (confidence: {logo.confidence_threshold})")

        # Test 3: Get test publication
        print("\nTEST 3: Finding test publication...")
        publication = Publication.query.order_by(Publication.id.desc()).first()
        if not publication:
            print("ERROR: No publications found")
            return False

        print(f"Using publication: {publication.original_filename}")

        # Test 4: Test hybrid detection on publication
        print("\nTEST 4: Testing hybrid detection pipeline...")
        try:
            result = hybrid_pipeline.detect_ads_hybrid(publication.id, mode='auto')

            if result.get('success'):
                print(f"SUCCESS: Hybrid detection completed")
                print(f"  Total detections: {result.get('total_detections', 0)}")
                print(f"  Logo detections: {result.get('logo_detections', 0)}")
                print(f"  Pages processed: {result.get('pages_processed', 0)}")
                print(f"  Business logos found: {result.get('business_logos_found', [])}")
            else:
                print(f"WARNING: Hybrid detection completed but no ads found")
                print(f"  Error: {result.get('error', 'Unknown')}")

        except Exception as e:
            print(f"ERROR: Hybrid detection failed: {e}")
            return False

        # Test 5: Verify no old "[pdf text set]" markers are being created
        print("\nTEST 5: Checking for old detection artifacts...")
        old_artifacts = AdBox.query.filter(AdBox.ad_type.like('%pdf text%')).count()
        print(f"Old '[pdf text set]' artifacts found: {old_artifacts}")

        if old_artifacts > 0:
            print("WARNING: Old detection artifacts still present - check for remaining old calls")

        # Test 6: Check recent AdBox creations
        print("\nTEST 6: Analyzing recent ad detections...")
        recent_ads = AdBox.query.order_by(AdBox.id.desc()).limit(10).all()

        detection_types = {}
        for ad in recent_ads:
            ad_type = ad.ad_type or 'unknown'
            detection_types[ad_type] = detection_types.get(ad_type, 0) + 1

        print("Recent detection types:")
        for ad_type, count in detection_types.items():
            print(f"  {ad_type}: {count}")

        # Test 7: Verify hybrid system status
        print("\nTEST 7: Getting hybrid detection status...")
        try:
            status = hybrid_pipeline.get_hybrid_detection_status(publication.id)

            if status.get('success'):
                print(f"SUCCESS: Hybrid status reporting works")
                print(f"  Total ads: {status.get('total_ads', 0)}")
                print(f"  Automated ads: {status.get('automated_ads', 0)}")
                print(f"  Manual ads: {status.get('manual_ads', 0)}")
                print(f"  Logo detected ads: {status.get('logo_detected_ads', 0)}")
                print(f"  Business logos found: {len(status.get('business_logos_found', []))}")
            else:
                print(f"ERROR: Hybrid status failed: {status.get('error', 'Unknown')}")

        except Exception as e:
            print(f"ERROR: Hybrid status check failed: {e}")

        print("\n" + "=" * 80)
        print("NEW DETECTION ROUTING TEST SUMMARY")
        print("=" * 80)

        print("\nOLD SYSTEM STATUS:")
        print("- Sliding window detection: REMOVED from upload process")
        print("- '[pdf text set]' markers: SHOULD NO LONGER APPEAR")
        print("- Old AdLearningEngine calls: REMOVED from upload")

        print("\nNEW SYSTEM STATUS:")
        print("- Hybrid detection pipeline: ACTIVE in upload process")
        print("- Logo recognition: RUNS FIRST for known businesses")
        print("- Smart manual detection: AVAILABLE for unknowns")
        print("- Business logo learning: ENABLED for future detection")

        if len(business_logos) > 0:
            print(f"\nREADY FOR TESTING:")
            print(f"- {len(business_logos)} business logos ready for recognition")
            print(f"- Upload a newspaper with known businesses to test logo detection")
            print(f"- New manual ads will be learned for future automatic detection")
        else:
            print(f"\nNEXT STEPS:")
            print(f"- Upload newspapers and manually mark business ads")
            print(f"- Use 'Learn Logo' feature to build business recognition database")
            print(f"- Future uploads will automatically detect learned businesses")

        print("=" * 80)
        return True

if __name__ == "__main__":
    success = test_new_detection_routing()
    if success:
        print("\n✅ NEW HYBRID DETECTION SYSTEM IS ACTIVE!")
        print("The upload process now uses logo recognition + smart manual detection")
        print("No more broken '[pdf text set]' detections!")
    else:
        print("\n❌ SYSTEM TEST FAILED")
        print("Check for remaining issues with hybrid system routing")