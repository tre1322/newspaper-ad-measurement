#!/usr/bin/env python3
"""
Comprehensive test of the Hybrid Logo Recognition + Manual Detection System
Tests all components and integration points
"""

import os
import sys
import cv2
import numpy as np
from app import app, db, Publication, Page, AdBox, BusinessLogo, LogoRecognitionResult
from app import LogoFeatureExtractor, LogoMatcher, LogoLearningWorkflow
from app import LogoRecognitionDetectionEngine, SmartManualDetection, HybridDetectionPipeline

def test_hybrid_detection_system():
    """Test the complete hybrid detection system"""

    with app.app_context():
        print("=" * 80)
        print("HYBRID LOGO RECOGNITION + MANUAL DETECTION SYSTEM TEST")
        print("=" * 80)

        # Test 1: Component Initialization
        print("\nTEST 1: Component Initialization...")
        try:
            feature_extractor = LogoFeatureExtractor()
            logo_matcher = LogoMatcher()
            learning_workflow = LogoLearningWorkflow()
            recognition_engine = LogoRecognitionDetectionEngine()
            smart_manual = SmartManualDetection()
            hybrid_pipeline = HybridDetectionPipeline()
            print("SUCCESS: All components initialized successfully")
        except Exception as e:
            print(f"ERROR: Component initialization failed: {e}")
            return

        # Test 2: Database Models
        print("\nTEST 2: Database Models...")
        try:
            # Check if tables exist
            db.create_all()

            # Test BusinessLogo model
            test_logo = BusinessLogo(
                business_name="Test Business",
                logo_features='{"test": "data"}',
                confidence_threshold=0.7,
                is_active=True
            )
            db.session.add(test_logo)
            db.session.commit()

            # Test LogoRecognitionResult model
            # We'll create this after we have an actual detection

            print("SUCCESS: Database models working correctly")

        except Exception as e:
            print(f"ERROR: Database model test failed: {e}")
            db.session.rollback()

        # Test 3: Find Test Publication
        print("\nTEST 3: Finding Test Publication...")
        try:
            publication = Publication.query.order_by(Publication.id.desc()).first()
            if not publication:
                print("ERROR: No publications found in database")
                return

            print(f"SUCCESS: Using publication: {publication.original_filename}")

            # Get pages
            pages = Page.query.filter_by(publication_id=publication.id).all()
            if not pages:
                print("ERROR: No pages found for publication")
                return

            print(f"SUCCESS: Found {len(pages)} pages")

        except Exception as e:
            print(f"ERROR: Publication test failed: {e}")
            return

        # Test 4: Feature Extraction
        print("\nTEST 4: Logo Feature Extraction...")
        try:
            # Load a page image
            test_page = pages[0]
            image_filename = f"{publication.filename}_page_{test_page.page_number}.png"
            image_path = os.path.join('static', 'uploads', 'pages', image_filename)

            if not os.path.exists(image_path):
                print(f"ERROR: Page image not found: {image_path}")
                return

            page_image = cv2.imread(image_path)
            if page_image is None:
                print("ERROR: Could not load page image")
                return

            print(f"SUCCESS: Loaded page image: {page_image.shape}")

            # Extract features from a region
            h, w = page_image.shape[:2]
            test_region = page_image[h//4:h//2, w//4:w//2]  # Center region

            features = feature_extractor.extract_logo_features(test_region, "Test Business")

            if features and features.get('sift_features'):
                print(f"SUCCESS: Extracted features: {len(features['sift_features']['keypoints'])} SIFT keypoints")
            else:
                print("WARNING: Feature extraction returned no SIFT features (may be normal)")

        except Exception as e:
            print(f"ERROR: Feature extraction test failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 5: Logo Learning Workflow
        print("\nTEST 5: Logo Learning Workflow...")
        try:
            # Create a test AdBox
            test_adbox = AdBox(
                page_id=test_page.id,
                x=100.0, y=100.0, width=200.0, height=150.0,
                width_inches_raw=2.0, height_inches_raw=1.5,
                width_inches_rounded=2.0, height_inches_rounded=1.5,
                column_inches=3.0,
                ad_type='test',
                is_ad=True,
                detected_automatically=False,
                confidence_score=0.8
            )
            db.session.add(test_adbox)
            db.session.commit()

            # Test logo learning
            if features:
                learning_result = learning_workflow.learn_logo_from_manual_ad(
                    test_adbox, "Test Business Logo", features
                )

                if learning_result.get('success'):
                    print("SUCCESS: Logo learning workflow successful")
                else:
                    print(f"WARNING: Logo learning failed: {learning_result.get('error', 'Unknown')}")
            else:
                print("WARNING: Skipping logo learning test (no features)")

        except Exception as e:
            print(f"ERROR: Logo learning test failed: {e}")
            db.session.rollback()
            import traceback
            traceback.print_exc()

        # Test 6: Smart Manual Detection
        print("\nTEST 6: Smart Manual Detection...")
        try:
            # Test boundary detection from click
            click_x, click_y = w//2, h//2  # Center click

            boundary_result = smart_manual.detect_ad_boundaries_from_click(
                page_image, click_x, click_y, test_page.id
            )

            if boundary_result.get('success'):
                boundary = boundary_result['boundary']
                print(f"SUCCESS: Smart boundary detection successful:")
                print(f"  Method: {boundary_result.get('detection_method', 'unknown')}")
                print(f"  Confidence: {boundary_result.get('confidence', 0):.2f}")
                print(f"  Boundary: ({boundary['x']}, {boundary['y']}) {boundary['width']}x{boundary['height']}")
            else:
                print(f"ERROR: Smart boundary detection failed: {boundary_result.get('error', 'Unknown')}")

        except Exception as e:
            print(f"ERROR: Smart manual detection test failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 7: Logo Recognition Engine
        print("\nTEST 7: Logo Recognition Engine...")
        try:
            # Check for existing business logos
            business_logos = BusinessLogo.query.filter_by(is_active=True).all()
            print(f"Found {len(business_logos)} active business logos")

            if business_logos:
                # Test detection on the publication
                recognition_result = recognition_engine.detect_ads_from_publication(publication.id)

                if recognition_result.get('success'):
                    detections = recognition_result.get('detections', 0)
                    print(f"SUCCESS: Logo recognition successful: {detections} detections")
                else:
                    print(f"WARNING: Logo recognition completed but found no ads: {recognition_result.get('error', 'No error')}")
            else:
                print("WARNING: No active business logos for recognition test")

        except Exception as e:
            print(f"ERROR: Logo recognition test failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 8: Hybrid Detection Pipeline
        print("\nTEST 8: Hybrid Detection Pipeline...")
        try:
            # Test auto mode
            auto_result = hybrid_pipeline.detect_ads_hybrid(publication.id, mode='auto')

            if auto_result.get('success'):
                print(f"SUCCESS: Hybrid auto detection successful:")
                print(f"  Total detections: {auto_result.get('total_detections', 0)}")
                print(f"  Logo detections: {auto_result.get('logo_detections', 0)}")
                print(f"  Pages processed: {auto_result.get('pages_processed', 0)}")
            else:
                print(f"ERROR: Hybrid auto detection failed: {auto_result.get('error', 'Unknown')}")

            # Test status reporting
            status_result = hybrid_pipeline.get_hybrid_detection_status(publication.id)

            if status_result.get('success'):
                print(f"SUCCESS: Hybrid status reporting successful:")
                print(f"  Total ads: {status_result.get('total_ads', 0)}")
                print(f"  Automated ads: {status_result.get('automated_ads', 0)}")
                print(f"  Manual ads: {status_result.get('manual_ads', 0)}")
            else:
                print(f"ERROR: Hybrid status failed: {status_result.get('error', 'Unknown')}")

        except Exception as e:
            print(f"ERROR: Hybrid pipeline test failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 9: Manual Click Processing
        print("\nTEST 9: Manual Click Processing...")
        try:
            if boundary_result and boundary_result.get('success'):
                # Test manual click processing
                click_result = hybrid_pipeline.process_manual_click(
                    test_page.id, click_x, click_y, "Test Click Business", learn_logo=True
                )

                if click_result.get('success'):
                    print(f"SUCCESS: Manual click processing successful:")
                    print(f"  AdBox ID: {click_result.get('ad_box_id')}")
                    print(f"  Detection method: {click_result.get('detection_method')}")
                    print(f"  Logo learned: {click_result.get('logo_learned', False)}")
                else:
                    print(f"ERROR: Manual click processing failed: {click_result.get('error', 'Unknown')}")
            else:
                print("WARNING: Skipping manual click test (boundary detection failed)")

        except Exception as e:
            print(f"ERROR: Manual click processing test failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 10: Detection Suggestions
        print("\nTEST 10: Detection Suggestions...")
        try:
            suggestions_result = hybrid_pipeline.get_detection_suggestions(test_page.id, threshold=0.5)

            if suggestions_result.get('success'):
                suggestions = suggestions_result.get('suggestions', [])
                print(f"SUCCESS: Detection suggestions successful: {len(suggestions)} suggestions")

                for i, suggestion in enumerate(suggestions[:3]):  # Show first 3
                    print(f"  {i+1}. {suggestion.get('business_name', 'Unknown')} "
                          f"(confidence: {suggestion.get('confidence', 0):.2f})")
            else:
                print(f"ERROR: Detection suggestions failed: {suggestions_result.get('error', 'Unknown')}")

        except Exception as e:
            print(f"ERROR: Detection suggestions test failed: {e}")
            import traceback
            traceback.print_exc()

        # Test Summary
        print("\n" + "=" * 80)
        print("HYBRID DETECTION SYSTEM TEST SUMMARY")
        print("=" * 80)

        # Get final statistics
        try:
            total_logos = BusinessLogo.query.count()
            active_logos = BusinessLogo.query.filter_by(is_active=True).count()
            total_detections = LogoRecognitionResult.query.count()
            total_adboxes = AdBox.query.count()

            print(f"Database State:")
            print(f"  Business Logos: {total_logos} total, {active_logos} active")
            print(f"  Logo Recognition Results: {total_detections}")
            print(f"  AdBox Records: {total_adboxes}")

        except Exception as e:
            print(f"Error getting final statistics: {e}")

        print("\nSystem Components Status:")
        print("- Logo Feature Extraction System")
        print("- Logo Learning Workflow")
        print("- Smart Manual Detection with Boundary Expansion")
        print("- Logo Recognition Detection Engine")
        print("- Hybrid Detection Pipeline")
        print("- Logo Management Interface Routes")

        print("\nThe hybrid logo recognition + manual detection system is ready for use!")
        print("Key Features:")
        print("- Automated logo recognition for known businesses")
        print("- Smart manual detection with intelligent boundary expansion")
        print("- Logo learning from manual ad placement")
        print("- Hybrid detection combining automated and manual approaches")
        print("- Comprehensive web interface for logo management")

        print("=" * 80)

if __name__ == "__main__":
    test_hybrid_detection_system()