#!/usr/bin/env python3
"""
Debug why logo learning is failing
"""

import os
from app import app, db, Publication, Page
from app import LogoFeatureExtractor, LogoLearningWorkflow
import cv2

def debug_logo_learning():
    """Debug logo learning issues"""

    with app.app_context():
        print("DEBUG: Logo Learning Issues")
        print("=" * 50)

        # Get test publication and page
        publication = Publication.query.order_by(Publication.id.desc()).first()
        page = Page.query.filter_by(publication_id=publication.id).first()

        print(f"Publication: {publication.original_filename}")
        print(f"Page: {page.page_number}")

        # Load page image
        image_filename = f"{publication.filename}_page_{page.page_number}.png"
        image_path = os.path.join('static', 'uploads', 'pages', image_filename)

        page_image = cv2.imread(image_path)
        if page_image is None:
            print("ERROR: Could not load page image")
            return

        print(f"Image loaded: {page_image.shape}")

        # Test feature extraction directly
        print("\nTesting feature extraction...")
        feature_extractor = LogoFeatureExtractor()

        # Extract a test region
        h, w = page_image.shape[:2]
        test_region = page_image[h//3:h//2, w//2:w//1]  # Right side region

        if test_region.size == 0:
            print("ERROR: Test region is empty")
            return

        print(f"Test region size: {test_region.shape}")

        # Extract features
        features = feature_extractor.extract_logo_features(test_region, "Test Business")

        if features:
            print("SUCCESS: Features extracted")
            print(f"Feature keys: {list(features.keys())}")

            if 'sift_features' in features:
                sift_data = features['sift_features']
                if sift_data and 'keypoints' in sift_data:
                    print(f"SIFT keypoints: {len(sift_data['keypoints'])}")
                else:
                    print("WARNING: No SIFT keypoints found")

            if 'color_histogram' in features:
                color_data = features['color_histogram']
                if color_data:
                    print(f"Color histogram: {len(color_data)} channels")
                else:
                    print("WARNING: No color histogram")

        else:
            print("ERROR: No features extracted")
            return

        # Test logo learning workflow
        print("\nTesting logo learning workflow...")
        learning_workflow = LogoLearningWorkflow()

        ad_coordinates = {
            'x': w//2,
            'y': h//3,
            'width': w//2,
            'height': h//6
        }

        # Test the learning process
        result = learning_workflow.analyze_manual_ad_for_logo_learning(
            page.id, ad_coordinates, "Test Cottonwood Vet"
        )

        if result.get('success'):
            print("SUCCESS: Logo learning workflow completed")
            print(f"Result: {result}")
        else:
            print(f"ERROR: Logo learning failed: {result.get('error', 'Unknown')}")

        print("\nDEBUG COMPLETE")

if __name__ == "__main__":
    debug_logo_learning()