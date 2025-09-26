#!/usr/bin/env python3
"""
Test to PROVE editorial photos are excluded from detection
Focus: Sports photos, news photos, editorial content should NOT be detected as ads
"""

import os
import cv2
from app import app, db, Publication, Page, SimpleAdDetector

def test_editorial_exclusion():
    """Test that editorial photos are NOT detected as ads"""

    with app.app_context():
        print("EDITORIAL EXCLUSION TEST - Proving Photos Are NOT Ads")
        print("=" * 60)

        # Get test publication
        publication = Publication.query.order_by(Publication.id.desc()).first()
        page_1 = Page.query.filter_by(publication_id=publication.id, page_number=1).first()

        # Load page 1 image directly
        image_filename = f"{publication.filename}_page_1.png"
        image_path = os.path.join('static', 'uploads', 'pages', image_filename)

        print(f"Loading image: {image_path}")
        page_image = cv2.imread(image_path)
        if page_image is None:
            print("ERROR: Could not load page image")
            return

        print(f"Image size: {page_image.shape}")

        # Test NEW content-aware detection
        print("Running NEW content-aware detection (excludes editorial photos)...")
        ad_regions = SimpleAdDetector._find_bordered_rectangles(page_image)
        print(f"Found {len(ad_regions)} potential ad regions (AFTER editorial exclusion)")

        # Apply refined filtering
        print("Applying content filtering and merging...")
        filtered_ads = SimpleAdDetector._filter_realistic_ads(ad_regions)

        print(f"Final business ads detected: {len(filtered_ads)}")

        # Analyze what was detected
        print(f"\nDETECTED REGIONS (should be BUSINESS ADS only, NO editorial photos):")
        print("-" * 60)

        page_height, page_width = page_image.shape[:2]

        for i, ad in enumerate(filtered_ads):
            x, y, w, h = ad['x'], ad['y'], ad['width'], ad['height']

            # Determine location
            location = ""
            if y < page_height * 0.3:
                location += "top "
            elif y > page_height * 0.7:
                location += "bottom "
            else:
                location += "middle "

            if x < page_width * 0.3:
                location += "left"
            elif x > page_width * 0.7:
                location += "right"
            else:
                location += "center"

            print(f"  {i+1}: ({x}, {y}) {w}x{h}, conf={ad['confidence']:.2f}, {location}")
            print(f"      Aspect ratio: {ad['aspect_ratio']:.2f}")

        # Create visualization with clear labeling
        output_image = page_image.copy()

        for i, ad in enumerate(filtered_ads):
            x, y, w, h = ad['x'], ad['y'], ad['width'], ad['height']

            # Use bright green for business ads (should be the only detections)
            color = (0, 255, 0)  # Green = Business Ad

            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 4)
            cv2.putText(output_image, f"BIZ-{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        # Save editorial exclusion test
        output_path = os.path.join('static', 'uploads', 'pages', f'editorial_exclusion_test.png')
        cv2.imwrite(output_path, output_image)
        print(f"\nSaved editorial exclusion test: {output_path}")

        # VERIFICATION CHECKS
        print(f"\nVERIFICATION RESULTS:")
        print("-" * 40)

        # Check 1: No detections in photo areas (rough estimates)
        photo_detections = 0
        for ad in filtered_ads:
            x, y, w, h = ad['x'], ad['y'], ad['width'], ad['height']

            # Check if detection overlaps with known photo areas
            # Top-left: sports/news photos typically here
            if (x < page_width * 0.6 and y < page_height * 0.5 and
                w > 200 and h > 150 and ad['aspect_ratio'] > 1.2 and ad['aspect_ratio'] < 1.8):
                photo_detections += 1
                print(f"  WARNING: Possible photo detection at ({x}, {y}) {w}x{h}")

        if photo_detections == 0:
            print(f"  ✓ SUCCESS: No editorial photos detected as ads")
        else:
            print(f"  ✗ FAILURE: {photo_detections} possible editorial photos detected")

        # Check 2: Reasonable detection count
        if 2 <= len(filtered_ads) <= 20:
            print(f"  ✓ SUCCESS: Reasonable ad count ({len(filtered_ads)} ads)")
        else:
            print(f"  ✗ WARNING: Unusual ad count ({len(filtered_ads)} ads)")

        # Check 3: Business-like aspect ratios
        business_ratios = 0
        for ad in filtered_ads:
            if 1.5 <= ad['aspect_ratio'] <= 4.0:  # Business ad ratios
                business_ratios += 1

        if business_ratios == len(filtered_ads):
            print(f"  ✓ SUCCESS: All detections have business ad aspect ratios")
        else:
            print(f"  ✗ WARNING: {len(filtered_ads) - business_ratios} detections have photo-like ratios")

        return photo_detections == 0

if __name__ == "__main__":
    success = test_editorial_exclusion()

    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: Editorial photos properly excluded from ad detection!")
        print("Check editorial_exclusion_test.png - should show ONLY business ads")
    else:
        print("FAILURE: Still detecting editorial photos as ads")
        print("Need to strengthen content filtering rules")