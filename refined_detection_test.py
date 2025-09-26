#!/usr/bin/env python3
"""
Test the refined SimpleAdDetector with improved filtering
Goal: Reduce detections from 143 to 30-50 high-quality business ads
"""

import os
import cv2
from app import app, db, Publication, Page, SimpleAdDetector

def test_refined_detection():
    """Test refined detection with all improvements"""

    with app.app_context():
        print("REFINED DETECTION TEST - Quality over Quantity")
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

        # Test refined detection directly on image
        print("Running refined border detection...")
        ad_regions = SimpleAdDetector._find_bordered_rectangles(page_image)
        print(f"Found {len(ad_regions)} potential ad regions")

        # Apply refined filtering
        print("Applying refined filtering (size, merge, business focus)...")
        filtered_ads = SimpleAdDetector._filter_realistic_ads(ad_regions)

        print(f"After refined filtering: {len(filtered_ads)} high-quality ads")
        print(f"Reduction: {len(ad_regions)} -> {len(filtered_ads)} (-{len(ad_regions) - len(filtered_ads)})")

        # Show details of all final candidates
        print(f"\nFINAL {len(filtered_ads)} HIGH-QUALITY BUSINESS AD CANDIDATES:")
        print("-" * 60)

        business_directory_count = 0
        for i, ad in enumerate(filtered_ads):
            x, y, w, h = ad['x'], ad['y'], ad['width'], ad['height']

            # Determine location
            page_height, page_width = page_image.shape[:2]
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

            # Check if it's in business directory area
            is_business_dir = (x > page_width * 0.6 and
                             y > page_height * 0.3 and
                             200 <= w <= 400 and 150 <= h <= 300)

            if is_business_dir:
                business_directory_count += 1
                print(f"  {i+1}: ({x}, {y}) {w}x{h}, conf={ad['confidence']:.2f}, {location} [BUSINESS DIR]")
            else:
                print(f"  {i+1}: ({x}, {y}) {w}x{h}, conf={ad['confidence']:.2f}, {location}")

        # Create refined visualization
        output_image = page_image.copy()

        for i, ad in enumerate(filtered_ads):
            x, y, w, h = ad['x'], ad['y'], ad['width'], ad['height']

            # Color code by confidence and type
            if ad['confidence'] > 0.8:
                color = (0, 255, 0)  # Bright green for high confidence
            elif ad['confidence'] > 0.6:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (255, 0, 255)  # Magenta for lower confidence

            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 3)
            cv2.putText(output_image, f"{i+1}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Save refined visualization
        output_path = os.path.join('static', 'uploads', 'pages', f'refined_detection_test.png')
        cv2.imwrite(output_path, output_image)
        print(f"\nSaved refined detection visualization: {output_path}")

        print(f"\nBusiness directory candidates: {business_directory_count}")

        # Check if we achieved our goal
        target_range = range(30, 51)  # 30-50 ads
        if len(filtered_ads) in target_range:
            print(f"\nSUCCESS: Achieved target range of 30-50 ads ({len(filtered_ads)} detected)")
            print("Quality filtering working properly!")
        elif len(filtered_ads) < 30:
            print(f"\nWARNING: Below target range ({len(filtered_ads)} < 30)")
            print("May need to relax filtering slightly")
        else:
            print(f"\nWARNING: Above target range ({len(filtered_ads)} > 50)")
            print("May need stricter filtering")

        return len(filtered_ads) in target_range

if __name__ == "__main__":
    success = test_refined_detection()

    print("\n" + "=" * 60)
    if success:
        print("SUCCESS: Refined detection achieved quality goals!")
        print("Check refined_detection_test.png for results")
    else:
        print("NEEDS ADJUSTMENT: Check results and fine-tune filtering")