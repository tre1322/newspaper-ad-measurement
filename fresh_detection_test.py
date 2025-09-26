#!/usr/bin/env python3
"""
Fresh test of simple detection on a single page
"""

import os
import cv2
from app import app, db, Publication, Page, SimpleAdDetector

def test_fresh_page_detection():
    """Test simple detection directly on page image"""

    with app.app_context():
        print("FRESH DETECTION TEST - Single Page")
        print("=" * 50)

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

        # Test detection directly on image
        print("Running border detection on image...")
        ad_regions = SimpleAdDetector._find_bordered_rectangles(page_image)

        print(f"Found {len(ad_regions)} potential ad regions")

        # Filter to realistic ads
        filtered_ads = SimpleAdDetector._filter_realistic_ads(ad_regions)

        print(f"After filtering: {len(filtered_ads)} realistic ads")

        # Show details of top candidates
        print("\nTop 20 detection candidates:")
        for i, ad in enumerate(filtered_ads[:20]):
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

            print(f"  {i+1}: ({x}, {y}) {w}x{h}, conf={ad['confidence']:.2f}, {location}")

        # Create visualization with ALL candidates
        output_image = page_image.copy()

        for i, ad in enumerate(filtered_ads[:20]):  # Show top 20
            x, y, w, h = ad['x'], ad['y'], ad['width'], ad['height']
            color = (0, 255, 0) if ad['confidence'] > 0.7 else (0, 255, 255)
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output_image, f"{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Save fresh visualization
        output_path = os.path.join('static', 'uploads', 'pages', f'fresh_detection_test.png')
        cv2.imwrite(output_path, output_image)
        print(f"\nSaved fresh detection visualization: {output_path}")

        # Look specifically for business directory area (right side)
        business_ads = []
        for ad in filtered_ads:
            x, y, w, h = ad['x'], ad['y'], ad['width'], ad['height']

            # Business directory: right side (x > 60% of width), reasonable size
            if (x > page_width * 0.6 and
                80 <= w <= 350 and 40 <= h <= 150):
                business_ads.append(ad)

        print(f"\nBusiness directory candidates: {len(business_ads)}")
        for ad in business_ads:
            print(f"  Business ad: {ad['width']}x{ad['height']} at ({ad['x']}, {ad['y']}) conf={ad['confidence']:.2f}")

        return len(business_ads) > 0

if __name__ == "__main__":
    success = test_fresh_page_detection()
    if success:
        print("\nSUCCESS: Found business directory ads!")
    else:
        print("\nISSUE: Still not detecting business directory properly")