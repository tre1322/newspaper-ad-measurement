#!/usr/bin/env python3
"""
Visual test to verify we're detecting actual business ads on page 1
"""

import os
import cv2
from app import app, db, Publication, Page, AdBox

def test_page_1_detection():
    """Test detection specifically on page 1 where business ads should be"""

    with app.app_context():
        print("VISUAL TEST: Page 1 Business Ad Detection")
        print("=" * 50)

        # Get test publication
        publication = Publication.query.order_by(Publication.id.desc()).first()
        page_1 = Page.query.filter_by(publication_id=publication.id, page_number=1).first()

        if not page_1:
            print("ERROR: Page 1 not found")
            return

        print(f"Testing page 1 (ID: {page_1.id})")

        # Load page 1 image
        image_filename = f"{publication.filename}_page_1.png"
        image_path = os.path.join('static', 'uploads', 'pages', image_filename)

        if not os.path.exists(image_path):
            print(f"ERROR: Page 1 image not found: {image_path}")
            return

        print(f"Loading image: {image_path}")
        page_image = cv2.imread(image_path)
        if page_image is None:
            print("ERROR: Could not load page image")
            return

        print(f"Image size: {page_image.shape}")

        # Get all bordered ads on page 1
        bordered_ads = AdBox.query.filter_by(
            page_id=page_1.id,
            ad_type='bordered_ad'
        ).order_by(AdBox.confidence_score.desc()).all()

        print(f"Found {len(bordered_ads)} bordered ads on page 1:")

        # Draw boxes on image and analyze locations
        output_image = page_image.copy()

        for i, ad in enumerate(bordered_ads):
            x, y, w, h = int(ad.x), int(ad.y), int(ad.width), int(ad.height)

            print(f"  Ad {i+1}: ({x}, {y}) {w}x{h} pixels, confidence={ad.confidence_score:.2f}")

            # Determine location on page
            page_height = page_image.shape[0]
            page_width = page_image.shape[1]

            # Analyze position
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

            print(f"    Location: {location}")

            # Draw rectangle on image
            color = (0, 255, 0) if ad.confidence_score > 0.7 else (0, 255, 255)  # Green for high confidence, yellow for medium
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output_image, f"Ad{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Save output image with detected boxes
        output_path = os.path.join('static', 'uploads', 'pages', f'page1_detected_ads.png')
        cv2.imwrite(output_path, output_image)
        print(f"\nSaved detection visualization: {output_path}")

        # Check if we're detecting business directory area (right side, middle)
        business_directory_ads = []
        for ad in bordered_ads:
            x, y, w, h = int(ad.x), int(ad.y), int(ad.width), int(ad.height)

            # Business directory is typically on right side, middle to bottom
            if (x > page_width * 0.6 and  # Right side
                y > page_height * 0.3 and y < page_height * 0.8 and  # Middle area
                w >= 150 and h >= 100):  # Reasonable ad size
                business_directory_ads.append(ad)

        print(f"\nBusiness directory candidates: {len(business_directory_ads)}")
        for ad in business_directory_ads:
            print(f"  Candidate: {ad.width}x{ad.height} at ({ad.x}, {ad.y})")

        if len(business_directory_ads) > 0:
            print("\nSUCCESS: Detecting ads in business directory area!")
            return True
        else:
            print("\nWARNING: No ads detected in expected business directory area")
            return False

if __name__ == "__main__":
    success = test_page_1_detection()

    print("\n" + "=" * 50)
    if success:
        print("SUCCESS: Business directory ads detected!")
        print("Check the generated page1_detected_ads.png file")
    else:
        print("ISSUE: May not be detecting business directory properly")
        print("Check the generated visualization to see what was detected")