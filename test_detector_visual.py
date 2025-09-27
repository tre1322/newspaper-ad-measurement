#!/usr/bin/env python3
"""
VISUAL TEST: See exactly what NewspaperDomainDetector detects on each page
"""
from newspaper_domain_detector import NewspaperDomainDetector
import fitz

def test_detector_visual():
    """Test what the detector actually finds on each page"""
    print("=== VISUAL TEST: NEWSPAPER DOMAIN DETECTOR ===")

    detector = NewspaperDomainDetector()
    pdf_path = "C:\\Users\\trevo\\newspaper-ad-measurement\\OA-2025-01-01.pdf"

    # Test each page individually
    for page_num in range(1, 11):
        print(f"\n=== PAGE {page_num} ===")
        ads = detector.detect_business_ads(pdf_path, page_num)
        print(f"Detected {len(ads)} ads")

        if len(ads) > 0:
            # Show details of each ad
            for i, ad in enumerate(ads):
                print(f"  Ad {i+1}:")
                print(f"    Position: ({ad.x:.1f}, {ad.y:.1f})")
                print(f"    Size: {ad.width:.1f} x {ad.height:.1f}")
                print(f"    Confidence: {ad.confidence:.1f}")
                print(f"    Type: {ad.ad_type}")
                print(f"    Indicators: {', '.join(ad.business_indicators)}")
                print(f"    Text snippet: {ad.text_snippet}")

                # Check if this looks like random editorial content
                if any(word in ad.text_snippet.lower() for word in ['reporter', 'editor', 'staff', 'byline', 'meanwhile', 'story', 'article']):
                    print(f"    WARNING: Might be editorial content!")

                # Check if too small to be a real ad
                if ad.width < 50 or ad.height < 20:
                    print(f"    WARNING: Very small - might be random text!")

                print()

        # Also check what text the detector is looking at
        if page_num in [1, 2, 7, 10]:  # Editorial pages
            print(f"  Editorial page - should have ZERO ads")
            if len(ads) > 0:
                print(f"  ERROR: Found {len(ads)} ads on editorial page!")
        elif page_num in [4, 6, 9]:  # Business/classified pages
            print(f"  Business/classified page - should have multiple ads")
            if len(ads) == 0:
                print(f"  ERROR: Found NO ads on business page!")

    # Summary
    total_ads = 0
    editorial_page_ads = 0
    business_page_ads = 0

    for page_num in range(1, 11):
        ads = detector.detect_business_ads(pdf_path, page_num)
        total_ads += len(ads)

        if page_num in [1, 2, 7, 10]:  # Editorial pages
            editorial_page_ads += len(ads)
        elif page_num in [4, 6, 9]:  # Business pages
            business_page_ads += len(ads)

    print(f"\n=== SUMMARY ===")
    print(f"Total ads detected: {total_ads}")
    print(f"Ads on editorial pages (1,2,7,10): {editorial_page_ads} - Should be 0!")
    print(f"Ads on business pages (4,6,9): {business_page_ads}")

    # Check specific problematic patterns
    print(f"\n=== CHECKING FOR COMMON PROBLEMS ===")

    # Check if it's detecting photos/images as ads
    doc = fitz.open(pdf_path)
    for page_num in [1, 2]:  # Editorial pages with photos
        print(f"\nPage {page_num} (editorial with photos):")
        ads = detector.detect_business_ads(pdf_path, page_num)
        for ad in ads:
            # Check if this ad overlaps with an image area
            print(f"  Detected ad at ({ad.x:.0f},{ad.y:.0f}) - {ad.text_snippet[:50]}...")

    doc.close()

    if editorial_page_ads == 0 and business_page_ads >= 30:
        print(f"\nSUCCESS: Clean detection - no editorial false positives")
        return True
    else:
        print(f"\nFAILURE: Still detecting editorial content as ads")
        return False

if __name__ == "__main__":
    success = test_detector_visual()
    if success:
        print("\nDETECTOR WORKING CORRECTLY")
    else:
        print("\nDETECTOR STILL HAS PROBLEMS")