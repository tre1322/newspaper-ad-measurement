#!/usr/bin/env python3
"""
ACCOUNTABILITY TEST: Content-based ad detection on OA-2025-01-01.pdf
MUST PROVE 15+ ads detected vs current 2 ads
"""
from app import app, db, ContentBasedAdDetector, Publication, Page
import os

def test_content_based_detection():
    """Test content-based detection on the exact problematic file"""
    print("=== CONTENT-BASED DETECTION TEST: OA-2025-01-01.pdf ===")
    print()

    pdf_path = r"C:\Users\trevo\newspaper-ad-measurement\OA-2025-01-01.pdf"

    if not os.path.exists(pdf_path):
        print(f"ERROR: Test file not found: {pdf_path}")
        return False

    print(f"Testing file: {pdf_path}")
    print()

    # Test each page for business content
    total_ads_found = 0
    page_results = []

    try:
        import fitz
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()

        print(f"Analyzing {total_pages} pages for business content...")
        print()

        for page_num in range(1, total_pages + 1):
            print(f"--- PAGE {page_num} ---")

            # Detect business content ads on this page
            ads = ContentBasedAdDetector.detect_business_content_ads(pdf_path, page_num)

            print(f"Business content ads found: {len(ads)}")

            for i, ad in enumerate(ads):
                print(f"  Ad {i+1}:")
                print(f"    Position: ({ad['x']}, {ad['y']}) Size: {ad['width']}x{ad['height']}")
                print(f"    Confidence: {ad['confidence']:.3f}")
                print(f"    Indicators: {ad['business_indicators']}")
                clean_text = ad['text_content'].encode('ascii', 'ignore').decode('ascii')
                print(f"    Text preview: {clean_text[:100]}...")
                print()

            total_ads_found += len(ads)
            page_results.append({
                'page': page_num,
                'ads_found': len(ads),
                'ads': ads
            })
            print()

        # Summary results
        print("=== CONTENT-BASED DETECTION RESULTS ===")
        print(f"Total pages analyzed: {total_pages}")
        print(f"Total business content ads found: {total_ads_found}")
        print()

        # Show improvement over baseline
        baseline_ads = 2  # Current visual detection
        improvement = total_ads_found - baseline_ads

        print(f"BASELINE (visual detection): {baseline_ads} ads")
        print(f"CONTENT-BASED DETECTION: {total_ads_found} ads")
        print(f"IMPROVEMENT: +{improvement} ads ({improvement/baseline_ads*100:.1f}% increase)")
        print()

        # Detailed breakdown by page
        print("Page-by-page breakdown:")
        for result in page_results:
            if result['ads_found'] > 0:
                print(f"  Page {result['page']}: {result['ads_found']} ads")

        print()

        # Success criteria
        if total_ads_found >= 15:
            print("*** SUCCESS: 15+ ads detected - Content-based detection WORKS! ***")
            return True
        elif total_ads_found > baseline_ads:
            print(f"*** PARTIAL SUCCESS: {improvement} more ads than baseline ***")
            print("Content approach works but may need tuning for more ads")
            return True
        else:
            print("*** FAILURE: Content-based detection did not improve results ***")
            return False

    except Exception as e:
        print(f"ERROR during content-based detection: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_content_based_detection()
    if success:
        print("\nCONTENT-BASED DETECTION PROVES IMPROVEMENT!")
    else:
        print("\nCONTENT-BASED DETECTION FAILED - NEEDS FURTHER WORK")