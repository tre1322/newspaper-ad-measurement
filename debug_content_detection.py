#!/usr/bin/env python3
"""
DEBUG: See what text is extracted and why business scoring fails
"""
from app import app, db, ContentBasedAdDetector
import fitz

def debug_content_detection():
    """Debug content extraction and business scoring"""
    print("=== DEBUGGING CONTENT DETECTION ===")

    pdf_path = r"C:\Users\trevo\newspaper-ad-measurement\OA-2025-01-01.pdf"

    # Test just page 1 first
    page_num = 6  # Test page 6 since visual detector found ads there

    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_num - 1)

        # Extract text with detailed positioning
        text_dict = page.get_text("dict")

        # Extract all text blocks
        text_blocks = ContentBasedAdDetector._extract_text_blocks(text_dict)

        print(f"PAGE {page_num}: Found {len(text_blocks)} text blocks")
        print()

        # Show first 10 text blocks and their business scores
        for i, block in enumerate(text_blocks[:15]):
            print(f"BLOCK {i+1}:")
            print(f"  Position: ({block['x']:.0f}, {block['y']:.0f}) Size: {block['width']:.0f}x{block['height']:.0f}")

            # Clean text for printing
            clean_text = block['text'].encode('ascii', 'ignore').decode('ascii')
            print(f"  Text: '{clean_text[:100]}...'")

            # Score business content
            business_score, indicators = ContentBasedAdDetector._score_business_content(block['text'])
            print(f"  Business Score: {business_score:.3f}")
            print(f"  Indicators: {indicators}")

            if business_score > 0.3:  # Lower threshold for debugging
                print(f"  *** POTENTIAL BUSINESS CONTENT ***")

            print()

        # Look for any blocks with business content
        business_blocks = []
        for block in text_blocks:
            business_score, indicators = ContentBasedAdDetector._score_business_content(block['text'])
            if business_score > 0.1:  # Very low threshold
                business_blocks.append({
                    'text': block['text'],
                    'score': business_score,
                    'indicators': indicators,
                    'position': (block['x'], block['y'])
                })

        print(f"BUSINESS CONTENT BLOCKS (score > 0.1): {len(business_blocks)}")
        for i, block in enumerate(business_blocks):
            print(f"  {i+1}. Score: {block['score']:.3f}, Indicators: {block['indicators']}")
            clean_text = block['text'].encode('ascii', 'ignore').decode('ascii')
            print(f"     Text: '{clean_text[:80]}...'")
            print()

        # Test specific business text patterns
        print("=== TESTING BUSINESS PATTERNS ===")
        test_texts = [
            "Call 555-123-4567 for service",
            "Auto Repair - Licensed & Insured",
            "www.example.com - Free Estimates",
            "Open Mon-Fri 9:00-5:00",
            "123 Main Street, Auto Service",
            "$50 Oil Change Special"
        ]

        for text in test_texts:
            score, indicators = ContentBasedAdDetector._score_business_content(text)
            print(f"Text: '{text}'")
            print(f"Score: {score:.3f}, Indicators: {indicators}")
            print()

        doc.close()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_content_detection()