#!/usr/bin/env python3
"""
Test the new PDF structure detection system
"""

import os
import sys
from pdf_structure_analyzer import PDFStructureAdDetector

def test_pdf_structure_detection():
    """Test the new PDF structure detection on the problematic newspaper"""

    # Try different PDFs
    pdf_candidates = [
        "./static/uploads/pdfs/0065da3f-f1a1-4ad3-9122-b66d349bb942.pdf",
        "./static/uploads/pdfs/041e9aba-830d-4b3b-947c-335a003579d8.pdf",
        "./static/uploads/pdfs/0e155c5c-9db0-4056-916f-d50f8cabe1c0.pdf"
    ]

    pdf_path = None
    for candidate in pdf_candidates:
        if os.path.exists(candidate):
            # Quick check if PDF has pages
            try:
                import fitz
                doc = fitz.open(candidate)
                if len(doc) > 0:
                    pdf_path = candidate
                    print(f"Using PDF: {candidate} (has {len(doc)} pages)")
                    doc.close()
                    break
                else:
                    print(f"Skipping {candidate} - has 0 pages")
                doc.close()
            except:
                print(f"Skipping {candidate} - failed to open")
                continue

    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        print("Please update the pdf_path to point to your newspaper PDF")
        return False

    print("Testing NEW PDF Structure Analysis for Ad Detection")
    print("=" * 60)

    try:
        # Test the new detection on page 1
        page_number = 1
        publication_type = 'broadsheet'  # or 'tabloid'

        print(f"Analyzing page {page_number}...")

        # Run the new PDF structure analysis
        detected_ads = PDFStructureAdDetector.detect_ads_from_pdf_structure(
            pdf_path, page_number, publication_type
        )

        print(f"\nRESULTS:")
        print(f"Found {len(detected_ads)} potential ads")

        for i, ad in enumerate(detected_ads):
            print(f"  Ad {i+1}: {ad['type']} - {ad['width']:.0f}x{ad['height']:.0f}px")
            print(f"         Position: ({ad['x']:.0f}, {ad['y']:.0f})")
            print(f"         Confidence: {ad['confidence']:.2f}")
            print(f"         Classification: {ad.get('classification', 'unknown')}")
            print()

        if len(detected_ads) > 0:
            print("SUCCESS: New PDF structure analysis detected ads!")
            print("This should replace the broken sliding window approach.")
        else:
            print("No ads detected - may need parameter tuning.")

        return True

    except Exception as e:
        print(f"ERROR in PDF structure detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_app():
    """Test integration with the app's detection function"""
    from app import PDFMetadataAdDetector

    # Find a valid PDF
    import fitz
    pdf_candidates = [
        "./static/uploads/pdfs/0065da3f-f1a1-4ad3-9122-b66d349bb942.pdf",
        "./static/uploads/pdfs/041e9aba-830d-4b3b-947c-335a003579d8.pdf",
        "./static/uploads/pdfs/0e155c5c-9db0-4056-916f-d50f8cabe1c0.pdf"
    ]

    pdf_path = None
    for candidate in pdf_candidates:
        if os.path.exists(candidate):
            try:
                doc = fitz.open(candidate)
                if len(doc) > 0:
                    pdf_path = candidate
                    print(f"Using PDF for integration test: {candidate} (has {len(doc)} pages)")
                    doc.close()
                    break
                doc.close()
            except:
                continue

    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return False

    print("\nTesting Integration with App...")
    print("=" * 40)

    try:
        # Test the updated function in app.py
        detected_ads = PDFMetadataAdDetector.detect_ads_from_pdf_metadata(
            pdf_path, 1, 'broadsheet', 'test_file.pdf'
        )

        print(f"App integration found {len(detected_ads)} ads")

        for i, ad in enumerate(detected_ads):
            print(f"  Ad {i+1}: {ad.get('ad_type', 'unknown')} - {ad['width']:.0f}x{ad['height']:.0f}px")
            print(f"         Confidence: {ad['confidence']:.2f}")

        return True

    except Exception as e:
        print(f"ERROR in app integration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("PDF Structure Analysis Test")
    print("Testing replacement for broken sliding window detection")
    print()

    # Test 1: Direct PDF structure analysis
    success1 = test_pdf_structure_detection()

    # Test 2: Integration with app
    success2 = test_integration_with_app()

    if success1 and success2:
        print("\nALL TESTS PASSED!")
        print("New PDF structure analysis is working correctly.")
    else:
        print("\nSome tests failed. Check the error messages above.")