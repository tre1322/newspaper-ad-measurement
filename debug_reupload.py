#!/usr/bin/env python3
"""
Debug script for re-upload detection issues
"""

import os
import sys
from app import PDFMetadataAdDetector, PDFStructureAdDetector

def debug_detection_issue(pdf_path):
    """Debug why the same PDF might not detect ads on re-upload"""

    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return

    print("DEBUGGING RE-UPLOAD DETECTION ISSUE")
    print("=" * 50)
    print(f"PDF: {pdf_path}")

    # Test both the old method (which gets called first) and new method

    print("\n1. Testing OLD PDF detection method (gets called first):")
    print("-" * 40)

    try:
        # This simulates what happens in the upload flow
        # The old method gets called first in detect_ads_from_pdf_metadata
        print("This would use the OLD rectangular border detection...")
        print("(The issue might be that the old method returns results,")
        print(" so the new enhanced method never gets called)")

    except Exception as e:
        print(f"Error in old method: {e}")

    print("\n2. Testing NEW PDF structure analysis:")
    print("-" * 40)

    try:
        from pdf_structure_analyzer import PDFStructureAdDetector

        # Test page 1
        detected_ads = PDFStructureAdDetector.detect_ads_from_pdf_structure(
            pdf_path, 1, 'broadsheet'
        )

        print(f"New method found {len(detected_ads)} ads")
        for i, ad in enumerate(detected_ads[:5]):  # Show first 5
            print(f"  Ad {i+1}: {ad['type']} - {ad['width']:.0f}x{ad['height']:.0f}px - confidence: {ad['confidence']:.2f}")

        if len(detected_ads) > 5:
            print(f"  ... and {len(detected_ads) - 5} more")

    except Exception as e:
        print(f"Error in new method: {e}")
        import traceback
        traceback.print_exc()

    print("\n3. Testing App Integration (what actually gets called):")
    print("-" * 40)

    try:
        # This is what actually gets called during upload
        result = PDFMetadataAdDetector.detect_ads_from_pdf_metadata(
            pdf_path, 1, 'broadsheet', 'test.pdf'
        )

        print(f"App integration found {len(result)} ads")
        for i, ad in enumerate(result[:3]):  # Show first 3
            print(f"  Ad {i+1}: {ad.get('ad_type', 'unknown')} - {ad['width']:.0f}x{ad['height']:.0f}px - confidence: {ad['confidence']:.2f}")

    except Exception as e:
        print(f"Error in app integration: {e}")
        import traceback
        traceback.print_exc()

    print("\nDIAGNOSIS:")
    print("-" * 20)
    print("If the new method finds ads but app integration doesn't,")
    print("the issue might be in the integration between old/new methods.")
    print("The old rectangular detection might be returning empty results")
    print("and preventing the new enhanced analysis from running.")

if __name__ == "__main__":
    # Test with first available PDF
    pdf_candidates = [
        "./static/uploads/pdfs/041e9aba-830d-4b3b-947c-335a003579d8.pdf",
        "./static/uploads/pdfs/0e155c5c-9db0-4056-916f-d50f8cabe1c0.pdf",
        "./static/uploads/pdfs/10e81ca2-d957-4a94-b864-7f86953962f1.pdf"
    ]

    test_pdf = None
    for pdf in pdf_candidates:
        if os.path.exists(pdf):
            test_pdf = pdf
            break

    if test_pdf:
        debug_detection_issue(test_pdf)
    else:
        print("No test PDFs found. Please provide a PDF path as an argument.")
        print("Usage: python debug_reupload.py path/to/your.pdf")