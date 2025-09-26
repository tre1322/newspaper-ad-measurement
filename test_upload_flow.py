#!/usr/bin/env python3
"""
Test the complete upload flow to ensure the fix works in the real application
"""

import os
import requests
import time

def test_upload_flow():
    """Test the upload flow with a PDF to verify AI detection works"""

    print("=== TESTING UPLOAD FLOW WITH CONFIDENCE THRESHOLD FIX ===")
    print()

    # Find a test PDF
    test_pdfs = [
        "./static/uploads/pdfs/041e9aba-830d-4b3b-947c-335a003579d8.pdf",
        "./static/uploads/pdfs/0e155c5c-9db0-4056-916f-d50f8cabe1c0.pdf",
        "./static/uploads/pdfs/10e81ca2-d957-4a94-b864-7f86953962f1.pdf"
    ]

    test_pdf = None
    for pdf in test_pdfs:
        if os.path.exists(pdf):
            test_pdf = pdf
            break

    if not test_pdf:
        print("[ERROR] No test PDF found")
        return

    print(f"[INFO] Using test PDF: {test_pdf}")
    print()

    # Start the Flask app in the background
    print("[INFO] Starting Flask application...")
    print("[INFO] Please manually:")
    print("       1. Run: python app.py")
    print("       2. Open browser to http://localhost:5000")
    print("       3. Upload the same publication you used before")
    print("       4. Check if ads are auto-detected")
    print("       5. The confidence threshold is now 0.1 instead of 0.25")
    print()
    print("[EXPECTED RESULT] You should now see ads automatically detected during upload!")
    print("[EXPECTED RESULT] The system should display: 'AI detection complete: X ads automatically detected'")

if __name__ == "__main__":
    test_upload_flow()