#!/usr/bin/env python3
"""
DEBUG UPLOAD: Find exact validation failure
"""
import requests
import os

def debug_upload():
    """Debug the upload request to find exact failure"""
    print("=== DEBUG UPLOAD REQUEST ===")

    # Test the exact request from ACTUAL_TEST.py
    session = requests.Session()

    # Login first
    login_data = {'password': 'CCCitizen56101!'}
    login_response = session.post('http://localhost:5000/login', data=login_data)
    print(f"Login status: {login_response.status_code}")

    pdf_path = "C:\\Users\\trevo\\newspaper-ad-measurement\\OA-2025-01-01.pdf"
    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        return

    print(f"File exists: {pdf_path}")
    print(f"File size: {os.path.getsize(pdf_path)} bytes")

    # Prepare upload data - exact same as ACTUAL_TEST.py
    upload_data = {
        'publication_type': 'broadsheet'
    }

    print(f"Upload data: {upload_data}")

    try:
        with open(pdf_path, 'rb') as pdf_file:
            files = {
                'pdf_file': ('OA-2025-01-01.pdf', pdf_file, 'application/pdf')
            }

            print("Making upload request...")
            upload_response = session.post('http://localhost:5000/upload',
                                         data=upload_data,
                                         files=files,
                                         timeout=60)

        print(f"Upload response status: {upload_response.status_code}")
        print(f"Upload response headers: {dict(upload_response.headers)}")

        if upload_response.status_code == 200:
            # This means validation failed - look for flash messages
            response_text = upload_response.text
            print("=== RESPONSE CONTENT ===")
            print(response_text[:1000])  # First 1000 chars

            # Look for flash messages or error indicators
            if 'alert' in response_text.lower():
                print("\n=== ALERT MESSAGES FOUND ===")
                import re
                alerts = re.findall(r'alert[^>]*>(.*?)</div>', response_text, re.DOTALL | re.IGNORECASE)
                for alert in alerts:
                    print(f"Alert: {alert.strip()}")

            if 'error' in response_text.lower():
                print("\n=== ERROR INDICATORS FOUND ===")
                # Look for error messages
                error_lines = [line for line in response_text.split('\n') if 'error' in line.lower()][:5]
                for line in error_lines:
                    print(f"Error line: {line.strip()}")

    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    debug_upload()