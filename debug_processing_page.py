#!/usr/bin/env python3
"""
DEBUG: Extract publication ID from processing page
"""
import requests
import os
import re

def debug_processing_page():
    """Debug what the processing page contains"""
    print("=== DEBUG PROCESSING PAGE ===")

    session = requests.Session()

    # Login
    login_data = {'password': 'CCCitizen56101!'}
    login_response = session.post('http://localhost:5000/login', data=login_data)
    print(f"Login status: {login_response.status_code}")

    # Upload file
    pdf_path = "C:\\Users\\trevo\\newspaper-ad-measurement\\OA-2025-01-01.pdf"
    upload_data = {'publication_type': 'broadsheet'}

    try:
        with open(pdf_path, 'rb') as pdf_file:
            files = {'pdf_file': ('OA-2025-01-01.pdf', pdf_file, 'application/pdf')}
            upload_response = session.post('http://localhost:5000/upload',
                                         data=upload_data,
                                         files=files,
                                         timeout=60)

        print(f"Upload response status: {upload_response.status_code}")

        if upload_response.status_code == 200:
            response_text = upload_response.text
            print("=== PROCESSING PAGE CONTENT ===")

            # Save full content to file for analysis
            with open("processing_page_debug.html", "w", encoding="utf-8") as f:
                f.write(response_text)
            print("Full content saved to processing_page_debug.html")

            # Look for any numbers that could be publication ID
            pub_id_patterns = [
                r'"pub_id":\s*(\d+)',
                r'publication[_-]?id["\']:\s*["\']?(\d+)',
                r'/processing/(\d+)',
                r'data-pub-id["\']?=["\']?(\d+)',
                r'publication[_\s]+(\d+)',
                r'id["\']?:["\']?(\d+)',
                r'var\s+pubId\s*=\s*["\']?(\d+)',
                r'pubId["\']?:["\']?(\d+)'
            ]

            print("\n=== SEARCHING FOR PUBLICATION ID PATTERNS ===")
            for pattern in pub_id_patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                if matches:
                    print(f"Pattern '{pattern}' found: {matches}")

            # Look for any JavaScript variables
            print("\n=== JAVASCRIPT VARIABLES ===")
            js_vars = re.findall(r'var\s+(\w+)\s*=\s*([^;]+);', response_text)
            for var_name, var_value in js_vars:
                if any(keyword in var_name.lower() for keyword in ['id', 'pub', 'publication']):
                    print(f"{var_name} = {var_value}")

            # Look for URL patterns
            print("\n=== URL PATTERNS ===")
            urls = re.findall(r'["\']([^"\']*(?:processing|publication)[^"\']*)["\']', response_text, re.IGNORECASE)
            for url in urls[:10]:  # First 10 URLs
                print(f"URL: {url}")

            # Check if we can find the latest publication ID from database
            print("\n=== CHECKING LATEST PUBLICATION ===")
            from app import app, db, Publication
            with app.app_context():
                latest_pub = Publication.query.filter_by(original_filename="OA-2025-01-01.pdf").order_by(Publication.id.desc()).first()
                if latest_pub:
                    print(f"Latest publication ID: {latest_pub.id}")
                    print(f"Filename: {latest_pub.original_filename}")
                    print(f"Processed: {latest_pub.processed}")
                    return latest_pub.id
                else:
                    print("No publication found in database")
                    return None

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    pub_id = debug_processing_page()
    if pub_id:
        print(f"\nFound publication ID: {pub_id}")
    else:
        print("\nCould not find publication ID")