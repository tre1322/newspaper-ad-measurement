#!/usr/bin/env python3
"""
NEWSPAPER STRUCTURE ANALYSIS: Understand actual business ads vs editorial content
"""
import fitz  # PyMuPDF
import re

def analyze_newspaper_structure():
    """Manually analyze the OA newspaper to understand ad vs editorial patterns"""
    print("=== NEWSPAPER STRUCTURE ANALYSIS ===")

    pdf_path = "C:\\Users\\trevo\\newspaper-ad-measurement\\OA-2025-01-01.pdf"
    doc = fitz.open(pdf_path)

    print(f"Analyzing {doc.page_count} pages of {pdf_path}")
    print()

    # Business indicators to look for
    phone_pattern = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    price_pattern = re.compile(r'\$\d+(?:\.\d{2})?')
    business_hours_pattern = re.compile(r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun).*(?:am|pm|AM|PM)\b')
    address_pattern = re.compile(r'\b\d+\s+[A-Z][a-z]+\s+(?:St|Ave|Rd|Dr|Blvd|Street|Avenue|Road|Drive|Boulevard)\b')

    business_keywords = [
        'service', 'repair', 'insurance', 'real estate', 'attorney', 'law', 'medical',
        'dental', 'restaurant', 'auto', 'plumbing', 'electrical', 'construction',
        'cleaning', 'salon', 'spa', 'fitness', 'gym', 'store', 'shop', 'business',
        'company', 'inc', 'llc', 'corp', 'professional', 'licensed', 'certified'
    ]

    editorial_keywords = [
        'reporter', 'editor', 'staff writer', 'correspondent', 'byline', 'dateline',
        'associated press', 'ap', 'news', 'breaking', 'update', 'story', 'article',
        'county', 'city council', 'mayor', 'sheriff', 'police', 'fire department',
        'school district', 'board meeting', 'public hearing', 'court', 'trial'
    ]

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()

        print(f"=== PAGE {page_num + 1} ANALYSIS ===")

        # Extract text blocks with positions
        text_dict = page.get_text("dict")

        potential_ads = []
        editorial_content = []

        for block in text_dict["blocks"]:
            if "lines" in block:
                block_text = ""
                for line in block["lines"]:
                    for span in line["spans"]:
                        block_text += span["text"] + " "

                # Clean text to avoid Unicode issues
                try:
                    block_text = block_text.strip().lower().encode('ascii', 'ignore').decode('ascii')
                except:
                    continue
                if len(block_text) < 10:  # Skip very short text
                    continue

                # Score this text block
                business_score = 0
                editorial_score = 0

                # Check for business indicators
                if phone_pattern.search(block_text):
                    business_score += 3
                    print(f"  PHONE found: {phone_pattern.search(block_text).group()}")

                if price_pattern.search(block_text):
                    business_score += 2
                    print(f"  PRICE found: {price_pattern.search(block_text).group()}")

                if business_hours_pattern.search(block_text):
                    business_score += 2
                    print(f"  HOURS found: {business_hours_pattern.search(block_text).group()}")

                if address_pattern.search(block_text):
                    business_score += 2
                    print(f"  ADDRESS found: {address_pattern.search(block_text).group()}")

                # Check for business keywords
                for keyword in business_keywords:
                    if keyword in block_text:
                        business_score += 1

                # Check for editorial keywords
                for keyword in editorial_keywords:
                    if keyword in block_text:
                        editorial_score += 2

                # Determine type
                if business_score >= 3:  # Strong business indicators
                    bbox = block["bbox"]
                    potential_ads.append({
                        'text': block_text[:100] + "..." if len(block_text) > 100 else block_text,
                        'score': business_score,
                        'bbox': bbox,
                        'width': bbox[2] - bbox[0],
                        'height': bbox[3] - bbox[1]
                    })
                elif editorial_score >= 2:  # Strong editorial indicators
                    editorial_content.append({
                        'text': block_text[:100] + "..." if len(block_text) > 100 else block_text,
                        'score': editorial_score
                    })

        print(f"  POTENTIAL BUSINESS ADS: {len(potential_ads)}")
        for ad in potential_ads:
            print(f"    Score {ad['score']}: {ad['text']}")
            print(f"    Size: {ad['width']:.1f} x {ad['height']:.1f}")

        print(f"  EDITORIAL CONTENT: {len(editorial_content)}")
        for content in editorial_content[:3]:  # Show first 3
            print(f"    Score {content['score']}: {content['text']}")

        print()

    doc.close()
    print("Analysis complete. Use this to build constraint-based detection.")

if __name__ == "__main__":
    analyze_newspaper_structure()