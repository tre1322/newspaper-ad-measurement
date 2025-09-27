#!/usr/bin/env python3
"""
NEWSPAPER DOMAIN DETECTOR: Constraint-based business ad detection
Based on actual analysis of OA-2025-01-01.pdf newspaper structure
"""
import fitz
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class BusinessAd:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    ad_type: str
    text_snippet: str
    business_indicators: List[str]

class NewspaperDomainDetector:
    """Newspaper-specific business ad detector using domain constraints"""

    def __init__(self):
        # Newspaper business patterns (from actual analysis)
        self.phone_pattern = re.compile(r'\b507[-.]?\d{3}[-.]?\d{4}\b')  # Local area code
        self.price_pattern = re.compile(r'\$\d+(?:\.\d{2})?')
        self.business_hours_pattern = re.compile(r'\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun).*(?:am|pm|AM|PM)\b')
        self.address_pattern = re.compile(r'\b\d+.*(?:St|Ave|Rd|Dr|Street|Avenue|Road|Drive)\b', re.IGNORECASE)

        # Business service keywords (from analysis)
        self.business_services = {
            'construction', 'repair', 'roofing', 'plumbing', 'electrical',
            'insurance', 'dental', 'medical', 'hardware', 'lumber', 'auto',
            'cleaning', 'lawn', 'landscaping', 'glass', 'contracting'
        }

        # Editorial exclusion patterns
        self.editorial_patterns = {
            'staff writer', 'reporter', 'editor', 'byline', 'associated press',
            'city council', 'school district', 'county', 'police', 'fire department',
            'meanwhile', 'story', 'article', 'news', 'breaking'
        }

        # Legal/court exclusion patterns (these contain prices but are not business ads)
        self.legal_patterns = {
            'dwi', 'dui', 'speeding', 'violation', 'fine', 'court', 'license', 'citation',
            'revocation', 'suspension', 'no insurance', 'driving', 'vehicle', 'operate',
            'minnesota housing finance', 'agency approved', 'million grant', 'hotel project',
            'estimated cost', 'total cost', 'project will bring'
        }

        # Size constraints for real business ads (from analysis)
        self.min_ad_width = 40.0
        self.min_ad_height = 15.0
        self.max_ad_width = 800.0
        self.max_ad_height = 400.0

    def detect_business_ads(self, pdf_path: str, page_number: int) -> List[BusinessAd]:
        """Detect business ads using newspaper domain constraints"""
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]

        # Get text blocks with coordinates
        text_dict = page.get_text("dict")
        business_ads = []

        for block in text_dict["blocks"]:
            if "lines" not in block:
                continue

            # Extract text and coordinates
            block_text = ""
            for line in block["lines"]:
                for span in line["spans"]:
                    block_text += span["text"] + " "

            # Clean text
            try:
                block_text = block_text.strip().encode('ascii', 'ignore').decode('ascii')
            except:
                continue

            if len(block_text) < 10:
                continue

            # Get bounding box
            bbox = block["bbox"]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # Size constraints
            if (width < self.min_ad_width or height < self.min_ad_height or
                width > self.max_ad_width or height > self.max_ad_height):
                continue

            # Score this text block
            ad_result = self._score_business_content(block_text, bbox)
            if ad_result:
                business_ads.append(ad_result)

        doc.close()
        return business_ads

    def _score_business_content(self, text: str, bbox: Tuple[float, float, float, float]) -> BusinessAd:
        """Score text block for business content using newspaper constraints"""
        text_lower = text.lower()
        confidence = 0.0
        business_indicators = []
        ad_type = "unknown"

        # CONSTRAINT 1: Phone number (highest confidence for business ads)
        phone_matches = self.phone_pattern.findall(text)
        if phone_matches:
            confidence += 40.0  # Strong business indicator
            business_indicators.append(f"phone: {phone_matches[0]}")
            ad_type = "business_directory"

        # CONSTRAINT 2: Pricing information (classified ads)
        price_matches = self.price_pattern.findall(text)
        if price_matches:
            confidence += 25.0
            business_indicators.append(f"pricing: {len(price_matches)} prices")
            if "asking" in text_lower or "call" in text_lower:
                ad_type = "classified_ad"
                confidence += 10.0

        # CONSTRAINT 3: Business services
        service_count = 0
        for service in self.business_services:
            if service in text_lower:
                service_count += 1
                business_indicators.append(f"service: {service}")

        if service_count > 0:
            confidence += service_count * 8.0
            ad_type = "service_ad"

        # CONSTRAINT 4: Business hours
        if self.business_hours_pattern.search(text):
            confidence += 15.0
            business_indicators.append("business_hours")

        # CONSTRAINT 5: Address information
        if self.address_pattern.search(text):
            confidence += 10.0
            business_indicators.append("address")

        # EXCLUSION: Editorial content patterns
        editorial_penalty = 0
        for pattern in self.editorial_patterns:
            if pattern in text_lower:
                editorial_penalty += 15.0

        # EXCLUSION: Legal/court content patterns (these have prices but are not business ads)
        legal_penalty = 0
        for pattern in self.legal_patterns:
            if pattern in text_lower:
                legal_penalty += 25.0  # Heavy penalty for legal content

        total_penalty = editorial_penalty + legal_penalty
        confidence -= total_penalty

        # HARD EXCLUSION: If this looks like legal/court content, reject completely
        if legal_penalty >= 25.0:
            return None

        # CONSTRAINT 6: Business directory structure
        if ("507-" in text and confidence > 20.0):
            # This is likely a business directory entry
            confidence += 15.0
            ad_type = "business_directory"

        # MINIMUM THRESHOLD for business ads (very conservative)
        if confidence >= 40.0:  # Only very high-confidence business content
            return BusinessAd(
                x=bbox[0],
                y=bbox[1],
                width=bbox[2] - bbox[0],
                height=bbox[3] - bbox[1],
                confidence=min(confidence, 100.0),
                ad_type=ad_type,
                text_snippet=text[:100] + "..." if len(text) > 100 else text,
                business_indicators=business_indicators
            )

        return None

def test_newspaper_detector():
    """Test the newspaper domain detector on actual file"""
    print("=== TESTING NEWSPAPER DOMAIN DETECTOR ===")

    detector = NewspaperDomainDetector()
    pdf_path = "C:\\Users\\trevo\\newspaper-ad-measurement\\OA-2025-01-01.pdf"

    total_ads = 0
    for page_num in range(1, 11):
        print(f"\n--- PAGE {page_num} ---")
        ads = detector.detect_business_ads(pdf_path, page_num)
        print(f"Detected {len(ads)} business ads")

        for i, ad in enumerate(ads):
            print(f"  Ad {i+1}: {ad.ad_type} (confidence: {ad.confidence:.1f})")
            print(f"    Indicators: {', '.join(ad.business_indicators)}")
            print(f"    Text: {ad.text_snippet}")
            print(f"    Size: {ad.width:.1f} x {ad.height:.1f}")

        total_ads += len(ads)

    print(f"\n=== TOTAL BUSINESS ADS DETECTED: {total_ads} ===")

    if total_ads >= 15:
        print("SUCCESS: Detected 15+ business ads using newspaper domain constraints")
        return True
    else:
        print(f"NEEDS REFINEMENT: Only {total_ads} ads detected")
        return False

if __name__ == "__main__":
    test_newspaper_detector()