#!/usr/bin/env python3
"""
PDF Structure Analysis for Newspaper Ad Detection
Test script to see what metadata your InDesign PDFs contain
"""

import fitz  # PyMuPDF (you already have this)
import os
import json
from collections import defaultdict

def analyze_pdf_structure(pdf_path):
    """Complete analysis of PDF structure for ad detection"""
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return None
    
    try:
        doc = fitz.open(pdf_path)
        print(f"📄 Analyzing PDF: {os.path.basename(pdf_path)}")
        print(f"📊 Pages: {doc.page_count}")
        print(f"🏗️  Creator: {doc.metadata.get('creator', 'Unknown')}")
        print(f"📝 Producer: {doc.metadata.get('producer', 'Unknown')}")
        print("-" * 60)
        
        all_results = []
        
        # Analyze each page (limit to first 3 for testing)
        for page_num in range(min(3, doc.page_count)):
            print(f"\n🔍 PAGE {page_num + 1} ANALYSIS:")
            page_results = analyze_page_structure(doc[page_num], page_num + 1)
            all_results.append(page_results)
        
        doc.close()
        return all_results
        
    except Exception as e:
        print(f"❌ Error analyzing PDF: {e}")
        return None

def analyze_page_structure(page, page_num):
    """Detailed analysis of a single page"""
    
    results = {
        'page_number': page_num,
        'images': [],
        'text_blocks': [],
        'drawings': [],
        'potential_ads': []
    }
    
    # 1. IMAGE ANALYSIS
    print("🖼️  IMAGES:")
    images = page.get_images()
    print(f"   Found {len(images)} images")
    
    for i, img in enumerate(images):
        try:
            # Get image metadata
            xref = img[0]
            img_dict = page.parent.extract_image(xref)
            
            # Get image placement coordinates
            img_rects = page.get_image_rects(img)
            
            for j, rect in enumerate(img_rects):
                image_info = {
                    'id': f"img_{i}_{j}",
                    'bounds': [float(rect.x0), float(rect.y0), 
                              float(rect.x1), float(rect.y1)],
                    'width': float(rect.width),
                    'height': float(rect.height),
                    'area': float(rect.width * rect.height),
                    'aspect_ratio': float(rect.width / rect.height) if rect.height > 0 else 0,
                    'image_width_px': img_dict.get('width', 0),
                    'image_height_px': img_dict.get('height', 0),
                    'colorspace': img_dict.get('colorspace', 'unknown'),
                    'bpc': img_dict.get('bpc', 0),  # Bits per component
                    'size_bytes': len(img_dict.get('image', b'')),
                    'likely_ad_score': 0
                }
                
                # Score as potential ad
                score = score_image_as_ad(image_info)
                image_info['likely_ad_score'] = score
                
                print(f"   Image {i}.{j}: {rect.width:.0f}x{rect.height:.0f} at ({rect.x0:.0f},{rect.y0:.0f}) - Score: {score}")
                results['images'].append(image_info)
                
        except Exception as e:
            print(f"   ⚠️ Error processing image {i}: {e}")
    
    # 2. TEXT BLOCK ANALYSIS
    print("\n📝 TEXT BLOCKS:")
    text_dict = page.get_text("dict")
    text_blocks = text_dict.get("blocks", [])
    print(f"   Found {len(text_blocks)} text blocks")
    
    for i, block in enumerate(text_blocks):
        if "lines" in block:  # Text block (not image block)
            try:
                block_info = analyze_text_block(block, i)
                score = score_text_block_as_ad(block_info)
                block_info['likely_ad_score'] = score
                
                print(f"   Block {i}: {block_info['width']:.0f}x{block_info['height']:.0f} "
                      f"({block_info['font_count']} fonts) - Score: {score}")
                print(f"      Text preview: \"{block_info['text_preview']}\"")
                
                results['text_blocks'].append(block_info)
                
            except Exception as e:
                print(f"   ⚠️ Error processing text block {i}: {e}")
    
    # 3. VECTOR GRAPHICS ANALYSIS
    print("\n🎨 VECTOR GRAPHICS:")
    drawings = page.get_drawings()
    print(f"   Found {len(drawings)} vector elements")
    
    for i, drawing in enumerate(drawings):
        try:
            drawing_info = {
                'id': f"drawing_{i}",
                'bounds': [float(drawing['rect'].x0), float(drawing['rect'].y0),
                          float(drawing['rect'].x1), float(drawing['rect'].y1)],
                'width': float(drawing['rect'].width),
                'height': float(drawing['rect'].height),
                'area': float(drawing['rect'].width * drawing['rect'].height),
                'items': len(drawing.get('items', [])),
                'likely_border': is_likely_border(drawing)
            }
            
            print(f"   Drawing {i}: {drawing_info['width']:.0f}x{drawing_info['height']:.0f} "
                  f"({drawing_info['items']} items) - Border: {drawing_info['likely_border']}")
            
            results['drawings'].append(drawing_info)
            
        except Exception as e:
            print(f"   ⚠️ Error processing drawing {i}: {e}")
    
    # 4. COMBINE INTO POTENTIAL ADS
    print("\n🎯 POTENTIAL ADS:")
    potential_ads = find_potential_ads(results)
    results['potential_ads'] = potential_ads
    
    for i, ad in enumerate(potential_ads):
        print(f"   Ad {i+1}: {ad['width']:.0f}x{ad['height']:.0f} at ({ad['x']:.0f},{ad['y']:.0f}) "
              f"- Confidence: {ad['confidence']:.1f}% - Type: {ad['type']}")
    
    print(f"\n📊 SUMMARY: {len(potential_ads)} potential ads detected")
    
    return results

def analyze_text_block(block, block_id):
    """Analyze typography and content of a text block"""
    
    bbox = block["bbox"]
    fonts = set()
    font_sizes = set()
    total_chars = 0
    text_content = ""
    
    # Analyze all text in the block
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            fonts.add(span.get("font", "unknown"))
            font_sizes.add(span.get("size", 0))
            span_text = span.get("text", "")
            text_content += span_text
            total_chars += len(span_text)
    
    return {
        'id': f"text_{block_id}",
        'bounds': [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
        'x': float(bbox[0]),
        'y': float(bbox[1]),
        'width': float(bbox[2] - bbox[0]),
        'height': float(bbox[3] - bbox[1]),
        'area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
        'font_count': len(fonts),
        'fonts': list(fonts),
        'font_sizes': list(font_sizes),
        'char_count': total_chars,
        'text_content': text_content,
        'text_preview': text_content[:50].replace('\n', ' ').strip()
    }

def score_image_as_ad(image_info):
    """Score an image as likely advertisement"""
    score = 0
    
    # Size scoring
    area = image_info['area']
    if 20000 <= area <= 200000:  # Reasonable ad size
        score += 30
    elif area > 200000:
        score += 15  # Might be too big
    
    # Aspect ratio scoring
    aspect = image_info['aspect_ratio']
    if 0.3 <= aspect <= 3.0:  # Reasonable ad proportions
        score += 25
    
    # Quality scoring
    if image_info['bpc'] >= 8:  # High quality
        score += 15
    
    # Size scoring
    if image_info['size_bytes'] > 50000:  # Substantial image
        score += 10
    
    return score

def score_text_block_as_ad(block_info):
    """Score a text block as likely advertisement"""
    score = 0
    
    # Multiple fonts = designed layout (ad characteristic)
    if block_info['font_count'] >= 2:
        score += 30
    elif block_info['font_count'] >= 3:
        score += 45
    
    # Size scoring
    area = block_info['area']
    if 10000 <= area <= 100000:  # Reasonable ad text size
        score += 20
    
    # Look for commercial text patterns
    text = block_info['text_content'].lower()
    commercial_keywords = [
        'call', 'phone', 'contact', '$', 'price', 'sale', 'free', 
        'visit', 'www.', '.com', 'email', 'hours', 'located'
    ]
    
    for keyword in commercial_keywords:
        if keyword in text:
            score += 5
    
    return score

def is_likely_border(drawing):
    """Check if vector drawing is likely an ad border"""
    rect = drawing['rect']
    items = drawing.get('items', [])
    
    # Simple rectangle with few items = likely border
    if len(items) <= 4 and rect.width > 100 and rect.height > 50:
        return True
    
    return False

def find_potential_ads(page_results):
    """Combine images, text, and vectors into potential ad regions"""
    potential_ads = []
    
    # High-scoring images are likely ads
    for img in page_results['images']:
        if img['likely_ad_score'] >= 40:
            potential_ads.append({
                'x': img['bounds'][0],
                'y': img['bounds'][1], 
                'width': img['width'],
                'height': img['height'],
                'confidence': img['likely_ad_score'],
                'type': 'image_ad',
                'components': ['image']
            })
    
    # High-scoring text blocks are likely ads
    for text in page_results['text_blocks']:
        if text['likely_ad_score'] >= 40:
            potential_ads.append({
                'x': text['x'],
                'y': text['y'],
                'width': text['width'], 
                'height': text['height'],
                'confidence': text['likely_ad_score'],
                'type': 'text_ad',
                'components': ['text']
            })
    
    # Look for image+text combinations (common ad pattern)
    for img in page_results['images']:
        for text in page_results['text_blocks']:
            if are_objects_grouped(img, text):
                combined_bounds = combine_bounds(img['bounds'], text['bounds'])
                combined_confidence = (img['likely_ad_score'] + text['likely_ad_score']) / 2
                
                if combined_confidence >= 30:
                    potential_ads.append({
                        'x': combined_bounds[0],
                        'y': combined_bounds[1],
                        'width': combined_bounds[2] - combined_bounds[0],
                        'height': combined_bounds[3] - combined_bounds[1],
                        'confidence': combined_confidence + 20,  # Bonus for combination
                        'type': 'mixed_ad',
                        'components': ['image', 'text']
                    })
    
    # Remove overlapping detections
    potential_ads = remove_overlapping_ads(potential_ads)
    
    return potential_ads

def are_objects_grouped(obj1, obj2, proximity_threshold=50):
    """Check if two objects are close enough to be part of same ad"""
    bounds1 = obj1['bounds']
    bounds2 = obj2['bounds']
    
    # Calculate distance between objects
    center1_x = (bounds1[0] + bounds1[2]) / 2
    center1_y = (bounds1[1] + bounds1[3]) / 2
    center2_x = (bounds2[0] + bounds2[2]) / 2  
    center2_y = (bounds2[1] + bounds2[3]) / 2
    
    distance = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
    
    return distance <= proximity_threshold

def combine_bounds(bounds1, bounds2):
    """Combine two bounding boxes into one"""
    return [
        min(bounds1[0], bounds2[0]),  # min x
        min(bounds1[1], bounds2[1]),  # min y  
        max(bounds1[2], bounds2[2]),  # max x
        max(bounds1[3], bounds2[3])   # max y
    ]

def remove_overlapping_ads(ads, overlap_threshold=0.3):
    """Remove overlapping ad detections, keep highest confidence"""
    if not ads:
        return ads
    
    # Sort by confidence
    ads = sorted(ads, key=lambda x: x['confidence'], reverse=True)
    
    filtered = []
    for ad in ads:
        overlaps = False
        for existing in filtered:
            if calculate_overlap_ratio(ad, existing) > overlap_threshold:
                overlaps = True
                break
        
        if not overlaps:
            filtered.append(ad)
    
    return filtered

def calculate_overlap_ratio(ad1, ad2):
    """Calculate overlap ratio between two ads"""
    x1 = max(ad1['x'], ad2['x'])
    y1 = max(ad1['y'], ad2['y'])
    x2 = min(ad1['x'] + ad1['width'], ad2['x'] + ad2['width'])
    y2 = min(ad1['y'] + ad1['height'], ad2['y'] + ad2['height'])
    
    if x2 <= x1 or y2 <= y1:
        return 0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = ad1['width'] * ad1['height']
    area2 = ad2['width'] * ad2['height']
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def save_results_to_json(results, output_file):
    """Save analysis results to JSON file"""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📁 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving results: {e}")

def main():
    """Main function - modify the PDF path below"""
    
    # CHANGE THIS PATH to your newspaper PDF
    pdf_path = r"c:\Users\trevo\Downloads\ccc-2025-08-20-11-12.pdf"
    
    # Example paths (uncomment the one that matches your setup):
    # pdf_path = "static/uploads/pdfs/your_newspaper.pdf"
    # pdf_path = "/path/to/cottonwood_county_citizen_aug_20_2025.pdf"
    # pdf_path = "C:/Users/YourName/Documents/newspaper.pdf"
    
    print("🔬 PDF STRUCTURE ANALYSIS FOR AD DETECTION")
    print("=" * 60)
    
    if not os.path.exists(pdf_path):
        print(f"❌ Please update the pdf_path in this script to point to your newspaper PDF")
        print(f"   Current path: {pdf_path}")
        return
    
    # Run the analysis
    results = analyze_pdf_structure(pdf_path)
    
    if results:
        # Save detailed results
        output_file = f"pdf_analysis_{os.path.basename(pdf_path)}.json"
        save_results_to_json(results, output_file)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 FINAL SUMMARY:")
        
        total_ads = sum(len(page['potential_ads']) for page in results)
        total_images = sum(len(page['images']) for page in results)
        total_text_blocks = sum(len(page['text_blocks']) for page in results)
        total_drawings = sum(len(page['drawings']) for page in results)
        
        print(f"🎯 Total potential ads detected: {total_ads}")
        print(f"🖼️  Total images found: {total_images}")
        print(f"📝 Total text blocks found: {total_text_blocks}")
        print(f"🎨 Total vector graphics found: {total_drawings}")
        
        if total_ads > 0:
            print("\n✅ SUCCESS: PDF contains structured ad data!")
            print("   The metadata approach should work well.")
        else:
            print("\n⚠️  No clear ads detected in PDF structure.")
            print("   PDF may not contain rich metadata.")

if __name__ == "__main__":
    main()