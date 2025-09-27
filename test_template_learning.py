#!/usr/bin/env python3
"""
Test Template Learning System with Cottonwood Vet Clinic
"""
import os
from app import app, db, AdTemplate
from template_learning_system import TemplateExtractor, TemplateMatcher

def test_cottonwood_vet_learning():
    """Test learning template from Cottonwood Vet Clinic ad"""

    # Test file path
    pdf_path = "OA-2025-01-01.pdf"

    if not os.path.exists(pdf_path):
        print(f"ERROR: Test file {pdf_path} not found")
        return False

    print("Testing Template Learning System")
    print("=" * 50)

    with app.app_context():
        try:
            # Example coordinates for Cottonwood Vet ad (these would come from user drawing)
            # Based on the screenshot, the ad appears to be in the lower portion
            page_number = 1
            x = 50    # Left position
            y = 600   # Top position (approximate)
            width = 200   # Width of ad
            height = 150  # Height of ad
            business_name = "Cottonwood Vet Clinic"

            print(f"1. Extracting template from {business_name}")
            print(f"   Location: Page {page_number}, ({x}, {y}) - {width}x{height}")

            # Extract template features
            template_data = TemplateExtractor.extract_template_from_region(
                pdf_path, page_number, x, y, width, height, business_name
            )

            if not template_data:
                print("ERROR: Failed to extract template")
                return False

            print("SUCCESS: Template features extracted successfully")

            # Save template to database
            print("2. Saving template to database...")

            # Check if template already exists
            existing = AdTemplate.query.filter_by(business_name=business_name).first()
            if existing:
                print(f"   Template for {business_name} already exists, updating...")
                for key, value in template_data.items():
                    if key != 'business_name':
                        setattr(existing, key, value)
                template = existing
            else:
                print(f"   Creating new template for {business_name}")
                template = AdTemplate(**template_data)
                db.session.add(template)

            db.session.commit()
            print(f"SUCCESS: Template saved with ID: {template.id}")

            # Test template matching
            print("3. Testing template matching...")

            active_templates = AdTemplate.query.filter_by(is_active=True).all()
            print(f"   Found {len(active_templates)} active templates")

            matches = TemplateMatcher.find_template_matches(
                pdf_path, page_number, active_templates, confidence_threshold=0.5
            )

            print(f"SUCCESS: Template matching complete")
            print(f"   Found {len(matches)} potential matches")

            for i, match in enumerate(matches):
                print(f"   Match {i+1}: {match['business_name']} (confidence: {match['confidence']:.3f})")
                print(f"     Position: ({match['x']:.0f}, {match['y']:.0f}) - {match['width']:.0f}x{match['height']:.0f}")

            print("\n" + "=" * 50)
            print("SUCCESS: Template Learning Test Complete!")
            print(f"Template for '{business_name}' is ready for automatic detection")

            return True

        except Exception as e:
            print(f"ERROR: Error during template learning test: {e}")
            import traceback
            traceback.print_exc()
            return False

def list_existing_templates():
    """List all existing templates in database"""
    print("\nExisting Templates:")
    print("-" * 30)

    with app.app_context():
        templates = AdTemplate.query.all()

        if not templates:
            print("No templates found in database")
            return

        for template in templates:
            status = "Active" if template.is_active else "Inactive"
            print(f"ID {template.id}: {template.business_name}")
            print(f"  Template: {template.template_name}")
            print(f"  Dimensions: {template.typical_width}x{template.typical_height}")
            print(f"  Aspect Ratio: {template.aspect_ratio:.2f}")
            print(f"  Status: {status}")
            print(f"  Created: {template.created_date}")
            print(f"  Detections: {template.detection_count}")
            print()

if __name__ == "__main__":
    print("TEMPLATE LEARNING SYSTEM TEST")
    print("Testing with Cottonwood Vet Clinic ad")
    print()

    # List existing templates first
    list_existing_templates()

    # Run the test
    success = test_cottonwood_vet_learning()

    if success:
        print("\nSUCCESS: Template learning system is working!")
        print("Ready for integration with manual ad placement interface")
    else:
        print("\nERROR: Template learning test failed")
        print("Check the error messages above for debugging")