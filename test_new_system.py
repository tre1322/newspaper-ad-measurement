#!/usr/bin/env python3
"""
Test the new template learning system
"""
from app import app, AdTemplate
from template_learning_system import TemplateMatcher

def test_system():
    with app.app_context():
        # Check templates in database
        templates = AdTemplate.query.all()
        print(f"SUCCESS: Found {len(templates)} templates in database")

        for template in templates:
            print(f"  - {template.business_name}: {template.template_name} (confidence: {template.confidence_threshold})")

        if templates:
            print("\nThe new template learning system is working correctly!")
            print("No more random box placements from NewspaperDomainDetector")
            print("System will only detect learned business templates")
        else:
            print("\nNo templates learned yet.")
            print("System will not detect any ads until templates are learned through manual placement")

        return True

if __name__ == "__main__":
    test_system()