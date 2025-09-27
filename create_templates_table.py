#!/usr/bin/env python3
"""
Create ad_templates table in database
"""
from app import app, db, AdTemplate

def create_templates_table():
    """Create the ad_templates table"""
    print("Creating ad_templates table...")

    with app.app_context():
        try:
            # Create the table
            db.create_all()
            print("SUCCESS: ad_templates table created successfully")

            # Verify table exists
            tables = db.engine.table_names()
            if 'ad_templates' in tables:
                print("SUCCESS: Table confirmed in database")
            else:
                print("ERROR: Table not found in database")

        except Exception as e:
            print(f"ERROR: Error creating table: {e}")

if __name__ == "__main__":
    create_templates_table()