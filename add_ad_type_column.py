#!/usr/bin/env python3
"""
Simple database migration script to add ad_type column to AdBox table
This script directly modifies the database without loading the full app
"""

import sqlite3
import os

def add_ad_type_column():
    """Add ad_type column to AdBox table"""
    
    # Find the database file
    db_paths = [
        'newspaper_ads.db',
        'instance/newspaper_ads.db',
        os.path.join('instance', 'newspaper_ads.db')
    ]
    
    db_path = None
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if not db_path:
        print("ERROR: Database file not found. Looking for:")
        for path in db_paths:
            print(f"   - {path}")
        return False
    
    print(f"Found database: {db_path}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if ad_type column already exists
        cursor.execute("PRAGMA table_info(ad_box)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'ad_type' in columns:
            print("SUCCESS: ad_type column already exists!")
            conn.close()
            return True
        
        print("Adding ad_type column...")
        
        # Add the ad_type column
        cursor.execute("ALTER TABLE ad_box ADD COLUMN ad_type VARCHAR(50) DEFAULT 'manual'")
        
        # Update existing records
        cursor.execute("UPDATE ad_box SET ad_type = 'manual' WHERE ad_type IS NULL")
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print("SUCCESS: Successfully added ad_type column!")
        print("SUCCESS: Updated existing records with 'manual' ad_type")
        return True
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("Adding ad_type column to database...")
    
    success = add_ad_type_column()
    
    if success:
        print("\nMigration completed successfully!")
        print("Your database is now ready for intelligent ad detection!")
        print("\nYou can now:")
        print("   - Start your Flask app")
        print("   - Upload broadsheet publications")  
        print("   - Use intelligent click detection")
    else:
        print("\nMigration failed. Please check the error messages above.")