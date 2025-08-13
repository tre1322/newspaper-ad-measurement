#!/usr/bin/env python3
"""
Database migration script to add ML tables for the AI learning system
This script adds MLModel and TrainingData tables to the existing database
"""

import os
import sqlite3
from datetime import datetime

def add_ml_tables():
    """Add ML tables to existing database"""
    
    # Find the database file
    db_paths = [
        'newspaper_ads.db',
        'instance/newspaper_ads.db',
        os.path.join(os.path.dirname(__file__), 'newspaper_ads.db'),
        os.path.join(os.path.dirname(__file__), 'instance', 'newspaper_ads.db')
    ]
    
    db_path = None
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if not db_path:
        print("Database file not found! Please make sure the app has been run at least once.")
        print("Looking for database in:")
        for path in db_paths:
            print(f"  - {path}")
        return
    
    print(f"Found database: {db_path}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("Adding ML tables to database...")
        
        # Create MLModel table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ml_model (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name VARCHAR(100) NOT NULL,
                model_type VARCHAR(50) NOT NULL,
                publication_type VARCHAR(50) NOT NULL,
                version VARCHAR(20) NOT NULL,
                model_data BLOB,
                training_accuracy FLOAT,
                validation_accuracy FLOAT,
                training_samples INTEGER,
                feature_names TEXT,
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Create TrainingData table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ad_box_id INTEGER NOT NULL,
                publication_type VARCHAR(50) NOT NULL,
                features TEXT,
                label VARCHAR(50) NOT NULL,
                confidence_score FLOAT,
                extracted_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                used_in_training BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (ad_box_id) REFERENCES ad_box (id)
            )
        ''')
        
        # Commit changes
        conn.commit()
        print("Successfully added ML tables!")
        
        # Check existing data
        cursor.execute("SELECT COUNT(*) FROM ad_box WHERE user_verified = 1")
        verified_ads = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT publication_type) FROM publication")
        pub_types = cursor.fetchone()[0]
        
        print(f"Database ready for ML training:")
        print(f"   - {verified_ads} verified ads available for training")
        print(f"   - {pub_types} publication types in database")
        
        if verified_ads >= 20:
            print("You have enough data to start training ML models!")
        else:
            print(f"You need {20 - verified_ads} more verified ads to train your first model")
        
        conn.close()
        
    except Exception as e:
        print(f"Error updating database: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Adding AI Learning System tables to database...")
    print("=" * 50)
    
    if add_ml_tables():
        print("=" * 50)
        print("Database migration completed successfully!")
        print("Your AI Learning System is now ready to use!")
        print("\nNext steps:")
        print("1. Run the application: python app.py")
        print("2. Go to AI Learning dashboard: http://localhost:5000/ml")
        print("3. Start adding verified ads to build training data")
        print("4. Train your first model when you have 20+ verified ads")
    else:
        print("Migration failed. Please check the error messages above.")