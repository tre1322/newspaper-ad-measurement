#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, '.')

from app import app, db
from sqlalchemy import text
import traceback

def add_production_columns():
    """Add processing_status and processing_error columns to production database"""
    
    with app.app_context():
        try:
            # Check if we're using PostgreSQL or SQLite
            engine = db.engine
            is_postgres = 'postgresql' in str(engine.url)
            is_sqlite = 'sqlite' in str(engine.url)
            
            print(f"Database type detected: {'PostgreSQL' if is_postgres else 'SQLite' if is_sqlite else 'Unknown'}")
            print(f"Database URL: {engine.url}")
            
            # Check current table structure
            if is_postgres:
                result = db.session.execute(text("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'publication'
                """))
            else:  # SQLite
                result = db.session.execute(text("PRAGMA table_info(publication)"))
            
            columns = [row[0] if is_postgres else row[1] for row in result.fetchall()]
            print(f"Existing columns: {columns}")
            
            # Add processing_status column if it doesn't exist
            if 'processing_status' not in columns:
                print("Adding processing_status column...")
                if is_postgres:
                    db.session.execute(text("""
                        ALTER TABLE publication 
                        ADD COLUMN processing_status VARCHAR(50) DEFAULT 'uploaded'
                    """))
                else:  # SQLite
                    db.session.execute(text("""
                        ALTER TABLE publication 
                        ADD COLUMN processing_status VARCHAR(50) DEFAULT 'uploaded'
                    """))
                print("SUCCESS: Added processing_status column")
            else:
                print("processing_status column already exists")
            
            # Add processing_error column if it doesn't exist
            if 'processing_error' not in columns:
                print("Adding processing_error column...")
                if is_postgres:
                    db.session.execute(text("""
                        ALTER TABLE publication 
                        ADD COLUMN processing_error VARCHAR(500)
                    """))
                else:  # SQLite
                    db.session.execute(text("""
                        ALTER TABLE publication 
                        ADD COLUMN processing_error VARCHAR(500)
                    """))
                print("SUCCESS: Added processing_error column")
            else:
                print("processing_error column already exists")
            
            # Update existing records
            if is_postgres:
                result = db.session.execute(text("""
                    UPDATE publication 
                    SET processing_status = CASE 
                        WHEN processed = true THEN 'completed'
                        ELSE 'uploaded'
                    END
                    WHERE processing_status IS NULL
                """))
            else:  # SQLite
                result = db.session.execute(text("""
                    UPDATE publication 
                    SET processing_status = CASE 
                        WHEN processed = 1 THEN 'completed'
                        ELSE 'uploaded'
                    END
                    WHERE processing_status IS NULL
                """))
            
            updated_count = result.rowcount
            print(f"SUCCESS: Updated {updated_count} existing records")
            
            db.session.commit()
            print("SUCCESS: Database schema updated successfully!")
            return True
            
        except Exception as e:
            print(f"ERROR: {str(e)}")
            print("Full traceback:")
            traceback.print_exc()
            db.session.rollback()
            return False

if __name__ == "__main__":
    print("Adding processing columns to production database...")
    success = add_production_columns()
    if success:
        print("Database update completed successfully!")
        exit(0)
    else:
        print("Database update failed!")
        exit(1)