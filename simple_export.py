#!/usr/bin/env python3
"""
Simple SQLite to JSON Exporter

Exports all tables from ge_prices.db to separate JSON files.
Handles corrupted tables gracefully.
"""

import sqlite3
import json
import os
from pathlib import Path

def export_sqlite_to_json(db_path='ge_prices.db', output_dir='mongo_export'):
    """Export SQLite database to JSON files."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"Connecting to {db_path}...")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    print(f"\nFound {len(tables)} tables: {', '.join(tables)}")
    print("\n" + "="*60)
    
    exported_tables = []
    failed_tables = []
    
    for table_name in tables:
        print(f"\nExporting: {table_name}")
        output_file = os.path.join(output_dir, f"{table_name}.json")
        
        try:
            # Get all rows
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            # Convert to list of dicts
            data = [dict(row) for row in rows]
            
            # Write to JSON file
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"  ✓ Exported {len(data)} records to {output_file}")
            exported_tables.append((table_name, len(data)))
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed_tables.append(table_name)
            
            # Try to export partial data if possible
            try:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 1000")
                rows = cursor.fetchall()
                data = [dict(row) for row in rows]
                
                partial_file = os.path.join(output_dir, f"{table_name}_partial.json")
                with open(partial_file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"  ~ Partial export: {len(data)} records to {partial_file}")
            except:
                print(f"  ~ Could not export any data from {table_name}")
    
    conn.close()
    
    # Summary
    print("\n" + "="*60)
    print("\nEXPORT SUMMARY")
    print("="*60)
    
    if exported_tables:
        print("\n✓ Successfully Exported:")
        for table, count in exported_tables:
            print(f"  - {table}: {count:,} records")
    
    if failed_tables:
        print("\n✗ Failed (corrupted):")
        for table in failed_tables:
            print(f"  - {table}")
    
    print(f"\n✓ Files saved to: {output_dir}/")
    print("\nTo import to MongoDB:")
    for table, _ in exported_tables:
        print(f"  mongoimport --db osrs_ge --collection {table} --file {output_dir}/{table}.json --jsonArray")


if __name__ == "__main__":
    export_sqlite_to_json()
