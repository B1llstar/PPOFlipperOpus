import sqlite3
import json
from datetime import datetime

def export_db_to_json(db_path='ge_prices.db', output_file='ge_prices_export.json'):
    """
    Export SQLite database to JSON format for MongoDB import.
    Each table will be exported as a separate collection.
    """
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # This allows us to access columns by name
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    result = {}
    
    for table in tables:
        table_name = table[0]
        print(f"\nExporting table: {table_name}")
        
        try:
            # Get all rows from the table
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            table_data = []
            for row in rows:
                row_dict = {}
                for key in row.keys():
                    value = row[key]
                    # Handle any special conversions if needed
                    if value is not None:
                        row_dict[key] = value
                    else:
                        row_dict[key] = None
                table_data.append(row_dict)
            
            result[table_name] = table_data
            print(f"  ✓ Exported {len(table_data)} rows")
        except Exception as e:
            print(f"  ✗ Error exporting {table_name}: {e}")
            print(f"    Skipping this table...")
            result[table_name] = []
    
    conn.close()
    
    # Write to JSON file
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"✓ Export complete! Data saved to {output_file}")
    print(f"\nSummary:")
    for table_name, data in result.items():
        print(f"  - {table_name}: {len(data)} documents")
    
    return result


def export_db_to_separate_json_files(db_path='ge_prices.db', output_dir='mongo_export'):
    """
    Export SQLite database to separate JSON files for each table.
    This format is better for mongoimport.
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    for table in tables:
        table_name = table[0]
        print(f"\nExporting table: {table_name}")
        
        try:
            # Get all rows from the table
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            table_data = []
            for row in rows:
                row_dict = {}
                for key in row.keys():
                    value = row[key]
                    if value is not None:
                        row_dict[key] = value
                    else:
                        row_dict[key] = None
                table_data.append(row_dict)
            
            # Write to separate JSON file (JSONL format for mongoimport)
            output_file = os.path.join(output_dir, f"{table_name}.json")
            with open(output_file, 'w') as f:
                # Write as JSON array (can also do JSONL by writing one object per line)
                json.dump(table_data, f, indent=2, default=str)
            
            print(f"  ✓ Exported {len(table_data)} rows to {output_file}")
        except Exception as e:
            print(f"  ✗ Error exporting {table_name}: {e}")
            print(f"    Skipping this table...")
    
    conn.close()
    
    print(f"\n✓ All tables exported to {output_dir}/")
    print(f"\nTo import into MongoDB, use:")
    print(f"  mongoimport --db your_database --collection items --file {output_dir}/items.json --jsonArray")
    print(f"  mongoimport --db your_database --collection backfill_5m --file {output_dir}/backfill_5m.json --jsonArray")


if __name__ == "__main__":
    print("=" * 60)
    print("SQLite to MongoDB JSON Exporter")
    print("=" * 60)
    
    # Export as single JSON file
    print("\n[Option 1] Exporting all tables to single JSON file...")
    export_db_to_json()
    
    # Export as separate files
    print("\n" + "=" * 60)
    print("\n[Option 2] Exporting tables to separate JSON files...")
    export_db_to_separate_json_files()
    
    print("\n" + "=" * 60)
    print("✓ Export complete!")
