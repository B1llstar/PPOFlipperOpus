import sqlite3
import shutil
import os

db_path = 'ge_prices.db'
backup_path = 'ge_prices_backup.db'
repaired_path = 'ge_prices_repaired.db'

print("Attempting to repair database...")

# Backup original
if os.path.exists(db_path):
    print(f"Creating backup: {backup_path}")
    shutil.copy2(db_path, backup_path)

try:
    # Try to dump and restore
    print("Dumping database...")
    conn = sqlite3.connect(db_path)
    
    # Create repaired database
    new_conn = sqlite3.connect(repaired_path)
    
    # Dump and restore
    for line in conn.iterdump():
        if line not in ('BEGIN;', 'COMMIT;'):
            new_conn.execute(line)
    
    new_conn.commit()
    new_conn.close()
    conn.close()
    
    print(f"Repaired database created: {repaired_path}")
    print(f"To use it, run: move {repaired_path} {db_path}")
    print("Or manually rename the file.")
    
except Exception as e:
    print(f"Repair failed: {e}")
    print("\nThe database is too corrupted to repair.")
    print("You need to recreate it by running:")
    print("  cd data")
    print("  python collect_all.py")
