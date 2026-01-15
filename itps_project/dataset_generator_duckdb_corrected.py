"""
DuckDB-based Dataset Generator - CORRECTED VERSION
Uses ACTUAL column structure from CERT r4.2 dataset
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import json
from collections import defaultdict
import hashlib

# ============================================================================
# DUCKDB DATASET LOADER - CORRECTED
# ============================================================================

@dataclass
class CERTMetadata:
    """Metadata about a CERT dataset"""
    version: str
    path: str
    total_users: int
    date_range: Tuple[str, str]
    malicious_users: Set[str]
    benign_users: Set[str]

class DuckDBCERTLoader:
    """High-performance CERT dataset loader using DuckDB with CORRECT columns"""
    
    def __init__(self, cert_root_path: str):
        """
        Initialize DuckDB loader with path to CERT dataset
        
        Args:
            cert_root_path: Path to CERT r4.2 root directory
        """
        self.root = Path(cert_root_path)
        self.conn = duckdb.connect()
        self.metadata = None
        self._validate_structure()
        self._setup_tables()
    
    def _validate_structure(self) -> None:
        """Verify CERT dataset structure"""
        required_files = ['insiders.csv', 'logon.csv', 'file.csv', 
                         'device.csv', 'email.csv', 'http.csv']
        
        for fname in required_files:
            fpath = self.root / fname
            if not fpath.exists():
                raise FileNotFoundError(
                    f"Missing {fname} in {self.root}. "
                    f"Ensure path points to CERT r4.2 root directory."
                )
        
        print(f"✓ CERT r4.2 dataset structure validated at {self.root}")
    
    def _setup_tables(self) -> None:
        """Create DuckDB tables from CSV files with CORRECT column structure"""
        # Create tables with actual CERT r4.2 columns
        self.conn.execute("""
            CREATE TABLE insiders AS 
            SELECT * FROM read_csv_auto('{}')
        """.format(self.root / 'insiders.csv'))
        
        # logon.csv: id,date,user,pc,activity
        self.conn.execute("""
            CREATE TABLE logon AS 
            SELECT 
                id,
                CASE 
                    WHEN CAST("date" AS VARCHAR) LIKE '____-__-__ __:__:__' THEN
                        DATE(STRPTIME(CAST("date" AS VARCHAR), '%Y-%m-%d %H:%M:%S'))
                    ELSE
                        DATE(STRPTIME(CAST("date" AS VARCHAR), '%m/%d/%Y %H:%M:%S'))
                END as "date",
                "user",
                pc,
                activity
            FROM read_csv_auto('{}', types={{'date': 'VARCHAR'}})
        """.format(self.root / 'logon.csv'))
        
        # file.csv: id,date,user,pc,filename,content
        self.conn.execute("""
            CREATE TABLE file_events AS 
            SELECT 
                id,
                CASE 
                    WHEN CAST("date" AS VARCHAR) LIKE '____-__-__ __:__:__' THEN
                        DATE(STRPTIME(CAST("date" AS VARCHAR), '%Y-%m-%d %H:%M:%S'))
                    ELSE
                        DATE(STRPTIME(CAST("date" AS VARCHAR), '%m/%d/%Y %H:%M:%S'))
                END as "date",
                "user",
                pc,
                filename,
                content
            FROM read_csv_auto('{}', types={{'date': 'VARCHAR'}})
        """.format(self.root / 'file.csv'))
        
        # device.csv: id,date,user,pc,activity
        self.conn.execute("""
            CREATE TABLE device_events AS 
            SELECT 
                id,
                CASE 
                    WHEN CAST("date" AS VARCHAR) LIKE '____-__-__ __:__:__' THEN
                        DATE(STRPTIME(CAST("date" AS VARCHAR), '%Y-%m-%d %H:%M:%S'))
                    ELSE
                        DATE(STRPTIME(CAST("date" AS VARCHAR), '%m/%d/%Y %H:%M:%S'))
                END as "date",
                "user",
                pc,
                activity
            FROM read_csv_auto('{}', types={{'date': 'VARCHAR'}})
        """.format(self.root / 'device.csv'))
        
        # email.csv: id,date,user,pc,to,cc,bcc,from,size,attachments,content
        self.conn.execute("""
            CREATE TABLE email_events AS 
            SELECT 
                id,
                CASE 
                    WHEN CAST("date" AS VARCHAR) LIKE '____-__-__ __:__:__' THEN
                        DATE(STRPTIME(CAST("date" AS VARCHAR), '%Y-%m-%d %H:%M:%S'))
                    ELSE
                        DATE(STRPTIME(CAST("date" AS VARCHAR), '%m/%d/%Y %H:%M:%S'))
                END as "date",
                "user",
                pc,
                "to",
                cc,
                bcc,
                "from",
                size,
                attachments,
                content
            FROM read_csv_auto('{}', types={{'date': 'VARCHAR'}})
        """.format(self.root / 'email.csv'))
        
        # http.csv: id,date,user,pc,url,content
        self.conn.execute("""
            CREATE TABLE http_events AS 
            SELECT 
                id,
                CASE 
                    WHEN CAST("date" AS VARCHAR) LIKE '____-__-__ __:__:__' THEN
                        DATE(STRPTIME(CAST("date" AS VARCHAR), '%Y-%m-%d %H:%M:%S'))
                    ELSE
                        DATE(STRPTIME(CAST("date" AS VARCHAR), '%m/%d/%Y %H:%M:%S'))
                END as "date",
                "user",
                pc,
                url,
                content
            FROM read_csv_auto('{}', types={{'date': 'VARCHAR'}})
        """.format(self.root / 'http.csv'))
        
        print("✓ DuckDB tables created with CORRECT CERT r4.2 column structure")
    
    def get_dataset_metadata(self) -> CERTMetadata:
        """Get comprehensive dataset metadata using SQL queries"""
        if self.metadata:
            return self.metadata
            
        # Get malicious users from insiders table
        malicious_users = set(self.conn.execute(
            'SELECT DISTINCT "user" FROM insiders'
        ).fetchall())
        malicious_users = {user[0] for user in malicious_users}
        
        # Get all users from logon events
        all_users = set(self.conn.execute(
            'SELECT DISTINCT "user" FROM logon'
        ).fetchall())
        all_users = {user[0] for user in all_users}
        
        benign_users = all_users - malicious_users
        
        # Get date range
        date_range = self.conn.execute("""
            SELECT MIN("date") as min_date, MAX("date") as max_date 
            FROM logon
        """).fetchone()
        
        self.metadata = CERTMetadata(
            version="r4.2",
            path=str(self.root),
            total_users=len(all_users),
            date_range=(str(date_range[0]), str(date_range[1])),
            malicious_users=malicious_users,
            benign_users=benign_users
        )
        
        return self.metadata
    
    def load_insiders(self) -> pd.DataFrame:
        """Load insider metadata using DuckDB"""
        return self.conn.execute("SELECT * FROM insiders").df()
    
    def get_user_events(self, user_id: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get all events for a specific user using optimized SQL with CORRECT columns"""
        where_clause = f'WHERE "user" = \'{user_id}\''
        
        if start_date:
            where_clause += f' AND "date" >= \'{start_date}\''
        if end_date:
            where_clause += f' AND "date" <= \'{end_date}\''
        
        # Union all event types for the user with ACTUAL columns
        query = f"""
        SELECT 'logon' as event_type, id, "user", "date", pc, activity, NULL as filename, NULL as content, NULL as url
        FROM logon {where_clause}
        UNION ALL
        SELECT 'file' as event_type, id, "user", "date", pc, NULL as activity, filename, content, NULL as url
        FROM file_events {where_clause}
        UNION ALL
        SELECT 'device' as event_type, id, "user", "date", pc, activity, NULL as filename, NULL as content, NULL as url
        FROM device_events {where_clause}
        UNION ALL
        SELECT 'email' as event_type, id, "user", "date", pc, NULL as activity, NULL as filename, content, NULL as url
        FROM email_events {where_clause}
        UNION ALL
        SELECT 'http' as event_type, id, "user", "date", pc, NULL as activity, NULL as filename, content, url
        FROM http_events {where_clause}
        ORDER BY "date"
        """
        
        return self.conn.execute(query).df()
    
    def get_events_by_type(self, event_type: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get events by type with optional date filtering"""
        table_map = {
            'logon': 'logon',
            'file': 'file_events', 
            'device': 'device_events',
            'email': 'email_events',
            'http': 'http_events'
        }
        
        if event_type not in table_map:
            raise ValueError(f"Unknown event type: {event_type}")
        
        where_clause = ""
        if start_date or end_date:
            conditions = []
            if start_date:
                conditions.append(f'"date" >= \'{start_date}\'')
            if end_date:
                conditions.append(f'"date" <= \'{end_date}\'')
            where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"SELECT * FROM {table_map[event_type]} {where_clause}"
        return self.conn.execute(query).df()
    
    def get_user_statistics(self, user_id: str) -> Dict:
        """Get comprehensive statistics for a user using SQL aggregations"""
        stats = {}
        
        # Basic event counts using ACTUAL columns
        event_counts = self.conn.execute(f"""
            SELECT 
                COUNT(*) as total_events,
                COUNT(DISTINCT DATE("date")) as active_days,
                MIN("date") as first_activity,
                MAX("date") as last_activity
            FROM (
                SELECT "user", "date" FROM logon WHERE "user" = '{user_id}'
                UNION ALL
                SELECT "user", "date" FROM file_events WHERE "user" = '{user_id}'
                UNION ALL
                SELECT "user", "date" FROM device_events WHERE "user" = '{user_id}'
                UNION ALL
                SELECT "user", "date" FROM email_events WHERE "user" = '{user_id}'
                UNION ALL
                SELECT "user", "date" FROM http_events WHERE "user" = '{user_id}'
            )
        """).fetchone()
        
        stats.update({
            'total_events': event_counts[0],
            'active_days': event_counts[1],
            'first_activity': str(event_counts[2]),
            'last_activity': str(event_counts[3])
        })
        
        # Event type breakdown using ACTUAL tables
        event_breakdown = self.conn.execute(f"""
            SELECT 
                'logon' as event_type, COUNT(*) as count
                FROM logon WHERE "user" = '{user_id}'
            UNION ALL
            SELECT 'file' as event_type, COUNT(*) as count
                FROM file_events WHERE "user" = '{user_id}'
            UNION ALL
            SELECT 'device' as event_type, COUNT(*) as count
                FROM device_events WHERE "user" = '{user_id}'
            UNION ALL
            SELECT 'email' as event_type, COUNT(*) as count
                FROM email_events WHERE "user" = '{user_id}'
            UNION ALL
            SELECT 'http' as event_type, COUNT(*) as count
                FROM http_events WHERE "user" = '{user_id}'
        """).fetchall()
        
        stats['event_breakdown'] = {row[0]: row[1] for row in event_breakdown}
        
        return stats
    
    def close(self):
        """Close DuckDB connection"""
        self.conn.close()

# ============================================================================
# DUCKDB DATASET GENERATOR - CORRECTED
# ============================================================================

class DuckDBITSDatasetGenerator:
    """High-performance dataset generator using DuckDB with CORRECT columns"""
    
    def __init__(self, cert_path: str, output_dir: str):
        self.loader = DuckDBCERTLoader(cert_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_user_sequences(self, sequence_length: int = 50, stride: int = 1) -> Dict:
        """Generate user sequences using DuckDB for performance with CORRECT columns"""
        print(f"Generating sequences with length={sequence_length}, stride={stride}")
        
        # Get all users
        metadata = self.loader.get_dataset_metadata()
        all_users = list(metadata.malicious_users | metadata.benign_users)
        
        sequences = []
        
        for user_id in all_users:
            print(f"Processing user: {user_id}")
            
            # Get user events using CORRECT column structure
            user_events = self.loader.get_user_events(user_id)
            
            if len(user_events) < sequence_length:
                continue
            
            # Generate sequences using SQL window functions with ACTUAL columns
            sequence_query = f"""
            WITH user_events AS (
                SELECT *, ROW_NUMBER() OVER (ORDER BY "date") as row_num
                FROM (
                    SELECT 'logon' as event_type, id, "user", "date", pc, activity, NULL as filename, NULL as content, NULL as url
                    FROM logon WHERE "user" = '{user_id}'
                    UNION ALL
                    SELECT 'file' as event_type, id, "user", "date", pc, NULL as activity, filename, content, NULL as url
                    FROM file_events WHERE "user" = '{user_id}'
                    UNION ALL
                    SELECT 'device' as event_type, id, "user", "date", pc, activity, NULL as filename, NULL as content, NULL as url
                    FROM device_events WHERE "user" = '{user_id}'
                    UNION ALL
                    SELECT 'email' as event_type, id, "user", "date", pc, NULL as activity, NULL as filename, content, NULL as url
                    FROM email_events WHERE "user" = '{user_id}'
                    UNION ALL
                    SELECT 'http' as event_type, id, "user", "date", pc, NULL as activity, NULL as filename, content, url
                    FROM http_events WHERE "user" = '{user_id}'
                )
            )
            SELECT 
                event_type,
                id,
                "user",
                "date",
                pc,
                activity,
                filename,
                content,
                url,
                row_num
            FROM user_events
            ORDER BY row_num
            LIMIT 1000
            """
            
            user_events_df = self.loader.conn.execute(sequence_query).df()
            
            # Generate sequences in Python (could be optimized further with SQL)
            for i in range(0, len(user_events_df) - sequence_length + 1, stride):
                sequence = user_events_df.iloc[i:i + sequence_length]
                
                sequences.append({
                    'user_id': user_id,
                    'sequence_id': f"{user_id}_{i}",
                    'is_malicious': user_id in metadata.malicious_users,
                    'events': sequence.to_dict('records')
                })
        
        print(f"✓ Generated {len(sequences)} sequences")
        return {'sequences': sequences, 'metadata': metadata}
    
    def save_sequences(self, sequences_data: Dict, filename: str = "user_sequences.json"):
        """Save sequences to JSON file"""
        output_path = self.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(sequences_data, f, indent=2, default=str)
        print(f"✓ Sequences saved to {output_path}")
    
    def close(self):
        """Close the loader connection"""
        self.loader.close()

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    CERT_PATH = "../cert_r4.2"
    OUTPUT_DIR = "./datasets"
    
    # Initialize DuckDB generator
    print("Initializing DuckDB generator with CORRECT column structure...")
    generator = DuckDBITSDatasetGenerator(CERT_PATH, OUTPUT_DIR)
    
    # Get dataset overview
    metadata = generator.loader.get_dataset_metadata()
    print(f"CERT r4.2 Dataset Overview")
    print(f"=" * 50)
    print(f"Total users: {metadata.total_users}")
    print(f"Benign users: {len(metadata.benign_users)}")
    print(f"Malicious users: {len(metadata.malicious_users)}")
    print(f"Date range: {metadata.date_range[0]} to {metadata.date_range[1]}")
    print(f"Malicious user ratio: {len(metadata.malicious_users)/metadata.total_users:.2%}")
    
    # Test loading actual data
    print("\n=== Testing with ACTUAL CERT r4.2 data ===")
    
    # Test logon events
    logon_sample = generator.loader.conn.execute("SELECT * FROM logon LIMIT 3").df()
    print(f"Logon sample columns: {list(logon_sample.columns)}")
    print(f"Logon sample data:\n{logon_sample}")
    
    # Test file events
    file_sample = generator.loader.conn.execute("SELECT * FROM file_events LIMIT 3").df()
    print(f"\nFile sample columns: {list(file_sample.columns)}")
    print(f"File sample data:\n{file_sample}")
    
    # Test device events
    device_sample = generator.loader.conn.execute("SELECT * FROM device_events LIMIT 3").df()
    print(f"\nDevice sample columns: {list(device_sample.columns)}")
    print(f"Device sample data:\n{device_sample}")
    
    # Test email events
    email_sample = generator.loader.conn.execute("SELECT * FROM email_events LIMIT 3").df()
    print(f"\nEmail sample columns: {list(email_sample.columns)}")
    print(f"Email sample data:\n{email_sample}")
    
    # Test http events
    http_sample = generator.loader.conn.execute("SELECT * FROM http_events LIMIT 3").df()
    print(f"\nHTTP sample columns: {list(http_sample.columns)}")
    print(f"HTTP sample data:\n{http_sample}")
    
    # Generate sequences
    print("\nGenerating user sequences...")
    sequences_data = generator.generate_user_sequences(
        sequence_length=20,  # Smaller for testing
        stride=5
    )
    
    # Save results
    generator.save_sequences(sequences_data)
    
    # Close connection
    generator.close()
    print("\n✓ DuckDB processing complete with CORRECT column structure!")

