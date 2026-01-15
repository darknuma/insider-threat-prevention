"""
CERT r4.2 Dataset Generator - CORRECTED VERSION
Uses ACTUAL column structure from CERT r4.2 dataset
"""

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
# CERT DATASET LOADER - CORRECTED
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

class CERTDatasetLoader:
    """Load and validate CERT r4.2 dataset with CORRECT column structure"""
    
    def __init__(self, cert_root_path: str):
        """
        Initialize loader with path to CERT dataset root directory
        
        Args:
            cert_root_path: Path to CERT r4.2 root (e.g., '/data/cert_r4.2')
        """
        self.root = Path(cert_root_path)
        self.metadata = None
        self.events_cache = {}
        self._validate_structure()
    
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
    
    def load_insiders(self) -> pd.DataFrame:
        """Load insider metadata (malicious user definitions)"""
        df = pd.read_csv(self.root / 'insiders.csv')
        print(f"✓ Loaded {len(df)} insider records")
        return df
    
    def load_logon_events(self) -> pd.DataFrame:
        """Load logon/logoff events - CORRECTED for actual columns"""
        df = pd.read_csv(self.root / 'logon.csv')
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Loaded {len(df)} logon events ({df['user'].nunique()} unique users)")
        print(f"  Columns: {list(df.columns)}")
        return df
    
    def load_file_events(self) -> pd.DataFrame:
        """Load file access events - CORRECTED for actual columns"""
        df = pd.read_csv(self.root / 'file.csv')
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Loaded {len(df)} file events ({df['user'].nunique()} unique users)")
        print(f"  Columns: {list(df.columns)}")
        return df
    
    def load_device_events(self) -> pd.DataFrame:
        """Load device/USB events - CORRECTED for actual columns"""
        df = pd.read_csv(self.root / 'device.csv')
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Loaded {len(df)} device events ({df['user'].nunique()} unique users)")
        print(f"  Columns: {list(df.columns)}")
        return df
    
    def load_email_events(self) -> pd.DataFrame:
        """Load email communication events - CORRECTED for actual columns"""
        df = pd.read_csv(self.root / 'email.csv')
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Loaded {len(df)} email events ({df['user'].nunique()} unique users)")
        print(f"  Columns: {list(df.columns)}")
        return df
    
    def load_http_events(self) -> pd.DataFrame:
        """Load web browsing events - CORRECTED for actual columns"""
        df = pd.read_csv(self.root / 'http.csv')
        df['date'] = pd.to_datetime(df['date'])
        print(f"✓ Loaded {len(df)} http events ({df['user'].nunique()} unique users)")
        print(f"  Columns: {list(df.columns)}")
        return df
    
    def get_dataset_metadata(self) -> CERTMetadata:
        """Get comprehensive dataset metadata"""
        if self.metadata:
            return self.metadata
        
        # Load insiders to get malicious users
        insiders_df = self.load_insiders()
        malicious_users = set(insiders_df['user'].unique())
        
        # Get all users from logon events
        logon_df = self.load_logon_events()
        all_users = set(logon_df['user'].unique())
        
        benign_users = all_users - malicious_users
        
        # Get date range
        min_date = logon_df['date'].min()
        max_date = logon_df['date'].max()
        
        self.metadata = CERTMetadata(
            version="r4.2",
            path=str(self.root),
            total_users=len(all_users),
            date_range=(str(min_date), str(max_date)),
            malicious_users=malicious_users,
            benign_users=benign_users
        )
        
        return self.metadata

# ============================================================================
# DATA PROCESSOR - CORRECTED
# ============================================================================

class CERTDataProcessor:
    """Process CERT events with CORRECT column structure"""
    
    def __init__(self, loader: CERTDatasetLoader):
        self.loader = loader
        self.merged_events = None
    
    def merge_all_events(self) -> pd.DataFrame:
        """Merge all event types with CORRECT column handling"""
        print("Merging all event types...")
        
        # Load all event types
        logon_df = self.loader.load_logon_events()
        file_df = self.loader.load_file_events()
        device_df = self.loader.load_device_events()
        email_df = self.loader.load_email_events()
        http_df = self.loader.load_http_events()
        
        # Add event type column to each
        logon_df['event_type'] = 'logon'
        file_df['event_type'] = 'file'
        device_df['event_type'] = 'device'
        email_df['event_type'] = 'email'
        http_df['event_type'] = 'http'
        
        # Select common columns for merging
        common_cols = ['id', 'date', 'user', 'pc', 'event_type']
        
        # Prepare each dataframe with common columns
        logon_common = logon_df[common_cols + ['activity']].copy()
        file_common = file_df[common_cols + ['filename', 'content']].copy()
        device_common = device_df[common_cols + ['activity']].copy()
        email_common = email_df[common_cols + ['to', 'cc', 'bcc', 'from', 'size', 'attachments', 'content']].copy()
        http_common = http_df[common_cols + ['url', 'content']].copy()
        
        # Add missing columns with NaN for each type
        all_cols = set()
        for df in [logon_common, file_common, device_common, email_common, http_common]:
            all_cols.update(df.columns)
        
        # Add missing columns to each dataframe
        for df in [logon_common, file_common, device_common, email_common, http_common]:
            for col in all_cols:
                if col not in df.columns:
                    df[col] = np.nan
        
        # Merge all events
        merged = pd.concat([
            logon_common, file_common, device_common, email_common, http_common
        ], ignore_index=True)
        
        # Sort by user and date
        merged = merged.sort_values(['user', 'date']).reset_index(drop=True)
        
        self.merged_events = merged
        print(f"✓ Merged {len(merged)} events from all sources")
        print(f"  Event types: {merged['event_type'].value_counts().to_dict()}")
        
        return merged
    
    def create_user_sequences(self, sequence_length: int = 50, 
                            stride: int = 1) -> Dict[str, List[pd.DataFrame]]:
        """Create sequences for each user with CORRECT column handling"""
        if self.merged_events is None:
            self.merge_all_events()
        
        sequences = defaultdict(list)
        
        for user in self.merged_events['user'].unique():
            user_events = self.merged_events[self.merged_events['user'] == user]
            
            if len(user_events) < sequence_length:
                continue
            
            # Create overlapping sequences
            for i in range(0, len(user_events) - sequence_length + 1, stride):
                sequence = user_events.iloc[i:i + sequence_length].copy()
                sequences[user].append(sequence)
        
        print(f"✓ Created sequences for {len(sequences)} users")
        total_sequences = sum(len(seqs) for seqs in sequences.values())
        print(f"  Total sequences: {total_sequences}")
        
        return dict(sequences)
    
    def split_benign_malicious(self, sequences: Dict[str, List[pd.DataFrame]]) -> Tuple[Dict, Dict]:
        """Split sequences by user type"""
        metadata = self.loader.get_dataset_metadata()
        
        benign_sequences = {}
        malicious_sequences = {}
        
        for user, user_sequences in sequences.items():
            if user in metadata.malicious_users:
                malicious_sequences[user] = user_sequences
            else:
                benign_sequences[user] = user_sequences
        
        print(f"✓ Split sequences:")
        print(f"  Benign users: {len(benign_sequences)}")
        print(f"  Malicious users: {len(malicious_sequences)}")
        
        return benign_sequences, malicious_sequences

# ============================================================================
# FEATURE EXTRACTOR - CORRECTED
# ============================================================================

class FeatureExtractor:
    """Extract features from events using ACTUAL column structure"""
    
    @staticmethod
    def extract_event_features(event_row: pd.Series) -> np.ndarray:
        """Extract features from a single event using ACTUAL columns"""
        features = []
        
        # Time features (from date column)
        if pd.notna(event_row['date']):
            dt = pd.to_datetime(event_row['date'])
            hour = dt.hour
            day_of_week = dt.dayofweek
            
            # Cyclical encoding
            features.extend([
                np.sin(2 * np.pi * hour / 24),      # hour_sin
                np.cos(2 * np.pi * hour / 24),      # hour_cos
                np.sin(2 * np.pi * day_of_week / 7), # day_sin
                np.cos(2 * np.pi * day_of_week / 7)  # day_cos
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Event type (one-hot encoding)
        event_type = event_row.get('event_type', 'unknown')
        event_types = ['logon', 'file', 'device', 'email', 'http']
        event_one_hot = [1 if event_type == et else 0 for et in event_types]
        features.extend(event_one_hot)
        
        # Activity features (for logon/device events)
        activity = event_row.get('activity', '')
        if activity in ['Logon', 'Connect']:
            features.extend([1, 0])  # [is_logon, is_logoff]
        elif activity in ['Logoff', 'Disconnect']:
            features.extend([0, 1])
        else:
            features.extend([0, 0])
        
        # File features (if available)
        filename = event_row.get('filename', '')
        if pd.notna(filename) and filename:
            # Simple file type detection
            is_sensitive = any(ext in filename.lower() for ext in ['.pdf', '.doc', '.xls', '.ppt'])
            features.extend([1 if is_sensitive else 0])
        else:
            features.extend([0])
        
        # Email features (if available)
        if event_row.get('event_type') == 'email':
            to_count = len(str(event_row.get('to', '')).split(',')) if pd.notna(event_row.get('to')) else 0
            has_attachments = pd.notna(event_row.get('attachments')) and str(event_row.get('attachments')) != ''
            features.extend([min(to_count, 5), 1 if has_attachments else 0])
        else:
            features.extend([0, 0])
        
        # HTTP features (if available)
        if event_row.get('event_type') == 'http':
            url = str(event_row.get('url', ''))
            is_external = any(domain in url.lower() for domain in ['google', 'yahoo', 'facebook', 'twitter'])
            features.extend([1 if is_external else 0])
        else:
            features.extend([0])
        
        # Pad to 32 dimensions
        while len(features) < 32:
            features.append(0)
        
        return np.array(features[:32])
    
    @staticmethod
    def sequence_to_features(sequence: pd.DataFrame) -> np.ndarray:
        """Convert sequence to feature matrix"""
        features = []
        for _, event in sequence.iterrows():
            event_features = FeatureExtractor.extract_event_features(event)
            features.append(event_features)
        
        return np.array(features)

# ============================================================================
# MISSING CLASSES FOR COMPATIBILITY
# ============================================================================

class DatasetSplitter:
    """Create train/val/test splits"""
    
    def __init__(self, benign_sequences: Dict, malicious_sequences: Dict):
        self.benign_sequences = benign_sequences
        self.malicious_sequences = malicious_sequences
    
    def create_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict:
        """Create train/val/test splits"""
        # Simple implementation - in practice, you'd want more sophisticated splitting
        splits = {
            'train': {'benign': list(self.benign_sequences.values())[:int(len(self.benign_sequences) * train_ratio)], 'malicious': []},
            'val': {'benign': [], 'malicious': list(self.malicious_sequences.values())[:int(len(self.malicious_sequences) * val_ratio)]},
            'test': {'benign': list(self.benign_sequences.values())[int(len(self.benign_sequences) * train_ratio):], 
                    'malicious': list(self.malicious_sequences.values())[int(len(self.malicious_sequences) * val_ratio):]}
        }
        return splits
    
    def save_splits(self, output_dir: str, splits: Dict):
        """Save splits to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, split_data in splits.items():
            split_path = output_path / split_name
            split_path.mkdir(exist_ok=True)
            
            # Save benign sequences
            if split_data['benign']:
                benign_data = np.concatenate([seq.values for seq in split_data['benign']])
                np.save(split_path / 'benign.npy', benign_data)
            
            # Save malicious sequences
            if split_data['malicious']:
                malicious_data = np.concatenate([seq.values for seq in split_data['malicious']])
                np.save(split_path / 'malicious.npy', malicious_data)
            
            # Save labels
            labels = np.concatenate([
                np.ones(len(split_data['benign'])) if split_data['benign'] else np.array([]),
                np.zeros(len(split_data['malicious'])) if split_data['malicious'] else np.array([])
            ])
            np.save(split_path / 'labels.npy', labels)

class DatasetAnalyzer:
    """Analyze dataset statistics"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
    
    def analyze_split_statistics(self, split: str) -> Dict:
        """Analyze statistics for a split"""
        split_path = self.dataset_path / split
        
        stats = {}
        
        # Load data if it exists
        if (split_path / 'benign.npy').exists():
            benign_data = np.load(split_path / 'benign.npy')
            stats['sequences'] = {'benign': {'count': len(benign_data)}}
        
        if (split_path / 'malicious.npy').exists():
            malicious_data = np.load(split_path / 'malicious.npy')
            stats['sequences']['malicious'] = {'count': len(malicious_data)}
        
        if (split_path / 'labels.npy').exists():
            labels = np.load(split_path / 'labels.npy')
            stats['labels'] = {
                'benign_count': int(np.sum(labels == 1)),
                'malicious_count': int(np.sum(labels == 0)),
                'class_balance': float(np.mean(labels))
            }
        
        return stats
    
    def generate_report(self) -> str:
        """Generate comprehensive report"""
        report = "DATASET ANALYSIS REPORT\n"
        report += "=" * 70 + "\n\n"
        
        for split in ['train', 'val', 'test']:
            stats = self.analyze_split_statistics(split)
            report += f"{split.upper()} Split:\n"
            
            if 'sequences' in stats:
                for seq_type, seq_stats in stats['sequences'].items():
                    report += f"  {seq_type.title()} sequences: {seq_stats.get('count', 0)}\n"
            
            if 'labels' in stats:
                report += f"  Class balance: {stats['labels']['class_balance']:.2%} benign\n"
        
        return report

class ITSDataLoader:
    """Data loader for training"""
    
    def __init__(self, dataset_path: str, split: str = 'train', batch_size: int = 32, shuffle: bool = True):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._load_data()
    
    def _load_data(self):
        """Load data for the split"""
        split_path = self.dataset_path / self.split
        
        sequences = []
        labels = []
        
        # Load benign data
        if (split_path / 'benign.npy').exists():
            benign_data = np.load(split_path / 'benign.npy')
            sequences.append(benign_data)
            labels.extend([1] * len(benign_data))
        
        # Load malicious data
        if (split_path / 'malicious.npy').exists():
            malicious_data = np.load(split_path / 'malicious.npy')
            sequences.append(malicious_data)
            labels.extend([0] * len(malicious_data))
        
        if sequences:
            self.sequences = np.concatenate(sequences)
            self.labels = np.array(labels)
        else:
            self.sequences = np.array([])
            self.labels = np.array([])
    
    def __len__(self):
        return len(self.sequences)
    
    def __iter__(self):
        """Iterate over batches"""
        if self.shuffle:
            indices = np.random.permutation(len(self.sequences))
        else:
            indices = np.arange(len(self.sequences))
        
        for i in range(0, len(self.sequences), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield self.sequences[batch_indices], self.labels[batch_indices]
    
    @property
    def data(self):
        """Get all data"""
        return self.sequences, self.labels
    
    def get_class_weights(self) -> Dict:
        """Get class weights for imbalanced data"""
        if len(self.labels) == 0:
            return {0: 1.0, 1: 1.0}
        
        class_counts = np.bincount(self.labels.astype(int))
        total = len(self.labels)
        
        weights = {}
        for i, count in enumerate(class_counts):
            if count > 0:
                weights[i] = total / (len(class_counts) * count)
            else:
                weights[i] = 1.0
        
        return weights

class ITSDatasetGenerator:
    """Complete pipeline for ITPS training dataset generation"""
    
    def __init__(self, cert_path: str, output_dir: str):
        self.cert_path = cert_path
        self.output_dir = Path(output_dir)
        self.loader = CERTDatasetLoader(cert_path)
        self.processor = CERTDataProcessor(self.loader)
    
    def generate_baseline_dataset(self, sequence_length: int = 50, stride: int = 1) -> Dict:
        """Generate baseline dataset"""
        print("=" * 70)
        print("BASELINE DATASET GENERATION")
        print("=" * 70)
        
        # Merge and sequence
        self.processor.merge_all_events()
        sequences = self.processor.create_user_sequences(
            sequence_length=sequence_length,
            stride=stride
        )
        
        # Split benign/malicious
        benign_seqs, malicious_seqs = self.processor.split_benign_malicious(sequences)
        
        # Create train/val/test splits
        splitter = DatasetSplitter(benign_seqs, malicious_seqs)
        splits = splitter.create_splits()
        
        # Save splits
        splitter.save_splits(str(self.output_dir / 'baseline'), splits)
        
        return splits
    
    def generate_augmented_dataset(self, sequences: Dict, threat_types: List[str] = None) -> Dict:
        """Generate augmented dataset with synthetic threats"""
        print("\n" + "=" * 70)
        print("AUGMENTED DATASET GENERATION (SYNTHETIC THREATS)")
        print("=" * 70)
        
        # For now, just return the baseline sequences
        # In a full implementation, you'd inject synthetic threats here
        print("✓ Augmented dataset generation (placeholder)")
        return sequences

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Test with actual CERT r4.2 data
    CERT_PATH = "../cert_r4.2"
    
    print("Testing with ACTUAL CERT r4.2 column structure...")
    
    # Initialize loader
    loader = CERTDatasetLoader(CERT_PATH)
    
    # Test loading each file
    print("\n=== Testing File Loading ===")
    logon_df = loader.load_logon_events()
    print(f"Logon columns: {list(logon_df.columns)}")
    print(f"Sample logon data:\n{logon_df.head(2)}")
    
    file_df = loader.load_file_events()
    print(f"\nFile columns: {list(file_df.columns)}")
    print(f"Sample file data:\n{file_df.head(2)}")
    
    device_df = loader.load_device_events()
    print(f"\nDevice columns: {list(device_df.columns)}")
    print(f"Sample device data:\n{device_df.head(2)}")
    
    email_df = loader.load_email_events()
    print(f"\nEmail columns: {list(email_df.columns)}")
    print(f"Sample email data:\n{email_df.head(2)}")
    
    http_df = loader.load_http_events()
    print(f"\nHTTP columns: {list(http_df.columns)}")
    print(f"Sample HTTP data:\n{http_df.head(2)}")
    
    # Test metadata
    print("\n=== Testing Metadata ===")
    metadata = loader.get_dataset_metadata()
    print(f"Total users: {metadata.total_users}")
    print(f"Malicious users: {len(metadata.malicious_users)}")
    print(f"Benign users: {len(metadata.benign_users)}")
    
    # Test processing
    print("\n=== Testing Data Processing ===")
    processor = CERTDataProcessor(loader)
    merged = processor.merge_all_events()
    print(f"Merged events shape: {merged.shape}")
    print(f"Merged columns: {list(merged.columns)}")
    
    # Test feature extraction
    print("\n=== Testing Feature Extraction ===")
    sample_event = merged.iloc[0]
    features = FeatureExtractor.extract_event_features(sample_event)
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature vector: {features}")
    
    print("\n✓ All tests passed with ACTUAL CERT r4.2 column structure!")
