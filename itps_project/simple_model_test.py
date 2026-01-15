"""
Simple Model Test - Train a basic model on the processed data
"""

import json
import pandas as pd
import numpy as np
import re
from dataset_generator_corrected import FeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def load_and_process_data(max_sequences=1000):
    """Load and process a larger sample for model training"""
    print(f"Loading data (max {max_sequences} sequences)...")
    
    with open("./datasets/user_sequences.json", 'r') as f:
        data = json.load(f)
    
    sequences = data['sequences'][:max_sequences]
    metadata_str = data['metadata']
    
    # Parse metadata
    total_users_match = re.search(r"total_users=(\d+)", metadata_str)
    malicious_users_match = re.search(r"malicious_users=\{([^}]+)\}", metadata_str)
    
    metadata = {
        'total_users': int(total_users_match.group(1)) if total_users_match else 0,
        'malicious_users': set(malicious_users_match.group(1).split(', ')) if malicious_users_match else set(),
    }
    
    print(f"✓ Loaded {len(sequences)} sequences")
    print(f"  Benign: {sum(1 for seq in sequences if not seq['is_malicious'])}")
    print(f"  Malicious: {sum(1 for seq in sequences if seq['is_malicious'])}")
    
    return sequences, metadata

def convert_sequence_to_features(sequence):
    """Convert a sequence of events to feature matrix"""
    events = sequence['events']
    features = []
    
    for event in events:
        event_series = pd.Series(event)
        event_features = FeatureExtractor.extract_event_features(event_series)
        features.append(event_features)
    
    return np.array(features)

def create_sequence_features(sequences):
    """Create features from sequences"""
    print("Converting sequences to features...")
    
    feature_matrices = []
    labels = []
    
    for i, sequence in enumerate(sequences):
        if i % 100 == 0:
            print(f"  Processing sequence {i+1}/{len(sequences)}")
        
        try:
            features = convert_sequence_to_features(sequence)
            # Flatten the sequence features for traditional ML
            flattened_features = features.flatten()
            feature_matrices.append(flattened_features)
            
            label = 1 if not sequence['is_malicious'] else 0
            labels.append(label)
        except Exception as e:
            print(f"  Error processing sequence {i}: {e}")
            continue
    
    return np.array(feature_matrices), np.array(labels)

def train_simple_model():
    """Train a simple model on the data"""
    print("=" * 60)
    print("SIMPLE MODEL TRAINING")
    print("=" * 60)
    
    # Load and process data
    sequences, metadata = load_and_process_data(max_sequences=2000)  # Use more data for training
    
    # Convert to features
    X, y = create_sequence_features(sequences)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y)} (0=malicious, 1=benign)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain/Test split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print(f"\nTraining Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    print(f"\nModel Performance:")
    print(f"=" * 40)
    print(classification_report(y_test, y_pred, target_names=['Malicious', 'Benign']))
    
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = model.feature_importances_
    print(f"\nTop 10 Most Important Features:")
    top_features = np.argsort(feature_importance)[-10:][::-1]
    for i, feature_idx in enumerate(top_features):
        print(f"  {i+1}. Feature {feature_idx}: {feature_importance[feature_idx]:.4f}")
    
    print(f"\n✓ Model training completed successfully!")
    print(f"✓ Ready for full dataset processing and advanced model training")
    
    return model, scaler, X_test_scaled, y_test, y_pred

if __name__ == "__main__":
    try:
        model, scaler, X_test, y_test, y_pred = train_simple_model()
        print(f"\n🎉 SUCCESS! Your ITPS pipeline is working!")
        print(f"Next: Process full dataset and train advanced models")
    except Exception as e:
        print(f"\n❌ Error in model training: {e}")
        import traceback
        traceback.print_exc()
