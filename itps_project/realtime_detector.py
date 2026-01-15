"""
Real-Time Insider Threat Detection System
Processes new user events and detects malicious behavior in real-time
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
import threading
import time
import pickle
from pathlib import Path
from dataset_generator_corrected import FeatureExtractor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class RealTimeDetector:
    """Real-time insider threat detection system"""
    
    def __init__(self, model_path: str = None, sequence_length: int = 20):
        self.sequence_length = sequence_length
        self.user_sequences = defaultdict(lambda: deque(maxlen=sequence_length))
        self.user_metadata = defaultdict(dict)
        self.detection_history = []
        self.model = None
        self.scaler = None
        self.feature_extractor = FeatureExtractor()
        self.is_running = False
        self.detection_thread = None
        
        # Load model if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load trained model and scaler"""
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def save_model(self, model_path: str):
        """Save trained model and scaler"""
        if self.model is None or self.scaler is None:
            print("❌ No model to save")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to {model_path}")
    
    def add_event(self, user_id: str, event: Dict) -> Optional[Dict]:
        """
        Add new event for a user and return detection result if sequence is complete
        
        Args:
            user_id: User identifier
            event: Event dictionary with fields like 'event_type', 'date', 'pc', etc.
        
        Returns:
            Detection result dictionary or None if sequence not complete
        """
        # Add event to user's sequence
        self.user_sequences[user_id].append(event)
        
        # Update user metadata
        self.user_metadata[user_id].update({
            'last_activity': datetime.now(),
            'total_events': len(self.user_sequences[user_id]),
            'last_pc': event.get('pc', 'unknown')
        })
        
        # Check if we have enough events for detection
        if len(self.user_sequences[user_id]) >= self.sequence_length:
            return self.detect_threat(user_id)
        
        return None
    
    def detect_threat(self, user_id: str) -> Dict:
        """Detect threat for a user's current sequence"""
        if self.model is None or self.scaler is None:
            return {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'threat_detected': False,
                'confidence': 0.0,
                'reason': 'No model loaded'
            }
        
        try:
            # Get user's sequence
            sequence = list(self.user_sequences[user_id])
            
            # Convert to features
            features = []
            for event in sequence:
                event_series = pd.Series(event)
                event_features = self.feature_extractor.extract_event_features(event_series)
                features.append(event_features)
            
            # Flatten features for model
            feature_matrix = np.array(features).flatten().reshape(1, -1)
            
            # Scale features
            feature_matrix_scaled = self.scaler.transform(feature_matrix)
            
            # Make prediction
            prediction = self.model.predict(feature_matrix_scaled)[0]
            probability = self.model.predict_proba(feature_matrix_scaled)[0]
            
            # Determine threat level
            threat_detected = prediction == 0  # 0 = malicious, 1 = benign
            confidence = max(probability)
            
            # Create detection result
            result = {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'threat_detected': threat_detected,
                'confidence': float(confidence),
                'malicious_probability': float(probability[0]),
                'benign_probability': float(probability[1]),
                'sequence_length': len(sequence),
                'user_metadata': dict(self.user_metadata[user_id])
            }
            
            # Add to detection history
            self.detection_history.append(result)
            
            return result
            
        except Exception as e:
            return {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'threat_detected': False,
                'confidence': 0.0,
                'reason': f'Detection error: {str(e)}'
            }
    
    def get_user_status(self, user_id: str) -> Dict:
        """Get current status for a user"""
        sequence = list(self.user_sequences[user_id])
        metadata = dict(self.user_metadata[user_id])
        
        return {
            'user_id': user_id,
            'sequence_length': len(sequence),
            'is_ready_for_detection': len(sequence) >= self.sequence_length,
            'last_activity': metadata.get('last_activity'),
            'total_events': metadata.get('total_events', 0),
            'last_pc': metadata.get('last_pc', 'unknown')
        }
    
    def get_detection_history(self, limit: int = 100) -> List[Dict]:
        """Get recent detection history"""
        return self.detection_history[-limit:]
    
    def get_threat_summary(self) -> Dict:
        """Get summary of threat detections"""
        if not self.detection_history:
            return {
                'total_detections': 0,
                'threats_detected': 0,
                'threat_rate': 0.0,
                'recent_threats': []
            }
        
        recent_detections = self.detection_history[-100:]  # Last 100 detections
        threats_detected = sum(1 for d in recent_detections if d['threat_detected'])
        
        return {
            'total_detections': len(self.detection_history),
            'threats_detected': threats_detected,
            'threat_rate': threats_detected / len(recent_detections) if recent_detections else 0.0,
            'recent_threats': [d for d in recent_detections if d['threat_detected']][-10:]
        }

class EventSimulator:
    """Simulate real-time events for testing"""
    
    def __init__(self, detector: RealTimeDetector):
        self.detector = detector
        self.is_running = False
        self.simulation_thread = None
    
    def start_simulation(self, duration: int = 60, event_rate: float = 1.0):
        """Start simulating events"""
        self.is_running = True
        self.simulation_thread = threading.Thread(
            target=self._simulate_events,
            args=(duration, event_rate)
        )
        self.simulation_thread.start()
        print(f"🎬 Started event simulation for {duration} seconds")
    
    def stop_simulation(self):
        """Stop event simulation"""
        self.is_running = False
        if self.simulation_thread:
            self.simulation_thread.join()
        print("⏹️ Event simulation stopped")
    
    def _simulate_events(self, duration: int, event_rate: float):
        """Simulate events at specified rate"""
        start_time = time.time()
        event_types = ['logon', 'file', 'device', 'email', 'http']
        users = ['USER001', 'USER002', 'USER003', 'USER004', 'USER005']
        
        while self.is_running and (time.time() - start_time) < duration:
            # Generate random event
            user_id = np.random.choice(users)
            event_type = np.random.choice(event_types)
            
            event = {
                'event_type': event_type,
                'id': f"EVENT_{int(time.time() * 1000)}",
                'user': user_id,
                'date': datetime.now().isoformat(),
                'pc': f"PC-{np.random.randint(1000, 9999)}",
                'activity': np.random.choice(['Logon', 'Logoff', 'Connect', 'Disconnect']) if event_type in ['logon', 'device'] else None,
                'filename': f"file_{np.random.randint(1, 100)}.txt" if event_type == 'file' else None,
                'url': f"https://example.com/page_{np.random.randint(1, 50)}" if event_type == 'http' else None,
                'content': f"Content for {event_type} event"
            }
            
            # Add event to detector
            result = self.detector.add_event(user_id, event)
            
            # Print detection result if available
            if result and result['threat_detected']:
                print(f"🚨 THREAT DETECTED: {user_id} - Confidence: {result['confidence']:.2f}")
            
            # Wait for next event
            time.sleep(1.0 / event_rate)

# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = RealTimeDetector(sequence_length=20)
    
    # Train a simple model for testing
    print("Training simple model for testing...")
    from simple_model_test import train_simple_model
    model, scaler, _, _, _ = train_simple_model()
    
    detector.model = model
    detector.scaler = scaler
    
    # Save model
    detector.save_model("./models/itps_model.pkl")
    
    # Test with simulated events
    print("\nTesting real-time detection...")
    simulator = EventSimulator(detector)
    simulator.start_simulation(duration=30, event_rate=2.0)
    
    # Monitor results
    time.sleep(35)
    simulator.stop_simulation()
    
    # Print summary
    summary = detector.get_threat_summary()
    print(f"\nDetection Summary:")
    print(f"  Total detections: {summary['total_detections']}")
    print(f"  Threats detected: {summary['threats_detected']}")
    print(f"  Threat rate: {summary['threat_rate']:.2%}")
    
    print("\n✓ Real-time detection system ready!")
