"""
Advanced Real-Time Detector with LSTM, Transformer, and Autoencoder
Uses your trained models for real-time threat detection
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import deque, defaultdict
import joblib
from pathlib import Path
from dataset_generator_corrected import FeatureExtractor

class AdvancedRealTimeDetector:
    """Advanced real-time detector using multiple ML models"""
    
    def __init__(self, sequence_length: int = 20, models_dir: str = "./models"):
        self.sequence_length = sequence_length
        self.models_dir = Path(models_dir)
        self.user_sequences = defaultdict(lambda: deque(maxlen=sequence_length))
        self.user_metadata = defaultdict(dict)
        self.detection_history = []
        self.feature_extractor = FeatureExtractor()
        
        # Load trained models
        self.models = {}
        self._load_models()
    
    def _load_models(self):
        """Load all trained models"""
        print("Loading advanced models...")
        
        try:
            # Load LSTM model
            lstm_data = joblib.load(self.models_dir / 'lstm_model.pkl')
            self.models['lstm'] = {
                'model': lstm_data['model'],
                'scaler': lstm_data['scaler'],
                'accuracy': lstm_data['accuracy']
            }
            print("✓ LSTM model loaded")
            
            # Load Transformer model
            transformer_data = joblib.load(self.models_dir / 'transformer_model.pkl')
            self.models['transformer'] = {
                'model': transformer_data['model'],
                'scaler': transformer_data['scaler'],
                'accuracy': transformer_data['accuracy']
            }
            print("✓ Transformer model loaded")
            
            # Load Autoencoder model
            autoencoder_data = joblib.load(self.models_dir / 'autoencoder_model.pkl')
            self.models['autoencoder'] = {
                'model': autoencoder_data['model'],
                'scaler': autoencoder_data['scaler'],
                'threshold': autoencoder_data['threshold']
            }
            print("✓ Autoencoder model loaded")
            
            # Load Ensemble model
            ensemble_data = joblib.load(self.models_dir / 'ensemble_model.pkl')
            self.models['ensemble'] = {
                'model': ensemble_data['model'],
                'lstm_scaler': ensemble_data['lstm_scaler'],
                'transformer_scaler': ensemble_data['transformer_scaler'],
                'autoencoder_scaler': ensemble_data['autoencoder_scaler'],
                'autoencoder_threshold': ensemble_data['autoencoder_threshold'],
                'accuracy': ensemble_data['accuracy']
            }
            print("✓ Ensemble model loaded")
            
            print("✅ All advanced models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Make sure to run implement_advanced_models.py first")
    
    def add_event(self, user_id: str, event: Dict) -> Optional[Dict]:
        """Add new event and return advanced detection result"""
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
            return self.detect_threat_advanced(user_id)
        
        return None
    
    def detect_threat_advanced(self, user_id: str) -> Dict:
        """Advanced threat detection using all models"""
        try:
            # Get user's sequence
            sequence = list(self.user_sequences[user_id])
            
            # Convert to features
            features = self._convert_sequence_to_features(sequence)
            
            # Get predictions from all models
            predictions = {}
            explanations = {}
            
            # LSTM Prediction
            lstm_pred = self._get_lstm_prediction(features)
            predictions['lstm'] = lstm_pred
            explanations['lstm'] = self._explain_lstm_prediction(lstm_pred)
            
            # Transformer Prediction
            transformer_pred = self._get_transformer_prediction(features)
            predictions['transformer'] = transformer_pred
            explanations['transformer'] = self._explain_transformer_prediction(transformer_pred)
            
            # Autoencoder Prediction
            autoencoder_pred, reconstruction_error = self._get_autoencoder_prediction(features)
            predictions['autoencoder'] = autoencoder_pred
            explanations['autoencoder'] = self._explain_autoencoder_prediction(autoencoder_pred, reconstruction_error)
            
            # Ensemble Prediction
            ensemble_pred = self._get_ensemble_prediction(features)
            predictions['ensemble'] = ensemble_pred
            explanations['ensemble'] = self._explain_ensemble_prediction(ensemble_pred, predictions)
            
            # Determine final threat level
            threat_level = self._determine_threat_level(ensemble_pred)
            threat_detected = ensemble_pred > 0.7
            
            # Create comprehensive result
            result = {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'threat_detected': threat_detected,
                'threat_level': threat_level,
                'confidence': float(ensemble_pred),
                'model_predictions': predictions,
                'explanations': explanations,
                'model_agreement': self._calculate_model_agreement(predictions),
                'key_insights': self._generate_key_insights(predictions, explanations),
                'recommendations': self._generate_recommendations(threat_level, predictions),
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
                'error': f'Detection error: {str(e)}'
            }
    
    def _convert_sequence_to_features(self, sequence):
        """Convert sequence to feature matrix"""
        features = []
        for event in sequence:
            event_series = pd.Series(event)
            event_features = self.feature_extractor.extract_event_features(event_series)
            features.append(event_features)
        return np.array(features)
    
    def _get_lstm_prediction(self, features):
        """Get LSTM model prediction"""
        lstm_data = self.models['lstm']
        
        # Reshape for LSTM (1, sequence_length, features_per_event)
        features_reshaped = features.reshape(1, -1)
        
        # Scale features
        features_scaled = lstm_data['scaler'].transform(features_reshaped)
        
        # Get prediction
        prediction = lstm_data['model'].predict_proba(features_scaled)[0][1]
        return float(prediction)
    
    def _get_transformer_prediction(self, features):
        """Get Transformer model prediction"""
        transformer_data = self.models['transformer']
        
        # Reshape for Transformer
        features_reshaped = features.reshape(1, -1)
        
        # Scale features
        features_scaled = transformer_data['scaler'].transform(features_reshaped)
        
        # Get prediction
        prediction = transformer_data['model'].predict_proba(features_scaled)[0][1]
        return float(prediction)
    
    def _get_autoencoder_prediction(self, features):
        """Get Autoencoder anomaly detection"""
        autoencoder_data = self.models['autoencoder']
        
        # Flatten features
        features_flat = features.reshape(1, -1)
        
        # Scale features
        features_scaled = autoencoder_data['scaler'].transform(features_flat)
        
        # Get reconstruction
        reconstructed = autoencoder_data['model'].inverse_transform(
            autoencoder_data['model'].transform(features_scaled)
        )
        
        # Calculate reconstruction error
        reconstruction_error = np.mean((features_scaled - reconstructed) ** 2)
        
        # Determine if anomaly
        is_anomaly = reconstruction_error > autoencoder_data['threshold']
        anomaly_score = min(reconstruction_error / autoencoder_data['threshold'], 1.0)
        
        return float(anomaly_score), float(reconstruction_error)
    
    def _get_ensemble_prediction(self, features):
        """Get ensemble model prediction"""
        ensemble_data = self.models['ensemble']
        
        # Flatten features
        features_flat = features.reshape(1, -1)
        
        # Scale features
        features_scaled = ensemble_data['lstm_scaler'].transform(features_flat)
        
        # Get prediction
        prediction = ensemble_data['model'].predict_proba(features_scaled)[0][1]
        return float(prediction)
    
    def _determine_threat_level(self, confidence):
        """Determine threat level based on confidence"""
        if confidence >= 0.95:
            return "CRITICAL"
        elif confidence >= 0.85:
            return "HIGH"
        elif confidence >= 0.75:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_model_agreement(self, predictions):
        """Calculate agreement between models"""
        values = [predictions['lstm'], predictions['transformer'], predictions['autoencoder']]
        std_dev = np.std(values)
        
        if std_dev < 0.1:
            return "High agreement - all models agree"
        elif std_dev < 0.2:
            return "Moderate agreement - most models agree"
        else:
            return "Low agreement - models disagree"
    
    def _explain_lstm_prediction(self, prediction):
        """Explain LSTM prediction"""
        if prediction > 0.8:
            return "LSTM detected strong temporal threat pattern"
        elif prediction > 0.6:
            return "LSTM detected moderate temporal threat pattern"
        else:
            return "LSTM detected normal temporal pattern"
    
    def _explain_transformer_prediction(self, prediction):
        """Explain Transformer prediction"""
        if prediction > 0.8:
            return "Transformer detected high-attention threat events"
        elif prediction > 0.6:
            return "Transformer detected moderate-attention events"
        else:
            return "Transformer detected normal event patterns"
    
    def _explain_autoencoder_prediction(self, prediction, reconstruction_error):
        """Explain Autoencoder prediction"""
        if prediction > 0.8:
            return f"Autoencoder detected highly anomalous behavior (error: {reconstruction_error:.4f})"
        elif prediction > 0.6:
            return f"Autoencoder detected moderately anomalous behavior (error: {reconstruction_error:.4f})"
        else:
            return "Autoencoder detected normal behavior pattern"
    
    def _explain_ensemble_prediction(self, prediction, individual_predictions):
        """Explain ensemble prediction"""
        if prediction > 0.8:
            return "Ensemble: High confidence threat detected across multiple models"
        elif prediction > 0.6:
            return "Ensemble: Moderate confidence threat detected"
        else:
            return "Ensemble: Normal behavior detected"
    
    def _generate_key_insights(self, predictions, explanations):
        """Generate key insights from all models"""
        insights = []
        
        # LSTM insights
        if predictions['lstm'] > 0.7:
            insights.append("Temporal pattern analysis indicates suspicious behavior")
        
        # Transformer insights
        if predictions['transformer'] > 0.7:
            insights.append("Event attention analysis highlights concerning activities")
        
        # Autoencoder insights
        if predictions['autoencoder'] > 0.7:
            insights.append("Anomaly detection identified unusual behavior patterns")
        
        # Ensemble insights
        if predictions['ensemble'] > 0.8:
            insights.append("Multiple models agree on high threat probability")
        
        return insights if insights else ["No significant threat indicators detected"]
    
    def _generate_recommendations(self, threat_level, predictions):
        """Generate recommendations based on threat level"""
        recommendations = []
        
        if threat_level == "CRITICAL":
            recommendations.extend([
                "🚨 IMMEDIATE: Investigate user activity immediately",
                "🔒 SECURITY: Consider immediate access restrictions",
                "📊 MONITOR: Increase real-time surveillance"
            ])
        elif threat_level == "HIGH":
            recommendations.extend([
                "⚠️ URGENT: Investigate user activity within 1 hour",
                "📊 MONITOR: Increase surveillance on this user",
                "🔍 REVIEW: Check recent file access patterns"
            ])
        elif threat_level == "MEDIUM":
            recommendations.extend([
                "📋 REVIEW: Investigate user activity within 24 hours",
                "👀 MONITOR: Keep under observation",
                "📝 LOG: Document for future reference"
            ])
        else:
            recommendations.append("✅ NORMAL: Continue routine monitoring")
        
        return recommendations
    
    def get_user_status(self, user_id: str) -> Dict:
        """Get current status of a user's sequence"""
        sequence = self.user_sequences[user_id]
        metadata = self.user_metadata.get(user_id, {})
        
        return {
            'user_id': user_id,
            'sequence_length': len(sequence),
            'is_ready_for_detection': len(sequence) >= self.sequence_length,
            'last_activity': metadata.get('last_activity'),
            'total_events': metadata.get('total_events', 0),
            'last_pc': metadata.get('last_pc', 'unknown')
        }

    def get_detection_summary(self):
        """Get summary of all detections"""
        if not self.detection_history:
            return {
                'total_detections': 0,
                'threats_detected': 0,
                'model_performance': {},
                'threat_rate': 0.0
            }
        
        threats_detected = sum(1 for d in self.detection_history if d['threat_detected'])
        
        # Calculate model performance
        model_performance = {}
        for model_name in ['lstm', 'transformer', 'autoencoder', 'ensemble']:
            if model_name in self.detection_history[0]['model_predictions']:
                predictions = [d['model_predictions'][model_name] for d in self.detection_history]
                model_performance[model_name] = {
                    'avg_confidence': np.mean(predictions),
                    'max_confidence': np.max(predictions),
                    'min_confidence': np.min(predictions)
                }
        
        return {
            'total_detections': len(self.detection_history),
            'threats_detected': threats_detected,
            'threat_rate': threats_detected / len(self.detection_history),
            'model_performance': model_performance
        }

# Example usage
if __name__ == "__main__":
    # Initialize advanced detector
    detector = AdvancedRealTimeDetector()
    
    # Test with sample event
    sample_event = {
        'event_type': 'logon',
        'id': 'TEST_001',
        'user': 'USER001',
        'date': datetime.now().isoformat(),
        'pc': 'PC-1234',
        'activity': 'Logon'
    }
    
    # Add multiple events to build sequence
    for i in range(20):
        result = detector.add_event('USER001', sample_event)
        if result:
            print(f"Detection Result:")
            print(f"  Threat Detected: {result['threat_detected']}")
            print(f"  Threat Level: {result['threat_level']}")
            print(f"  Confidence: {result['confidence']:.3f}")
            print(f"  Model Agreement: {result['model_agreement']}")
            print(f"  Key Insights: {result['key_insights']}")
            break
    
    # Get summary
    summary = detector.get_detection_summary()
    print(f"\nDetection Summary:")
    print(f"  Total Detections: {summary['total_detections']}")
    print(f"  Threats Detected: {summary['threats_detected']}")
    print(f"  Threat Rate: {summary['threat_rate']:.2%}")
    
    print("\n✅ Advanced real-time detector ready!")
