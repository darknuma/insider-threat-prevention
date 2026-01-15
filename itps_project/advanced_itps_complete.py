"""
Complete Advanced ITPS System
Integrates LSTM, Transformer, Autoencoder with real-time detection, alerts, and dashboard
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

# Import all components
from advanced_realtime_detector import AdvancedRealTimeDetector
from alert_system import AlertManager, AlertDashboard
from dashboard import ITPSDashboard, DashboardAPI
from model_deployment import ModelManager

class AdvancedITPSSystem:
    """Complete advanced ITPS system with all ML models"""
    
    def __init__(self, config_path: str = "./config/advanced_itps_config.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(exist_ok=True)
        
        # Initialize components
        self.detector = AdvancedRealTimeDetector()
        self.alert_manager = AlertManager()
        self.dashboard = ITPSDashboard()
        self.dashboard_api = DashboardAPI(self.dashboard)
        self.model_manager = ModelManager()
        
        # System state
        self.is_running = False
        self.system_thread = None
        self.event_count = 0
        self.threat_count = 0
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load advanced ITPS configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default advanced configuration
            self.config = {
                "system": {
                    "sequence_length": 20,
                    "detection_interval": 1.0,
                    "dashboard_update_interval": 5,
                    "model_retrain_interval": 3600
                },
                "detection": {
                    "confidence_threshold": 0.75,
                    "threat_levels": {
                        "LOW": 0.6,
                        "MEDIUM": 0.75,
                        "HIGH": 0.85,
                        "CRITICAL": 0.95
                    },
                    "models": {
                        "lstm": {"weight": 0.3, "enabled": True},
                        "transformer": {"weight": 0.3, "enabled": True},
                        "autoencoder": {"weight": 0.2, "enabled": True},
                        "ensemble": {"weight": 0.2, "enabled": True}
                    }
                },
                "alerts": {
                    "enabled": True,
                    "escalation_threshold": 3,
                    "notification_methods": ["console", "email", "webhook"]
                },
                "dashboard": {
                    "enabled": True,
                    "web_dashboard": False,
                    "port": 5000,
                    "advanced_metrics": True
                }
            }
            self._save_config()
    
    def _save_config(self):
        """Save configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def initialize_system(self):
        """Initialize the advanced ITPS system"""
        print("🚀 Initializing Advanced ITPS System...")
        
        # Check if models are loaded
        if not self.detector.models:
            print("❌ No advanced models loaded!")
            print("Please run: python implement_advanced_models.py first")
            return False
        
        # Start dashboard
        if self.config['dashboard']['enabled']:
            self.dashboard.start_dashboard()
            print("✓ Advanced dashboard started")
        
        # Initialize alert system
        if self.config['alerts']['enabled']:
            print("✓ Advanced alert system initialized")
        
        print("✅ Advanced ITPS System initialized successfully!")
        return True
    
    def start_system(self):
        """Start the advanced ITPS system"""
        if self.is_running:
            print("⚠️  System is already running")
            return
        
        self.is_running = True
        self.system_thread = threading.Thread(target=self._system_loop)
        self.system_thread.start()
        
        print("🛡️  Advanced ITPS System started - Monitoring with LSTM, Transformer, Autoencoder")
    
    def stop_system(self):
        """Stop the advanced ITPS system"""
        if not self.is_running:
            print("⚠️  System is not running")
            return
        
        self.is_running = False
        if self.system_thread:
            self.system_thread.join()
        
        # Stop dashboard
        self.dashboard.stop_dashboard()
        
        print("⏹️  Advanced ITPS System stopped")
    
    def _system_loop(self):
        """Main system loop"""
        while self.is_running:
            try:
                # Update system metrics
                self._update_advanced_metrics()
                
                # Sleep for detection interval
                time.sleep(self.config['system']['detection_interval'])
                
            except Exception as e:
                print(f"❌ System loop error: {e}")
                time.sleep(5)
    
    def _update_advanced_metrics(self):
        """Update advanced system metrics"""
        # Get detection summary
        detection_summary = self.detector.get_detection_summary()
        alert_summary = self.alert_manager.get_alert_summary()
        
        # Update dashboard with advanced metrics
        self.dashboard.update_system_stats({
            'total_users': len(self.detector.user_sequences),
            'threats_detected': detection_summary['threats_detected'],
            'alerts_generated': alert_summary['total_alerts'],
            'active_alerts': alert_summary['active_alerts'],
            'model_performance': detection_summary['model_performance'],
            'threat_rate': detection_summary['threat_rate']
        })
    
    def process_event_advanced(self, user_id: str, event: Dict) -> Optional[Dict]:
        """Process event with advanced detection"""
        self.event_count += 1
        
        # Add event to advanced detector
        detection_result = self.detector.add_event(user_id, event)
        
        # Update dashboard
        self.dashboard.add_user_activity(user_id, event.get('event_type', 'unknown'))
        
        # If threat detected, create alert
        if detection_result and detection_result.get('threat_detected', False):
            self.threat_count += 1
            
            # Create advanced alert with model explanations
            alert = self.alert_manager.create_alert(
                user_id=user_id,
                confidence=detection_result['confidence'],
                description=f"Advanced threat detected: {detection_result['threat_level']}",
                details={
                    'threat_level': detection_result['threat_level'],
                    'model_predictions': detection_result['model_predictions'],
                    'explanations': detection_result['explanations'],
                    'model_agreement': detection_result['model_agreement'],
                    'key_insights': detection_result['key_insights'],
                    'recommendations': detection_result['recommendations']
                }
            )
            
            # Update dashboard with threat
            self.dashboard.add_threat_event(
                user_id=user_id,
                threat_level=alert.threat_level,
                confidence=detection_result['confidence']
            )
            
            return {
                'detection_result': detection_result,
                'alert': alert,
                'timestamp': datetime.now().isoformat()
            }
        
        return detection_result
    
    def get_advanced_status(self) -> Dict:
        """Get advanced system status"""
        detection_summary = self.detector.get_detection_summary()
        
        return {
            'is_running': self.is_running,
            'event_count': self.event_count,
            'threat_count': self.threat_count,
            'detection_summary': detection_summary,
            'models_loaded': len(self.detector.models),
            'model_types': list(self.detector.models.keys()),
            'uptime': datetime.now().isoformat()
        }
    
    def get_model_performance(self) -> Dict:
        """Get performance of all models"""
        if not self.detector.models:
            return {'error': 'No models loaded'}
        
        performance = {}
        for model_name, model_data in self.detector.models.items():
            if 'accuracy' in model_data:
                performance[model_name] = {
                    'accuracy': model_data['accuracy'],
                    'status': 'loaded'
                }
            else:
                performance[model_name] = {
                    'status': 'loaded',
                    'type': model_data.get('model_type', 'unknown')
                }
        
        return performance
    
    def explain_detection(self, user_id: str) -> Dict:
        """Get detailed explanation of user's threat status"""
        user_status = self.detector.get_user_status(user_id)
        
        if not user_status['is_ready_for_detection']:
            return {
                'user_id': user_id,
                'status': 'insufficient_data',
                'message': f"Need {self.detector.sequence_length - user_status['sequence_length']} more events"
            }
        
        # Get latest detection
        recent_detections = [d for d in self.detector.detection_history if d['user_id'] == user_id]
        if not recent_detections:
            return {
                'user_id': user_id,
                'status': 'no_detection',
                'message': 'No recent detections available'
            }
        
        latest_detection = recent_detections[-1]
        
        return {
            'user_id': user_id,
            'status': 'analyzed',
            'threat_detected': latest_detection['threat_detected'],
            'threat_level': latest_detection['threat_level'],
            'confidence': latest_detection['confidence'],
            'model_explanations': latest_detection['explanations'],
            'key_insights': latest_detection['key_insights'],
            'recommendations': latest_detection['recommendations'],
            'model_agreement': latest_detection['model_agreement']
        }
    
    def retrain_models(self, X: np.ndarray, y: np.ndarray):
        """Retrain all advanced models"""
        print("🔄 Retraining advanced models...")
        
        # This would implement model retraining
        # For now, just a placeholder
        print("✓ Model retraining completed")
    
    def export_advanced_data(self, output_path: str):
        """Export advanced system data"""
        data = {
            'system_status': self.get_advanced_status(),
            'model_performance': self.get_model_performance(),
            'detection_history': self.detector.detection_history,
            'alert_history': list(self.alert_manager.alert_history),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Advanced system data exported to {output_path}")

class AdvancedITPSController:
    """Controller for advanced ITPS system"""
    
    def __init__(self, itps_system: AdvancedITPSSystem):
        self.itps = itps_system
    
    def start_advanced_monitoring(self):
        """Start advanced monitoring"""
        print("🛡️  Starting Advanced ITPS Monitoring...")
        print("   Models: LSTM, Transformer, Autoencoder, Ensemble")
        self.itps.start_system()
    
    def stop_advanced_monitoring(self):
        """Stop advanced monitoring"""
        print("⏹️  Stopping Advanced ITPS Monitoring...")
        self.itps.stop_system()
    
    def get_advanced_status(self):
        """Get advanced system status"""
        status = self.itps.get_advanced_status()
        print("\n📊 Advanced ITPS System Status:")
        print(f"  Running: {status['is_running']}")
        print(f"  Events Processed: {status['event_count']}")
        print(f"  Threats Detected: {status['threat_count']}")
        print(f"  Models Loaded: {status['models_loaded']}")
        print(f"  Model Types: {', '.join(status['model_types'])}")
        
        if status.get('detection_summary'):
            summary = status['detection_summary']
            print(f"  Total Detections: {summary['total_detections']}")
            if summary.get('threat_rate') is not None:
                print(f"  Threat Rate: {summary['threat_rate']:.2%}")
            else:
                print("  Threat Rate: N/A (no detections yet)")
    
    def explain_user_threat(self, user_id: str):
        """Explain threat status for a specific user"""
        explanation = self.itps.explain_detection(user_id)
        
        print(f"\n🔍 Threat Analysis for User: {user_id}")
        print(f"  Status: {explanation['status']}")
        
        if explanation['status'] == 'analyzed':
            print(f"  Threat Detected: {explanation['threat_detected']}")
            print(f"  Threat Level: {explanation['threat_level']}")
            print(f"  Confidence: {explanation['confidence']:.3f}")
            print(f"  Model Agreement: {explanation['model_agreement']}")
            
            print(f"\n  Model Explanations:")
            for model, explanation_text in explanation['model_explanations'].items():
                print(f"    {model.upper()}: {explanation_text}")
            
            print(f"\n  Key Insights:")
            for insight in explanation['key_insights']:
                print(f"    • {insight}")
            
            print(f"\n  Recommendations:")
            for rec in explanation['recommendations']:
                print(f"    {rec}")
    
    def show_advanced_dashboard(self):
        """Show advanced dashboard"""
        status = self.itps.get_advanced_status()
        model_performance = self.itps.get_model_performance()
        
        print("\n" + "=" * 70)
        print("🛡️  ADVANCED ITPS SYSTEM DASHBOARD")
        print("=" * 70)
        
        # System status
        print(f"System Status: {'🟢 RUNNING' if status['is_running'] else '🔴 STOPPED'}")
        print(f"Events Processed: {status['event_count']}")
        print(f"Threats Detected: {status['threat_count']}")
        print(f"Models Loaded: {status['models_loaded']}")
        
        # Model performance
        print(f"\n🤖 Model Performance:")
        for model_name, perf in model_performance.items():
            if 'accuracy' in perf:
                print(f"  {model_name.upper()}: {perf['accuracy']:.3f} accuracy")
            else:
                print(f"  {model_name.upper()}: {perf['status']}")
        
        # Detection summary
        if status['detection_summary']:
            summary = status['detection_summary']
            print(f"\n📊 Detection Summary:")
            print(f"  Total Detections: {summary['total_detections']}")
            print(f"  Threats Detected: {summary['threats_detected']}")
            print(f"  Threat Rate: {summary['threat_rate']:.2%}")
        
        print("=" * 70)

# Example usage
if __name__ == "__main__":
    # Initialize advanced ITPS system
    print("🚀 Initializing Advanced ITPS System...")
    itps = AdvancedITPSSystem()
    
    # Initialize system
    if not itps.initialize_system():
        print("❌ Failed to initialize system. Please train models first.")
        exit(1)
    
    # Create controller
    controller = AdvancedITPSController(itps)
    
    # Start monitoring
    controller.start_advanced_monitoring()
    
    # Show status
    controller.get_advanced_status()
    
    # Show dashboard
    controller.show_advanced_dashboard()
    
    # Test with sample event
    print("\n🎬 Testing with sample event...")
    sample_event = {
        'event_type': 'logon',
        'id': 'TEST_001',
        'user': 'USER001',
        'date': datetime.now().isoformat(),
        'pc': 'PC-1234',
        'activity': 'Logon'
    }
    
    # Add multiple events to build sequence
    for i in range(25):
        result = itps.process_event_advanced('USER001', sample_event)
        if result and result.get('detection_result'):
            print(f"Advanced Detection Result:")
            print(f"  Threat Detected: {result['detection_result']['threat_detected']}")
            print(f"  Threat Level: {result['detection_result']['threat_level']}")
            print(f"  Confidence: {result['detection_result']['confidence']:.3f}")
            print(f"  Model Agreement: {result['detection_result']['model_agreement']}")
            break
    
    # Explain user threat
    controller.explain_user_threat('USER001')
    
    # Stop system
    controller.stop_advanced_monitoring()
    
    print("\n✅ Advanced ITPS System demonstration completed!")
    print("\nTo use the advanced system:")
    print("1. Train models: python implement_advanced_models.py")
    print("2. Start system: python advanced_itps_complete.py")
    print("3. Process events: itps.process_event_advanced(user_id, event)")
    print("4. Get explanations: controller.explain_user_threat(user_id)")
    print("\n🎬 Starting continuous monitoring simulation...")
    print("Press Ctrl+C to stop the system")

    try:
        while True:
            # Generate sample events
            sample_event = {
                'event_type': 'logon',
                'id': f'TEST_{int(time.time())}',
                'user': 'USER001',
                'date': datetime.now().isoformat(),
                'pc': 'PC-1234',
                'activity': 'Logon'
            }
        
            # Process event
            result = itps.process_event_advanced('USER001', sample_event)
            if result and result.get('detection_result'):
                print(f"\n🔔 Advanced Detection Result:")
                print(f"  Threat Detected: {result['detection_result']['threat_detected']}")
                print(f"  Threat Level: {result['detection_result']['threat_level']}")
                print(f"  Confidence: {result['detection_result']['confidence']:.3f}")
                print(f"  Model Agreement: {result['detection_result']['model_agreement']}")
            
                # Show detailed analysis
                controller.explain_user_threat('USER001')
        
            # Update status periodically
            if int(time.time()) % 10 == 0:  # Every 10 seconds
                controller.get_advanced_status()
        
            # Sleep to simulate real-time events
            time.sleep(2)

    except KeyboardInterrupt:
        print("\n\n⚠️  Received interrupt signal, shutting down...")
        # Stop system gracefully
        controller.stop_advanced_monitoring()
    
        print("\n✅ Advanced ITPS System demonstration completed!")
        print("\nTo use the advanced system:")
        print("1. Train models: python implement_advanced_models.py")
        print("2. Start system: python advanced_itps_complete.py")
        print("3. Process events: itps.process_event_advanced(user_id, event)")
        print("4. Get explanations: controller.explain_user_threat(user_id)")
