"""
Complete ITPS (Insider Threat Prevention System)
Integrates all components: detection, alerts, dashboard, and model management
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd

# Import all ITPS components
from realtime_detector import RealTimeDetector, EventSimulator
from model_deployment import ModelManager, ModelEvaluator
from alert_system import AlertManager, AlertDashboard
from dashboard import ITPSDashboard, WebDashboard, DashboardAPI

class ITPSSystem:
    """Complete ITPS system integrating all components"""
    
    def __init__(self, config_path: str = "./config/itps_config.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(exist_ok=True)
        
        # Initialize components
        self.detector = RealTimeDetector(sequence_length=20)
        self.model_manager = ModelManager()
        self.alert_manager = AlertManager()
        self.dashboard = ITPSDashboard()
        self.dashboard_api = DashboardAPI(self.dashboard)
        
        # System state
        self.is_running = False
        self.system_thread = None
        self.event_count = 0
        self.threat_count = 0
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load ITPS system configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                "system": {
                    "sequence_length": 20,
                    "detection_interval": 1.0,
                    "dashboard_update_interval": 5,
                    "model_retrain_interval": 3600  # 1 hour
                },
                "detection": {
                    "confidence_threshold": 0.75,
                    "threat_levels": {
                        "LOW": 0.6,
                        "MEDIUM": 0.75,
                        "HIGH": 0.85,
                        "CRITICAL": 0.95
                    }
                },
                "alerts": {
                    "enabled": True,
                    "escalation_threshold": 5,
                    "notification_methods": ["console", "email", "webhook"]
                },
                "dashboard": {
                    "enabled": True,
                    "web_dashboard": False,
                    "port": 5000
                }
            }
            self._save_config()
    
    def _save_config(self):
        """Save ITPS system configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def initialize_system(self, model_path: str = None):
        """Initialize the ITPS system"""
        print("🚀 Initializing ITPS System...")
        
        # Load model if provided
        if model_path and Path(model_path).exists():
            self.detector.load_model(model_path)
            print("✓ Model loaded")
        else:
            print("⚠️  No model loaded - system will use default detection")
        
        # Start dashboard
        if self.config['dashboard']['enabled']:
            self.dashboard.start_dashboard()
            print("✓ Dashboard started")
        
        # Initialize alert system
        if self.config['alerts']['enabled']:
            print("✓ Alert system initialized")
        
        print("✅ ITPS System initialized successfully!")
    
    def start_system(self):
        """Start the ITPS system"""
        if self.is_running:
            print("⚠️  System is already running")
            return
        
        self.is_running = True
        self.system_thread = threading.Thread(target=self._system_loop)
        self.system_thread.start()
        
        print("🛡️  ITPS System started - Monitoring for insider threats")
    
    def stop_system(self):
        """Stop the ITPS system"""
        if not self.is_running:
            print("⚠️  System is not running")
            return
        
        self.is_running = False
        if self.system_thread:
            self.system_thread.join()
        
        # Stop dashboard
        self.dashboard.stop_dashboard()
        
        print("⏹️  ITPS System stopped")
    
    def _system_loop(self):
        """Main system loop"""
        while self.is_running:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Check for model retraining
                self._check_model_retraining()
                
                # Sleep for detection interval
                time.sleep(self.config['system']['detection_interval'])
                
            except Exception as e:
                print(f"❌ System loop error: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _update_system_metrics(self):
        """Update system metrics"""
        # Get current metrics from components
        threat_summary = self.detector.get_threat_summary()
        alert_summary = self.alert_manager.get_alert_summary()
        
        # Update dashboard
        self.dashboard.update_system_stats({
            'total_users': threat_summary.get('total_detections', 0),
            'threats_detected': threat_summary.get('threats_detected', 0),
            'alerts_generated': alert_summary.get('total_alerts', 0),
            'active_alerts': alert_summary.get('active_alerts', 0)
        })
    
    def _check_model_retraining(self):
        """Check if model needs retraining"""
        # This would implement model retraining logic
        # For now, just a placeholder
        pass
    
    def process_event(self, user_id: str, event: Dict) -> Optional[Dict]:
        """Process a new event through the ITPS system"""
        self.event_count += 1
        
        # Add event to detector
        detection_result = self.detector.add_event(user_id, event)
        
        # Update dashboard with user activity
        self.dashboard.add_user_activity(user_id, event.get('event_type', 'unknown'))
        
        # If threat detected, create alert
        if detection_result and detection_result.get('threat_detected', False):
            self.threat_count += 1
            
            # Create alert
            alert = self.alert_manager.create_alert(
                user_id=user_id,
                confidence=detection_result['confidence'],
                description=f"Threat detected: {detection_result.get('reason', 'Unknown')}",
                details=detection_result
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
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'is_running': self.is_running,
            'event_count': self.event_count,
            'threat_count': self.threat_count,
            'detector_status': 'active' if self.detector.model else 'no_model',
            'alert_system_status': 'active',
            'dashboard_status': 'active' if self.dashboard.is_running else 'inactive',
            'uptime': datetime.now().isoformat()
        }
    
    def get_system_summary(self) -> Dict:
        """Get comprehensive system summary"""
        threat_summary = self.detector.get_threat_summary()
        alert_summary = self.alert_manager.get_alert_summary()
        dashboard_data = self.dashboard_api.get_metrics()
        
        return {
            'system_status': self.get_system_status(),
            'threat_summary': threat_summary,
            'alert_summary': alert_summary,
            'dashboard_metrics': dashboard_data,
            'configuration': self.config
        }
    
    def train_new_model(self, X: np.ndarray, y: np.ndarray, model_name: str = None) -> Dict:
        """Train a new model and deploy it"""
        print("🤖 Training new model...")
        
        # Train model
        model_metadata = self.model_manager.train_model(X, y, model_name)
        
        # Deploy model
        deployment_path = self.model_manager.deploy_model(model_metadata['name'])
        
        # Load new model into detector
        self.detector.load_model(deployment_path)
        
        print(f"✅ New model trained and deployed: {model_metadata['name']}")
        return model_metadata
    
    def simulate_events(self, duration: int = 60, event_rate: float = 1.0):
        """Simulate events for testing"""
        print(f"🎬 Starting event simulation for {duration} seconds...")
        
        simulator = EventSimulator(self.detector)
        simulator.start_simulation(duration, event_rate)
        
        # Let simulation run
        time.sleep(duration + 5)
        simulator.stop_simulation()
        
        print("✅ Event simulation completed")
    
    def export_system_data(self, output_path: str):
        """Export system data for analysis"""
        data = {
            'system_summary': self.get_system_summary(),
            'detection_history': self.detector.get_detection_history(),
            'alert_history': list(self.alert_manager.alert_history),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ System data exported to {output_path}")

class ITPSController:
    """Controller for managing ITPS system"""
    
    def __init__(self, itps_system: ITPSSystem):
        self.itps = itps_system
    
    def start_monitoring(self):
        """Start monitoring mode"""
        print("🛡️  Starting ITPS monitoring...")
        self.itps.start_system()
    
    def stop_monitoring(self):
        """Stop monitoring mode"""
        print("⏹️  Stopping ITPS monitoring...")
        self.itps.stop_system()
    
    def run_simulation(self, duration: int = 60):
        """Run event simulation"""
        self.itps.simulate_events(duration)
    
    def get_status(self):
        """Get system status"""
        status = self.itps.get_system_status()
        print("\n📊 ITPS System Status:")
        print(f"  Running: {status['is_running']}")
        print(f"  Events Processed: {status['event_count']}")
        print(f"  Threats Detected: {status['threat_count']}")
        print(f"  Detector Status: {status['detector_status']}")
        print(f"  Alert System: {status['alert_system_status']}")
        print(f"  Dashboard: {status['dashboard_status']}")
    
    def show_dashboard(self):
        """Show current dashboard"""
        summary = self.itps.get_system_summary()
        
        print("\n" + "=" * 60)
        print("🛡️  ITPS SYSTEM DASHBOARD")
        print("=" * 60)
        
        # System status
        status = summary['system_status']
        print(f"System Status: {'🟢 RUNNING' if status['is_running'] else '🔴 STOPPED'}")
        print(f"Events Processed: {status['event_count']}")
        print(f"Threats Detected: {status['threat_count']}")
        
        # Threat summary
        threat_summary = summary['threat_summary']
        print(f"\nThreat Detection:")
        print(f"  Total Detections: {threat_summary.get('total_detections', 0)}")
        print(f"  Threats Detected: {threat_summary.get('threats_detected', 0)}")
        print(f"  Threat Rate: {threat_summary.get('threat_rate', 0):.2%}")
        
        # Alert summary
        alert_summary = summary['alert_summary']
        print(f"\nAlert System:")
        print(f"  Total Alerts: {alert_summary.get('total_alerts', 0)}")
        print(f"  Active Alerts: {alert_summary.get('active_alerts', 0)}")
        print(f"  Resolved Alerts: {alert_summary.get('resolved_alerts', 0)}")
        
        print("=" * 60)

# Example usage and testing
if __name__ == "__main__":
    # Initialize ITPS system
    print("🚀 Initializing ITPS System...")
    itps = ITPSSystem()
    
    # Initialize system
    itps.initialize_system()
    
    # Create controller
    controller = ITPSController(itps)
    
    # Start monitoring
    controller.start_monitoring()
    
    # Run simulation
    print("\n🎬 Running event simulation...")
    controller.run_simulation(duration=30)
    
    # Show status
    controller.get_status()
    
    # Show dashboard
    controller.show_dashboard()
    
    # Stop system
    controller.stop_monitoring()
    
    print("\n✅ ITPS System demonstration completed!")
    print("\nTo use the system:")
    print("1. Initialize: itps = ITPSSystem()")
    print("2. Start: itps.start_system()")
    print("3. Process events: itps.process_event(user_id, event)")
    print("4. Monitor: itps.get_system_summary()")
