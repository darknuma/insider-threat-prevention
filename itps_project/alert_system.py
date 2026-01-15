"""
Alert System for ITPS
Handles threat alerts, notifications, and escalation
"""

import json
import smtplib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass, asdict
import threading
import time
from collections import defaultdict, deque

@dataclass
class Alert:
    """Alert data structure"""
    alert_id: str
    user_id: str
    threat_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    confidence: float
    timestamp: str
    description: str
    details: Dict
    status: str = 'ACTIVE'  # 'ACTIVE', 'ACKNOWLEDGED', 'RESOLVED'
    escalated: bool = False

class AlertManager:
    """Manages threat alerts and notifications"""
    
    def __init__(self, config_path: str = "./config/alert_config.json"):
        self.config_path = Path(config_path)
        self.config_path.parent.mkdir(exist_ok=True)
        self.alerts = []
        self.alert_history = deque(maxlen=1000)
        self.user_alert_counts = defaultdict(int)
        self.alert_rules = []
        self.notification_handlers = []
        self._load_config()
    
    def _load_config(self):
        """Load alert configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default configuration
            config = {
                "alert_thresholds": {
                    "LOW": 0.6,
                    "MEDIUM": 0.75,
                    "HIGH": 0.85,
                    "CRITICAL": 0.95
                },
                "escalation_rules": {
                    "max_alerts_per_user": 5,
                    "escalation_time_minutes": 30
                },
                "notifications": {
                    "email": {
                        "enabled": False,
                        "smtp_server": "smtp.gmail.com",
                        "smtp_port": 587,
                        "username": "",
                        "password": "",
                        "recipients": []
                    },
                    "webhook": {
                        "enabled": False,
                        "url": "",
                        "headers": {}
                    },
                    "console": {
                        "enabled": True
                    }
                }
            }
            self._save_config(config)
        
        self.config = config
    
    def _save_config(self, config: Dict):
        """Save alert configuration"""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def create_alert(self, user_id: str, confidence: float, 
                    description: str, details: Dict) -> Alert:
        """Create a new alert"""
        
        # Determine threat level
        threat_level = self._determine_threat_level(confidence)
        
        # Generate alert ID
        alert_id = f"ALERT_{int(datetime.now().timestamp() * 1000)}"
        
        # Create alert
        alert = Alert(
            alert_id=alert_id,
            user_id=user_id,
            threat_level=threat_level,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            description=description,
            details=details
        )
        
        # Add to alerts list
        self.alerts.append(alert)
        self.alert_history.append(alert)
        self.user_alert_counts[user_id] += 1
        
        # Check for escalation
        self._check_escalation(user_id)
        
        # Send notifications
        self._send_notifications(alert)
        
        return alert
    
    def _determine_threat_level(self, confidence: float) -> str:
        """Determine threat level based on confidence"""
        thresholds = self.config['alert_thresholds']
        
        if confidence >= thresholds['CRITICAL']:
            return 'CRITICAL'
        elif confidence >= thresholds['HIGH']:
            return 'HIGH'
        elif confidence >= thresholds['MEDIUM']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _check_escalation(self, user_id: str):
        """Check if user should be escalated"""
        max_alerts = self.config['escalation_rules']['max_alerts_per_user']
        
        if self.user_alert_counts[user_id] >= max_alerts:
            # Create escalation alert
            escalation_alert = Alert(
                alert_id=f"ESCALATION_{int(datetime.now().timestamp() * 1000)}",
                user_id=user_id,
                threat_level='CRITICAL',
                confidence=1.0,
                timestamp=datetime.now().isoformat(),
                description=f"User {user_id} has triggered {self.user_alert_counts[user_id]} alerts - ESCALATION REQUIRED",
                details={'escalation_reason': 'max_alerts_exceeded'},
                escalated=True
            )
            
            self.alerts.append(escalation_alert)
            self.alert_history.append(escalation_alert)
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert"""
        notifications = self.config['notifications']
        
        # Console notification
        if notifications['console']['enabled']:
            self._console_notification(alert)
        
        # Email notification
        if notifications['email']['enabled']:
            self._email_notification(alert)
        
        # Webhook notification
        if notifications['webhook']['enabled']:
            self._webhook_notification(alert)
    
    def _console_notification(self, alert: Alert):
        """Send console notification"""
        print(f"\n🚨 ALERT: {alert.threat_level} THREAT DETECTED")
        print(f"   User: {alert.user_id}")
        print(f"   Confidence: {alert.confidence:.2f}")
        print(f"   Time: {alert.timestamp}")
        print(f"   Description: {alert.description}")
        print(f"   Alert ID: {alert.alert_id}")
        print("-" * 50)
    
    def _email_notification(self, alert: Alert):
        """Send email notification"""
        try:
            email_config = self.config['notifications']['email']
            
            msg = MIMEMultipart()
            msg['From'] = email_config['username']
            msg['To'] = ', '.join(email_config['recipients'])
            msg['Subject'] = f"ITPS Alert: {alert.threat_level} Threat Detected"
            
            body = f"""
            THREAT ALERT DETECTED
            
            Alert ID: {alert.alert_id}
            User ID: {alert.user_id}
            Threat Level: {alert.threat_level}
            Confidence: {alert.confidence:.2f}
            Timestamp: {alert.timestamp}
            Description: {alert.description}
            
            Details: {json.dumps(alert.details, indent=2)}
            
            Please investigate immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
            server.starttls()
            server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
            server.quit()
            
            print(f"✓ Email notification sent for alert {alert.alert_id}")
            
        except Exception as e:
            print(f"❌ Email notification failed: {e}")
    
    def _webhook_notification(self, alert: Alert):
        """Send webhook notification"""
        try:
            webhook_config = self.config['notifications']['webhook']
            
            payload = {
                'alert_id': alert.alert_id,
                'user_id': alert.user_id,
                'threat_level': alert.threat_level,
                'confidence': alert.confidence,
                'timestamp': alert.timestamp,
                'description': alert.description,
                'details': alert.details
            }
            
            response = requests.post(
                webhook_config['url'],
                json=payload,
                headers=webhook_config.get('headers', {}),
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"✓ Webhook notification sent for alert {alert.alert_id}")
            else:
                print(f"❌ Webhook notification failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Webhook notification failed: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = 'ACKNOWLEDGED'
                alert.details['acknowledged_by'] = acknowledged_by
                alert.details['acknowledged_at'] = datetime.now().isoformat()
                return True
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str, resolution_notes: str = "") -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.status = 'RESOLVED'
                alert.details['resolved_by'] = resolved_by
                alert.details['resolved_at'] = datetime.now().isoformat()
                alert.details['resolution_notes'] = resolution_notes
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [alert for alert in self.alerts if alert.status == 'ACTIVE']
    
    def get_user_alerts(self, user_id: str) -> List[Alert]:
        """Get alerts for a specific user"""
        return [alert for alert in self.alerts if alert.user_id == user_id]
    
    def get_alert_summary(self) -> Dict:
        """Get alert summary statistics"""
        total_alerts = len(self.alerts)
        active_alerts = len(self.get_active_alerts())
        
        threat_levels = defaultdict(int)
        for alert in self.alerts:
            threat_levels[alert.threat_level] += 1
        
        return {
            'total_alerts': total_alerts,
            'active_alerts': active_alerts,
            'resolved_alerts': total_alerts - active_alerts,
            'threat_level_distribution': dict(threat_levels),
            'users_with_alerts': len(set(alert.user_id for alert in self.alerts)),
            'escalated_alerts': sum(1 for alert in self.alerts if alert.escalated)
        }
    
    def add_notification_handler(self, handler: Callable):
        """Add custom notification handler"""
        self.notification_handlers.append(handler)
    
    def update_config(self, new_config: Dict):
        """Update alert configuration"""
        self.config.update(new_config)
        self._save_config(self.config)

class AlertDashboard:
    """Simple alert dashboard for monitoring"""
    
    def __init__(self, alert_manager: AlertManager):
        self.alert_manager = alert_manager
    
    def display_dashboard(self):
        """Display alert dashboard"""
        print("\n" + "=" * 60)
        print("ITPS ALERT DASHBOARD")
        print("=" * 60)
        
        # Alert summary
        summary = self.alert_manager.get_alert_summary()
        print(f"Total Alerts: {summary['total_alerts']}")
        print(f"Active Alerts: {summary['active_alerts']}")
        print(f"Resolved Alerts: {summary['resolved_alerts']}")
        print(f"Users with Alerts: {summary['users_with_alerts']}")
        print(f"Escalated Alerts: {summary['escalated_alerts']}")
        
        # Threat level distribution
        print(f"\nThreat Level Distribution:")
        for level, count in summary['threat_level_distribution'].items():
            print(f"  {level}: {count}")
        
        # Recent alerts
        recent_alerts = list(self.alert_manager.alert_history)[-5:]
        print(f"\nRecent Alerts:")
        for alert in recent_alerts:
            status_icon = "🔴" if alert.status == 'ACTIVE' else "🟡" if alert.status == 'ACKNOWLEDGED' else "🟢"
            print(f"  {status_icon} {alert.threat_level} - {alert.user_id} - {alert.timestamp}")
        
        print("=" * 60)

# Example usage
if __name__ == "__main__":
    # Initialize alert manager
    alert_manager = AlertManager()
    
    # Create some test alerts
    test_alerts = [
        ("USER001", 0.85, "Suspicious file access pattern", {"files_accessed": 50, "time_window": "5min"}),
        ("USER002", 0.92, "Unusual login time and location", {"login_time": "03:00", "location": "unknown"}),
        ("USER003", 0.78, "High volume data transfer", {"data_size": "2GB", "destination": "external"})
    ]
    
    print("Creating test alerts...")
    for user_id, confidence, description, details in test_alerts:
        alert = alert_manager.create_alert(user_id, confidence, description, details)
        print(f"Created alert: {alert.alert_id}")
    
    # Display dashboard
    dashboard = AlertDashboard(alert_manager)
    dashboard.display_dashboard()
    
    # Test alert resolution
    alerts = alert_manager.get_active_alerts()
    if alerts:
        alert_manager.resolve_alert(alerts[0].alert_id, "admin", "False positive - normal behavior")
        print(f"\nResolved alert: {alerts[0].alert_id}")
    
    # Display updated dashboard
    dashboard.display_dashboard()
    
    print("\n✓ Alert system ready!")
