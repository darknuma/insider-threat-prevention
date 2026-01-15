"""
ITPS Monitoring Dashboard
Real-time dashboard for monitoring insider threats
"""

import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque, defaultdict
import numpy as np
import pandas as pd

class ITPSDashboard:
    """Real-time monitoring dashboard for ITPS"""
    
    def __init__(self, update_interval: int = 5):
        self.update_interval = update_interval
        self.is_running = False
        self.dashboard_thread = None
        
        # Data storage
        self.metrics_history = deque(maxlen=100)
        self.user_activities = defaultdict(list)
        self.threat_timeline = deque(maxlen=50)
        self.system_stats = {
            'total_users': 0,
            'active_users': 0,
            'total_events': 0,
            'threats_detected': 0,
            'alerts_generated': 0,
            'model_accuracy': 0.0
        }
        
        # Dashboard components
        self.fig = None
        self.axes = None
        self.animation = None
    
    def start_dashboard(self):
        """Start the dashboard"""
        self.is_running = True
        self.dashboard_thread = threading.Thread(target=self._run_dashboard)
        self.dashboard_thread.start()
        print("📊 ITPS Dashboard started")
    
    def stop_dashboard(self):
        """Stop the dashboard"""
        self.is_running = False
        if self.dashboard_thread:
            self.dashboard_thread.join()
        print("📊 ITPS Dashboard stopped")
    
    def _run_dashboard(self):
        """Run the dashboard loop"""
        while self.is_running:
            self._update_metrics()
            self._display_console_dashboard()
            time.sleep(self.update_interval)
    
    def _update_metrics(self):
        """Update dashboard metrics"""
        current_time = datetime.now()
        
        # Simulate some metrics (in real implementation, these would come from your system)
        metrics = {
            'timestamp': current_time.isoformat(),
            'active_users': np.random.randint(50, 200),
            'events_per_minute': np.random.randint(100, 500),
            'threats_detected': np.random.randint(0, 5),
            'alerts_active': np.random.randint(0, 10),
            'model_confidence': np.random.uniform(0.8, 0.99),
            'system_load': np.random.uniform(0.3, 0.8)
        }
        
        self.metrics_history.append(metrics)
    
    def _display_console_dashboard(self):
        """Display console-based dashboard"""
        # Clear screen (works on most terminals)
        print("\033[2J\033[H", end="")
        
        print("=" * 80)
        print("🛡️  ITPS (Insider Threat Prevention System) - Real-Time Dashboard")
        print("=" * 80)
        print(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # System Overview
        if self.metrics_history:
            latest = self.metrics_history[-1]
            print("📊 SYSTEM OVERVIEW")
            print("-" * 40)
            print(f"Active Users:        {latest['active_users']:>8}")
            print(f"Events/Minute:       {latest['events_per_minute']:>8}")
            print(f"Threats Detected:    {latest['threats_detected']:>8}")
            print(f"Active Alerts:       {latest['alerts_active']:>8}")
            print(f"Model Confidence:    {latest['model_confidence']:>8.2f}")
            print(f"System Load:         {latest['system_load']:>8.1%}")
            print()
        
        # Threat Timeline
        print("🚨 THREAT TIMELINE (Last 10 Events)")
        print("-" * 40)
        if self.threat_timeline:
            for i, threat in enumerate(list(self.threat_timeline)[-10:]):
                timestamp = threat.get('timestamp', 'Unknown')
                user_id = threat.get('user_id', 'Unknown')
                level = threat.get('threat_level', 'Unknown')
                print(f"{i+1:2d}. {timestamp} | {user_id} | {level}")
        else:
            print("No recent threats detected")
        print()
        
        # User Activity Summary
        print("👥 USER ACTIVITY SUMMARY")
        print("-" * 40)
        if self.user_activities:
            for user_id, activities in list(self.user_activities.items())[:5]:
                print(f"{user_id}: {len(activities)} activities")
        else:
            print("No user activity data")
        print()
        
        # System Status
        print("⚙️  SYSTEM STATUS")
        print("-" * 40)
        status_icons = {
            'detector': "🟢" if self.is_running else "🔴",
            'model': "🟢",
            'alerts': "🟢",
            'database': "🟢"
        }
        
        for component, icon in status_icons.items():
            print(f"{component.capitalize()}: {icon}")
        print()
        
        print("=" * 80)
        print("Press Ctrl+C to stop the dashboard")
    
    def add_threat_event(self, user_id: str, threat_level: str, confidence: float):
        """Add a threat event to the timeline"""
        threat_event = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'threat_level': threat_level,
            'confidence': confidence
        }
        self.threat_timeline.append(threat_event)
    
    def add_user_activity(self, user_id: str, activity: str):
        """Add user activity"""
        self.user_activities[user_id].append({
            'timestamp': datetime.now().isoformat(),
            'activity': activity
        })
        # Keep only last 100 activities per user
        if len(self.user_activities[user_id]) > 100:
            self.user_activities[user_id] = self.user_activities[user_id][-100:]
    
    def update_system_stats(self, stats: Dict):
        """Update system statistics"""
        self.system_stats.update(stats)
    
    def get_dashboard_data(self) -> Dict:
        """Get current dashboard data"""
        return {
            'metrics_history': list(self.metrics_history),
            'threat_timeline': list(self.threat_timeline),
            'user_activities': dict(self.user_activities),
            'system_stats': self.system_stats
        }

class WebDashboard:
    """Web-based dashboard using Flask (optional)"""
    
    def __init__(self, port: int = 5000):
        self.port = port
        self.app = None
        self.setup_flask_app()
    
    def setup_flask_app(self):
        """Setup Flask web application"""
        try:
            from flask import Flask, render_template, jsonify
            from flask_socketio import SocketIO, emit
            
            self.app = Flask(__name__)
            self.socketio = SocketIO(self.app)
            
            @self.app.route('/')
            def index():
                return render_template('dashboard.html')
            
            @self.app.route('/api/metrics')
            def get_metrics():
                # Return current metrics as JSON
                return jsonify({
                    'timestamp': datetime.now().isoformat(),
                    'active_users': 150,
                    'threats_detected': 3,
                    'alerts_active': 5
                })
            
            @self.socketio.on('connect')
            def handle_connect():
                print('Client connected to dashboard')
            
            @self.socketio.on('disconnect')
            def handle_disconnect():
                print('Client disconnected from dashboard')
            
        except ImportError:
            print("Flask not available. Install with: pip install flask flask-socketio")
            self.app = None
    
    def start_web_dashboard(self):
        """Start web dashboard"""
        if self.app:
            print(f"🌐 Starting web dashboard on http://localhost:{self.port}")
            self.socketio.run(self.app, host='0.0.0.0', port=self.port, debug=False)
        else:
            print("❌ Web dashboard not available (Flask not installed)")

class DashboardAPI:
    """API for dashboard data access"""
    
    def __init__(self, dashboard: ITPSDashboard):
        self.dashboard = dashboard
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        if self.dashboard.metrics_history:
            return self.dashboard.metrics_history[-1]
        return {}
    
    def get_threat_summary(self) -> Dict:
        """Get threat summary"""
        threats = list(self.dashboard.threat_timeline)
        if not threats:
            return {'total_threats': 0, 'recent_threats': []}
        
        threat_levels = defaultdict(int)
        for threat in threats:
            threat_levels[threat.get('threat_level', 'UNKNOWN')] += 1
        
        return {
            'total_threats': len(threats),
            'threat_levels': dict(threat_levels),
            'recent_threats': threats[-10:]
        }
    
    def get_user_summary(self) -> Dict:
        """Get user activity summary"""
        return {
            'total_users': len(self.dashboard.user_activities),
            'active_users': len([u for u in self.dashboard.user_activities.values() if u]),
            'user_activities': dict(self.dashboard.user_activities)
        }

# Example usage
if __name__ == "__main__":
    # Create dashboard
    dashboard = ITPSDashboard(update_interval=3)
    
    # Start dashboard
    dashboard.start_dashboard()
    
    # Simulate some events
    print("Simulating threat events...")
    time.sleep(2)
    dashboard.add_threat_event("USER001", "HIGH", 0.85)
    time.sleep(2)
    dashboard.add_threat_event("USER002", "CRITICAL", 0.95)
    time.sleep(2)
    dashboard.add_user_activity("USER003", "File access")
    time.sleep(2)
    dashboard.add_user_activity("USER001", "Login")
    
    # Let dashboard run for a bit
    time.sleep(10)
    
    # Stop dashboard
    dashboard.stop_dashboard()
    
    print("\n✓ Dashboard system ready!")
    print("To start web dashboard, run:")
    print("  web_dashboard = WebDashboard()")
    print("  web_dashboard.start_web_dashboard()")
