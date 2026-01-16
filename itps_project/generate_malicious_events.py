import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Optional
import uuid
from advanced_realtime_detector import AdvancedRealTimeDetector

class MaliciousEventGenerator:
    """Generates synthetic malicious events for testing the ITPS system."""

    def __init__(self, start_date: datetime = None):
        self.current_date = start_date if start_date else datetime.now()
        self.base_date = self.current_date

    def _get_random_id(self, prefix: str) -> str:
        return f"{prefix}_{str(uuid.uuid4())[:8]}"

    def generate_benign_sequence(self, user_id: str, pc_id: str, length: int = 10) -> List[Dict]:
        """Generate a sequence of normal, benign background noise."""
        events = []
        for _ in range(length):
            # Advance time slightly
            self.current_date += timedelta(minutes=random.randint(5, 60))
            
            # Ensure we are in work hours (8 AM - 6 PM) for benign
            if self.current_date.hour < 8:
                self.current_date = self.current_date.replace(hour=8, minute=0)
            elif self.current_date.hour > 18:
                self.current_date += timedelta(days=1)
                self.current_date = self.current_date.replace(hour=8, minute=0)
                
            # Random benign event type
            event_type = random.choice(['file', 'email', 'http'])
            
            event = {
                'id': self._get_random_id('EVT'),
                'user': user_id,
                'date': self.current_date.isoformat(),
                'pc': pc_id,
                'event_type': event_type
            }

            if event_type == 'file':
                event.update({
                    'filename': f"project_docs_{random.randint(1,100)}.txt",
                    'content': 'normal work content'
                })
            elif event_type == 'email':
                event.update({
                    'to': 'colleague@company.com',
                    'cc': '',
                    'bcc': '',
                    'from': user_id,
                    'size': random.randint(100, 5000),
                    'attachments': '',
                    'content': 'meeting notes'
                })
            elif event_type == 'http':
                event.update({
                    'url': 'http://intranet.company.com',
                    'content': 'work portal'
                })
            
            events.append(event)
        return events

    def generate_insider_threat_sequence(self, user_id: str, pc_id: str) -> List[Dict]:
        """
        Generate a cohesive and INTENSE sequence of malicious events.
        """
        events = []
        
        # 1. Late night login (Time anomaly)
        self.current_date += timedelta(days=1)
        self.current_date = self.current_date.replace(hour=3, minute=0) # 3 AM
        
        login_event = {
            'id': self._get_random_id('LOGON'),
            'user': user_id,
            'date': self.current_date.isoformat(),
            'pc': pc_id,
            'event_type': 'logon',
            'activity': 'Logon'
        }
        events.append(login_event)

        # 2. Burst of sensitive file accesses
        sensitive_files = [
            'confidential_passwords.xls', 'secret_plans.pdf', 'employee_salaries.doc',
            'network_diagram.vsd', 'customer_list.xls', 'source_code.zip',
            'budget_2025.xlsx', 'executive_meeting.ppt'
        ]
        
        for fname in sensitive_files:
            self.current_date += timedelta(seconds=random.randint(10, 30))
            file_event = {
                'id': self._get_random_id('FILE'),
                'user': user_id,
                'date': self.current_date.isoformat(),
                'pc': pc_id,
                'event_type': 'file',
                'filename': fname,
                'content': 'SENSITIVE CONFIDENTIAL DATA'
            }
            events.append(file_event)

        # 3. Connect removable device
        self.current_date += timedelta(minutes=2)
        device_event = {
            'id': self._get_random_id('DEV'),
            'user': user_id,
            'date': self.current_date.isoformat(),
            'pc': pc_id,
            'event_type': 'device',
            'activity': 'Connect',
            'file_tree': ' Removable Disk'
        }
        events.append(device_event)

        # 4. Multiple data exfiltrations via email
        for _ in range(3):
            self.current_date += timedelta(minutes=5)
            email_event = {
                'id': self._get_random_id('EMAIL'),
                'user': user_id,
                'date': self.current_date.isoformat(),
                'pc': pc_id,
                'event_type': 'email',
                'to': 'competitor@gmail.com, personal@yahoo.com',
                'cc': 'accomplice@protonmail.com',
                'bcc': '',
                'from': user_id,
                'size': 1024 * 1024 * 10, # 10MB
                'attachments': 'secret_plans.pdf;passwords.xls;dump.zip',
                'content': 'Exfiltrating batch data now.'
            }
            events.append(email_event)
        
        # 5. Suspicious HTTP activity to known bad domains
        bad_urls = ['http://www.blackmarket.com', 'http://192.168.1.100/upload', 'http://hacker-forum.net']
        for url in bad_urls:
            self.current_date += timedelta(minutes=2)
            http_event = {
                'id': self._get_random_id('HTTP'),
                'user': user_id,
                'date': self.current_date.isoformat(),
                'pc': pc_id,
                'event_type': 'http',
                'url': url,
                'content': 'uploading stolen data'
            }
            events.append(http_event)

        return events


def generate_malicious_sequence(length: int = 25) -> List[Dict]:
    """
    Wrapper to generate a malicious sequence using MaliciousEventGenerator.
    """
    generator = MaliciousEventGenerator()
    user_id = "ATTACKER_01"
    pc_id = "PC-666"
    
    # 1. Start with fewer benign events so the window fills with malicious ones faster
    events = generator.generate_benign_sequence(user_id, pc_id, length=5)
    
    # 2. Add the malicious payload (now ~14 events)
    malicious_events = generator.generate_insider_threat_sequence(user_id, pc_id)
    events.extend(malicious_events)
    
    # 3. Fill rest with random
    remaining = length - len(events)
    if remaining > 0:
        events.extend(generator.generate_benign_sequence(user_id, pc_id, length=remaining))
    
    return events[:length]

if __name__ == "__main__":
    try:
        detector = AdvancedRealTimeDetector(models_dir="./models")
        
        print("Generating intense malicious sequence...")
        malicious_events = generate_malicious_sequence(length=30) # Increase length to 30 to allow full window

        print("\n--- SCENARIO B OUTPUT ---")
        
        first_detection = True
        max_confidence = -1.0
        best_result = None

        for event in malicious_events:
            result = detector.add_event(event['user'], event)
            
            if result:
                conf = result['confidence']
                if conf > max_confidence:
                    max_confidence = conf
                    best_result = result

                # Stop and print as soon as we detect a threat
                if result['threat_detected']:
                    if first_detection:
                        print(f"Threat Detected: {result['threat_detected']}")
                        print(f"Threat Level: {result['threat_level']}")
                        print(f"Confidence: {result['confidence']:.3f}")
                        print(f"Insights: {result['key_insights']}")
                        first_detection = False
                        break 
        
        if first_detection: 
             # If no "threat_detected" flag was raised, print the highest confidence one anyway
             if best_result:
                 print(f"Threat Detected: {best_result['threat_detected']} (Threshold not met)")
                 print(f"Threat Level: {best_result['threat_level']}")
                 print(f"Confidence: {best_result['confidence']:.3f}")
                 print(f"Insights: {best_result['key_insights']}")
             else:
                 print("No detection results generated (Sequence length might be insufficient or errors occurred).")

    except Exception as e:
        print(f"Error running Scenario B: {e}")


