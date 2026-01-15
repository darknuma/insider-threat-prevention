"""
Model Deployment System for ITPS
Handles model training, persistence, versioning, and deployment
"""

import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import shutil

class ModelManager:
    """Manages model training, versioning, and deployment"""
    
    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.current_model = None
        self.model_versions = []
        self._load_model_registry()
    
    def _load_model_registry(self):
        """Load model registry from disk"""
        registry_path = self.models_dir / "model_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                self.model_versions = json.load(f)
        else:
            self.model_versions = []
    
    def _save_model_registry(self):
        """Save model registry to disk"""
        registry_path = self.models_dir / "model_registry.json"
        with open(registry_path, 'w') as f:
            json.dump(self.model_versions, f, indent=2)
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   model_name: str = None, 
                   model_params: Dict = None) -> Dict:
        """Train a new model and save it"""
        
        if model_name is None:
            model_name = f"itps_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if model_params is None:
            model_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42,
                'class_weight': 'balanced'
            }
        
        print(f"Training model: {model_name}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = model.score(X_test_scaled, y_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Create model metadata
        model_metadata = {
            'name': model_name,
            'version': len(self.model_versions) + 1,
            'timestamp': datetime.now().isoformat(),
            'model_params': model_params,
            'accuracy': float(accuracy),
            'classification_report': report,
            'feature_count': X.shape[1],
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'class_distribution': {
                'malicious': int(np.sum(y == 0)),
                'benign': int(np.sum(y == 1))
            }
        }
        
        # Save model files
        model_path = self.models_dir / f"{model_name}.pkl"
        scaler_path = self.models_dir / f"{model_name}_scaler.pkl"
        
        # Save model and scaler
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save combined model data
        combined_data = {
            'model': model,
            'scaler': scaler,
            'metadata': model_metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        combined_path = self.models_dir / f"{model_name}_combined.pkl"
        with open(combined_path, 'wb') as f:
            pickle.dump(combined_data, f)
        
        # Update registry
        self.model_versions.append(model_metadata)
        self._save_model_registry()
        
        # Set as current model
        self.current_model = model_name
        
        print(f"✓ Model trained and saved: {model_name}")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Training samples: {len(X_train)}")
        
        return model_metadata
    
    def load_model(self, model_name: str) -> Tuple[object, StandardScaler, Dict]:
        """Load a specific model"""
        model_path = self.models_dir / f"{model_name}_combined.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.current_model = model_name
        return data['model'], data['scaler'], data['metadata']
    
    def get_model_list(self) -> List[Dict]:
        """Get list of all available models"""
        return self.model_versions.copy()
    
    def get_best_model(self) -> str:
        """Get the best performing model"""
        if not self.model_versions:
            return None
        
        best_model = max(self.model_versions, key=lambda x: x['accuracy'])
        return best_model['name']
    
    def deploy_model(self, model_name: str, deployment_name: str = None) -> str:
        """Deploy a model for production use"""
        if deployment_name is None:
            deployment_name = f"production_{model_name}"
        
        # Create deployment directory
        deployment_dir = self.models_dir / "deployments" / deployment_name
        deployment_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        model_path = self.models_dir / f"{model_name}_combined.pkl"
        deployment_path = deployment_dir / "model.pkl"
        
        shutil.copy2(model_path, deployment_path)
        
        # Create deployment metadata
        deployment_metadata = {
            'deployment_name': deployment_name,
            'model_name': model_name,
            'deployed_at': datetime.now().isoformat(),
            'status': 'active'
        }
        
        with open(deployment_dir / "deployment_info.json", 'w') as f:
            json.dump(deployment_metadata, f, indent=2)
        
        print(f"✓ Model deployed: {deployment_name}")
        return str(deployment_path)
    
    def get_deployment_status(self) -> Dict:
        """Get current deployment status"""
        deployments_dir = self.models_dir / "deployments"
        if not deployments_dir.exists():
            return {'active_deployments': 0, 'deployments': []}
        
        deployments = []
        for deployment_dir in deployments_dir.iterdir():
            if deployment_dir.is_dir():
                info_path = deployment_dir / "deployment_info.json"
                if info_path.exists():
                    with open(info_path, 'r') as f:
                        deployments.append(json.load(f))
        
        return {
            'active_deployments': len(deployments),
            'deployments': deployments
        }

class ModelEvaluator:
    """Evaluates model performance on new data"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
    
    def evaluate_model(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate a model on test data"""
        try:
            model, scaler, metadata = self.model_manager.load_model(model_name)
            
            # Scale test data
            X_test_scaled = scaler.transform(X_test)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = model.score(X_test_scaled, y_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Feature importance
            feature_importance = model.feature_importances_
            top_features = np.argsort(feature_importance)[-10:][::-1]
            
            evaluation_results = {
                'model_name': model_name,
                'evaluation_timestamp': datetime.now().isoformat(),
                'accuracy': float(accuracy),
                'classification_report': report,
                'top_features': top_features.tolist(),
                'feature_importance': feature_importance.tolist(),
                'test_samples': len(X_test),
                'class_distribution': {
                    'malicious': int(np.sum(y_test == 0)),
                    'benign': int(np.sum(y_test == 1))
                }
            }
            
            return evaluation_results
            
        except Exception as e:
            return {
                'model_name': model_name,
                'error': str(e),
                'evaluation_timestamp': datetime.now().isoformat()
            }

# Example usage
if __name__ == "__main__":
    # Initialize model manager
    model_manager = ModelManager()
    
    # Load training data (you would load your actual data here)
    print("Loading training data...")
    # This would be replaced with your actual data loading
    # X, y = load_your_training_data()
    
    # For demonstration, create dummy data
    np.random.seed(42)
    X = np.random.randn(1000, 640)  # 1000 samples, 640 features
    y = np.random.randint(0, 2, 1000)  # Binary labels
    
    # Train model
    model_metadata = model_manager.train_model(X, y)
    
    # Deploy model
    deployment_path = model_manager.deploy_model(model_metadata['name'])
    
    # Get deployment status
    status = model_manager.get_deployment_status()
    print(f"\nDeployment Status:")
    print(f"  Active deployments: {status['active_deployments']}")
    
    # List all models
    models = model_manager.get_model_list()
    print(f"\nAvailable Models:")
    for model in models:
        print(f"  {model['name']} - Accuracy: {model['accuracy']:.3f}")
    
    print("\n✓ Model deployment system ready!")
