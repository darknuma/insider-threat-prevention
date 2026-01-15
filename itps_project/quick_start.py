"""
ITPS Quick Start Script
Easy way to get the complete ITPS system running
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from itps_system import ITPSSystem, ITPSController

def quick_start_demo():
    """Run a quick demonstration of the ITPS system"""
    print("🚀 ITPS Quick Start Demo")
    print("=" * 50)
    
    # Initialize ITPS system
    print("1. Initializing ITPS System...")
    itps = ITPSSystem()
    itps.initialize_system()
    
    # Create controller
    controller = ITPSController(itps)
    
    # Start monitoring
    print("2. Starting monitoring...")
    controller.start_monitoring()
    
    # Run simulation
    print("3. Running event simulation...")
    controller.run_simulation(duration=30)
    
    # Show results
    print("4. System Results:")
    controller.show_dashboard()
    
    # Stop system
    print("5. Stopping system...")
    controller.stop_monitoring()
    
    print("\n✅ ITPS Quick Start Demo completed!")
    print("\nYour ITPS system is ready to use!")

def train_and_deploy_model():
    """Train a model and deploy it for real-time detection"""
    print("🤖 Training and Deploying Model")
    print("=" * 50)
    
    # Initialize ITPS system
    itps = ITPSSystem()
    
    # Load your training data (replace with your actual data loading)
    print("Loading training data...")
    # This would be your actual data loading code
    # X, y = load_your_training_data()
    
    # For demo, create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 640)  # 1000 samples, 640 features
    y = np.random.randint(0, 2, 1000)  # Binary labels
    
    # Train model
    print("Training model...")
    model_metadata = itps.train_new_model(X, y, "itps_production_model")
    
    print(f"✅ Model trained: {model_metadata['name']}")
    print(f"   Accuracy: {model_metadata['accuracy']:.3f}")
    print(f"   Features: {model_metadata['feature_count']}")
    
    # Initialize system with new model
    itps.initialize_system()
    
    # Start monitoring
    controller = ITPSController(itps)
    controller.start_monitoring()
    
    print("✅ Model deployed and system ready!")
    return itps, controller

def run_production_system():
    """Run the production ITPS system"""
    print("🛡️  Starting Production ITPS System")
    print("=" * 50)
    
    # Initialize system
    itps = ITPSSystem()
    itps.initialize_system()
    
    # Start monitoring
    controller = ITPSController(itps)
    controller.start_monitoring()
    
    print("✅ Production ITPS System is running!")
    print("📊 Monitor the dashboard for real-time threat detection")
    print("🔄 System will continue running until stopped")
    
    try:
        # Keep system running
        while True:
            time.sleep(10)
            # You could add periodic status updates here
            pass
    except KeyboardInterrupt:
        print("\n⏹️  Stopping production system...")
        controller.stop_monitoring()
        print("✅ Production system stopped")

def main():
    """Main function with menu options"""
    print("🛡️  ITPS (Insider Threat Prevention System)")
    print("=" * 60)
    print("Choose an option:")
    print("1. Quick Demo - See the system in action")
    print("2. Train Model - Train and deploy a new model")
    print("3. Production Mode - Run the full system")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                quick_start_demo()
                break
            elif choice == '2':
                train_and_deploy_model()
                break
            elif choice == '3':
                run_production_system()
                break
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-4.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
