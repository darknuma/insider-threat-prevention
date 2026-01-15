"""
Run Advanced Models - Complete Pipeline
Trains and runs advanced ML models with your CERT data
"""

import sys
from pathlib import Path

def main():
    """Main function to run advanced models"""
    print("🚀 ADVANCED ITPS MODELS - COMPLETE PIPELINE")
    print("=" * 60)
    print("This will:")
    print("1. Train LSTM, Transformer, and Autoencoder models")
    print("2. Create ensemble model")
    print("3. Test advanced real-time detection")
    print("4. Show model performance")
    print()
    
    # Check if user_sequences.json exists
    sequences_file = Path("./datasets/user_sequences.json")
    if not sequences_file.exists():
        print("❌ user_sequences.json not found!")
        print("Please run the DuckDB generator first to create sequences.")
        return
    
    # Step 1: Train advanced models
    print("Step 1: Training Advanced Models...")
    print("-" * 40)
    
    try:
        from implement_advanced_models import main as train_models
        train_models()
        print("✅ Advanced models trained successfully!")
    except Exception as e:
        print(f"❌ Error training models: {e}")
        return
    
    # Step 2: Test advanced detection
    print("\nStep 2: Testing Advanced Detection...")
    print("-" * 40)
    
    try:
        from advanced_realtime_detector import AdvancedRealTimeDetector
        
        # Initialize detector
        detector = AdvancedRealTimeDetector()
        
        # Test with sample events
        print("Testing with sample events...")
        sample_event = {
            'event_type': 'logon',
            'id': 'TEST_001',
            'user': 'USER001',
            'date': '2024-01-01 10:00:00',
            'pc': 'PC-1234',
            'activity': 'Logon'
        }
        
        # Add events to build sequence
        for i in range(25):
            result = detector.add_event('USER001', sample_event)
            if result:
                print(f"Advanced Detection Result:")
                print(f"  Threat Detected: {result['threat_detected']}")
                print(f"  Threat Level: {result['threat_level']}")
                print(f"  Confidence: {result['confidence']:.3f}")
                print(f"  Model Agreement: {result['model_agreement']}")
                print(f"  Key Insights: {result['key_insights']}")
                break
        
        print("✅ Advanced detection working!")
        
    except Exception as e:
        print(f"❌ Error testing detection: {e}")
        return
    
    # Step 3: Show model performance
    print("\nStep 3: Model Performance Summary...")
    print("-" * 40)
    
    try:
        import joblib
        
        models_dir = Path("./models")
        if models_dir.exists():
            print("📊 Model Performance:")
            
            # LSTM
            if (models_dir / "lstm_model.pkl").exists():
                lstm_data = joblib.load(models_dir / "lstm_model.pkl")
                print(f"  LSTM: {lstm_data['accuracy']:.3f} accuracy")
            
            # Transformer
            if (models_dir / "transformer_model.pkl").exists():
                transformer_data = joblib.load(models_dir / "transformer_model.pkl")
                print(f"  Transformer: {transformer_data['accuracy']:.3f} accuracy")
            
            # Autoencoder
            if (models_dir / "autoencoder_model.pkl").exists():
                autoencoder_data = joblib.load(models_dir / "autoencoder_model.pkl")
                print(f"  Autoencoder: Threshold = {autoencoder_data['threshold']:.4f}")
            
            # Ensemble
            if (models_dir / "ensemble_model.pkl").exists():
                ensemble_data = joblib.load(models_dir / "ensemble_model.pkl")
                print(f"  Ensemble: {ensemble_data['accuracy']:.3f} accuracy")
        
        print("✅ Model performance summary complete!")
        
    except Exception as e:
        print(f"❌ Error getting performance: {e}")
    
    # Step 4: Next steps
    print("\nStep 4: Next Steps...")
    print("-" * 40)
    print("✅ Advanced models are ready!")
    print()
    print("To use the advanced system:")
    print("1. Run: python advanced_itps_complete.py")
    print("2. Or integrate with your existing ITPS system")
    print()
    print("Advanced features available:")
    print("• LSTM for temporal pattern detection")
    print("• Transformer for attention-based analysis")
    print("• Autoencoder for anomaly detection")
    print("• Ensemble for robust predictions")
    print("• Real-time threat explanations")
    print("• Model agreement analysis")
    
    print("\n🎉 Advanced ITPS models ready for production!")

if __name__ == "__main__":
    main()
