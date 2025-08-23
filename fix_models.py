#!/usr/bin/env python3
"""
Fix model compatibility issues by retraining with minimal dependencies
"""
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def retrain_lightweight_models():
    """Retrain models using only sklearn built-in models without complex pipelines"""
    
    print("Loading data...")
    try:
        dataset = pd.read_csv('dataset.csv')
        print(f"Dataset loaded: {dataset.shape}")
    except FileNotFoundError:
        print("Error: dataset.csv not found")
        return False
    
    # Features and targets
    X = dataset.iloc[:, :5].astype(str)
    targets = ['Type of Sanitiser','No. of Vents','Vents Sizes', 'Flut Size', 
               'Packaging Leaking', 'Package Satisfaction', 'Sanitiser Satisfaction']
    
    # Simple encoding for features
    X_encoded = pd.DataFrame()
    feature_encoders = {}
    
    for col in X.columns:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        feature_encoders[col] = le
    
    print("Training lightweight models...")
    
    # Simple models without pipelines or complex parameter tuning
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVC': SVC(kernel='rbf', random_state=42),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    saved_models = {}
    
    for target in targets:
        print(f"Training model for: {target}")
        
        # Get target values
        y = dataset[target].fillna(method='ffill')
        
        # Encode target
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42
        )
        
        best_accuracy = 0
        best_model = None
        best_model_type = None
        
        # Try each model type
        for model_name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                accuracy = model.score(X_test, y_test)
                print(f"  {model_name}: {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_model_type = model_name
                    
            except Exception as e:
                print(f"  {model_name}: Error - {e}")
                continue
        
        if best_model is not None:
            print(f"  Best: {best_model_type} ({best_accuracy:.4f})")
            
            # Store the best model
            saved_models[target] = {
                'model': best_model,
                'encoder': target_encoder,
                'accuracy': best_accuracy,
                'model_type': best_model_type,
                'feature_encoders': feature_encoders  # Store feature encoders too
            }
        else:
            print(f"  No model worked for {target}")
    
    # Save models
    if saved_models:
        with open('trained_models_fixed.pkl', 'wb') as f:
            pickle.dump(saved_models, f)
        
        print(f"\nFixed models saved to 'trained_models_fixed.pkl'")
        print(f"Successfully trained {len(saved_models)} models")
        
        # Test loading
        with open('trained_models_fixed.pkl', 'rb') as f:
            test_load = pickle.load(f)
        print("Models can be loaded successfully!")
        
        return True
    else:
        print("No models were successfully trained")
        return False

if __name__ == "__main__":
    success = retrain_lightweight_models()
    if success:
        print("\nTo use the fixed models, rename 'trained_models_fixed.pkl' to 'trained_models.pkl'")
    else:
        print("Model fixing failed")