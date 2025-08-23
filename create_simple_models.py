#!/usr/bin/env python3
"""
Create extremely simple and compatible models using basic algorithms
"""
import pandas as pd
import joblib
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier

def create_simple_models():
    """Create simple, version-agnostic models"""
    
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
    
    print("Training simple models...")
    
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
        
        # Try simple models in order of preference
        models_to_try = [
            ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
            ('MultinomialNB', MultinomialNB()),
            ('DummyMostFrequent', DummyClassifier(strategy='most_frequent')),
        ]
        
        best_accuracy = 0
        best_model = None
        best_model_type = None
        
        for model_name, model in models_to_try:
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
                'feature_encoders': feature_encoders,
                'classes': list(target_encoder.classes_)
            }
        else:
            print(f"  No model worked for {target}")
    
    # Save models using joblib (more version stable than pickle)
    if saved_models:
        joblib.dump(saved_models, 'trained_models.joblib')
        
        # Also save a JSON version for fallback
        json_models = {}
        for target, model_info in saved_models.items():
            json_models[target] = {
                'model_type': model_info['model_type'],
                'accuracy': float(model_info['accuracy']),
                'classes': model_info['classes'],
                'feature_names': list(X.columns)
            }
        
        with open('model_metadata.json', 'w') as f:
            json.dump(json_models, f, indent=2)
        
        print(f"\nModels saved to 'trained_models.joblib' and metadata to 'model_metadata.json'")
        print(f"Successfully trained {len(saved_models)} models")
        
        # Test loading
        try:
            test_load = joblib.load('trained_models.joblib')
            print("✓ Models can be loaded successfully with joblib!")
        except Exception as e:
            print(f"✗ Error loading with joblib: {e}")
        
        return True
    else:
        print("No models were successfully trained")
        return False

if __name__ == "__main__":
    success = create_simple_models()
    if success:
        print("\nSimple, compatible models created successfully!")
    else:
        print("Model creation failed")