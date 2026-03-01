"""
Load and Use Saved ML Models for Network Intrusion Detection
==============================================================

This script demonstrates how to:
1. Load the trained models from disk
2. Preprocess new network traffic data
3. Make predictions with XGBoost (best model)
4. Compare predictions from all 5 models
"""

import pickle
import pandas as pd
import numpy as np
import os

# ============================================================================
# 1. LOAD SAVED MODELS & PREPROCESSING TOOLS
# ============================================================================

def load_models():
    """Load all 5 trained models from disk"""
    models_dir = 'saved_models'
    
    models = {}
    model_names = [
        'logistic_regression.pkl',
        'naive_bayes.pkl',
        'random_forest.pkl',
        'xgboost.pkl',
        'lightgbm.pkl'
    ]
    
    print("Loading trained models...")
    for model_file in model_names:
        model_path = os.path.join(models_dir, model_file)
        with open(model_path, 'rb') as f:
            model_name = model_file.replace('.pkl', '')
            models[model_name] = pickle.load(f)
            print(f"  [OK] {model_name}")
    
    # Load StandardScaler
    with open(os.path.join(models_dir, 'feature_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    print(f"  [OK] feature_scaler")
    
    # Load feature names
    with open(os.path.join(models_dir, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    print(f"  [OK] feature_names ({len(feature_names)} features)")
    
    return models, scaler, feature_names


# ============================================================================
# 2. EXAMPLE: MAKE PREDICTIONS ON NEW DATA
# ============================================================================

def predict_attacks(new_data, models, scaler, feature_names):
    """
    Make predictions on new network traffic data
    
    Parameters:
    -----------
    new_data : pd.DataFrame or array
        New network traffic samples (shape: n_samples x 35 features)
    
    Returns:
    --------
    predictions : dict
        Dictionary with predictions from all 5 models
    """
    
    # Ensure correct feature order
    if isinstance(new_data, pd.DataFrame):
        X_new = new_data[feature_names].values
    else:
        X_new = new_data
    
    # Standardize features (CRITICAL - must match training)
    X_new_scaled = scaler.transform(X_new)
    
    # Get predictions from all models
    predictions = {}
    
    for model_name, model in models.items():
        pred_classes = model.predict(X_new_scaled)
        
        # Get probability predictions (confidence scores)
        if hasattr(model, 'predict_proba'):
            pred_probs = model.predict_proba(X_new_scaled)
            max_confidence = np.max(pred_probs, axis=1)
        else:
            max_confidence = np.ones(len(pred_classes))  # No probability for some models
        
        predictions[model_name] = {
            'predicted_class': pred_classes,
            'confidence': max_confidence
        }
    
    return predictions


# ============================================================================
# 3. EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("=" * 80)
    print("Network Intrusion Detection - Model Inference")
    print("=" * 80)
    
    # Load models
    models, scaler, feature_names = load_models()
    
    print(f"\n[OK] Loaded 5 models and preprocessing tools")
    print(f"  Features: {feature_names}")
    
    # Example 1: Load some test data from the full dataset
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Predicting on Random Test Samples")
    print("=" * 80)
    
    # Load raw data and preprocess
    df = pd.read_parquet('merged_clean.parquet')
    
    # Take 5 random samples from each attack class
    X_test_samples = df.sample(n=10, random_state=42)[feature_names]
    y_test_labels = df.sample(n=10, random_state=42)['label'].values
    
    # Make predictions
    predictions = predict_attacks(X_test_samples, models, scaler, feature_names)
    
    # Display results
    print(f"\nPredictions on {len(X_test_samples)} test samples:\n")
    
    print("Sample | True Class | XGBoost | Random Forest | Logistic Reg | Naive Bayes | LightGBM")
    print("-" * 95)
    
    for i in range(len(X_test_samples)):
        true_class = y_test_labels[i]
        xgb_pred = predictions['xgboost']['predicted_class'][i]
        rf_pred = predictions['random_forest']['predicted_class'][i]
        lr_pred = predictions['logistic_regression']['predicted_class'][i]
        nb_pred = predictions['naive_bayes']['predicted_class'][i]
        lgb_pred = predictions['lightgbm']['predicted_class'][i]
        
        xgb_conf = predictions['xgboost']['confidence'][i]
        
        match = "[OK]" if xgb_pred == true_class else "[X]"
        
        print(f"  {i+1:2d}   |    {true_class:2d}     |  {xgb_pred:2d}    |     {rf_pred:2d}       |     {lr_pred:2d}      |     {nb_pred:2d}     |   {lgb_pred:2d}    {match} ({xgb_conf:.2f})")
    
    # Example 2: Voting ensemble
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Voting Ensemble (Majority Vote from 5 Models)")
    print("=" * 80)
    
    all_predictions = np.array([
        predictions['logistic_regression']['predicted_class'],
        predictions['naive_bayes']['predicted_class'],
        predictions['random_forest']['predicted_class'],
        predictions['xgboost']['predicted_class'],
        predictions['lightgbm']['predicted_class'],
    ])
    
    # Majority vote
    from scipy.stats import mode
    ensemble_predictions = mode(all_predictions, axis=0, keepdims=True)[0].flatten()
    
    print(f"\nEnsemble Predictions (majority vote from 5 models):\n")
    print("Sample | True Class | Ensemble Vote | Match?")
    print("-" * 50)
    
    for i in range(len(X_test_samples)):
        true_class = y_test_labels[i]
        ensemble_pred = ensemble_predictions[i]
        match = "[OK]" if ensemble_pred == true_class else "[X]"
        print(f"  {i+1:2d}   |    {true_class:2d}     |      {ensemble_pred:2d}       |  {match}")
    
    # Example 3: XGBoost only (best single model)
    print("\n" + "=" * 80)
    print("EXAMPLE 3: XGBoost Only (Best Single Model - 98.95% accuracy)")
    print("=" * 80)
    
    xgb_predictions = predictions['xgboost']['predicted_class']
    xgb_confidence = predictions['xgboost']['confidence']
    
    accuracy = np.mean(xgb_predictions == y_test_labels)
    
    print(f"\nXGBoost Predictions:\n")
    print(f"Accuracy on 10 test samples: {accuracy:.1%}")
    print(f"\nDetailed Results:\n")
    print("Sample | True Class | Predicted | Confidence | Status")
    print("-" * 58)
    
    for i in range(len(X_test_samples)):
        true_class = y_test_labels[i]
        pred_class = xgb_predictions[i]
        conf = xgb_confidence[i]
        status = "[OK] CORRECT" if pred_class == true_class else "[X] WRONG"
        print(f"  {i+1:2d}   |    {true_class:2d}     |    {pred_class:2d}     |   {conf:.4f}   | {status}")
    
    print("\n" + "=" * 80)
    print("[OK] INFERENCE COMPLETE")
    print("=" * 80)
    print("\nTo use models on your own data:")
    print("  1. Load data: df = pd.read_parquet('your_data.parquet')")
    print("  2. Ensure 35 feature columns match feature_names")
    print("  3. Call: predictions = predict_attacks(df, models, scaler, feature_names)")
    print("  4. Access: predictions['xgboost']['predicted_class']  # [0-35 attack types]")
    print("  5. Check confidence: predictions['xgboost']['confidence']  # [0-1]")
