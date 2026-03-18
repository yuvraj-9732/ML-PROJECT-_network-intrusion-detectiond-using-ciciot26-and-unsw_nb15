"""
═══════════════════════════════════════════════════════════════════════════════
 MODEL 3: RANDOM FOREST (with Hyperparameter Tuning)
 Network Intrusion Detection System
═══════════════════════════════════════════════════════════════════════════════
"""

from data_setup import (
    X_train, X_test,
    y_train, y_test,
    evaluate_model, imbalance_ratio, X
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time

# ============================================================================
# RANDOM FOREST WITH RANDOMIZED SEARCH
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 3: RANDOM FOREST (Hyperparameter Tuning)")
print("=" * 80)

# Base model
rf_base = RandomForestClassifier(random_state=42)

# Hyperparameter space (VERY IMPORTANT DESIGN)
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
    'bootstrap': [True, False],
    'class_weight': ['balanced', 'balanced_subsample']
}

# Randomized Search
print("\n[INFO] Running Randomized Search...")

start_time = time.time()

rf_random = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=20,                 # Increase to 50+ for better tuning
    scoring='f1_weighted',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

rf_random.fit(X_train, y_train)

end_time = time.time()

print("\n[OK] Randomized Search Complete")
print(f"Time Taken: {end_time - start_time:.2f} seconds")

# Best model
best_rf = rf_random.best_estimator_

print("\nBest Parameters Found:")
for k, v in rf_random.best_params_.items():
    print(f"  {k}: {v}")

# ============================================================================
# EVALUATION USING BEST MODEL
# ============================================================================
results_rf = evaluate_model(
    best_rf,
    X_train,
    X_test,
    y_train,
    y_test,
    "3. Random Forest (Tuned)"
)

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("Feature Importance Analysis - Random Forest")
print("=" * 80)

feature_names = X.columns
rf_importance = best_rf.feature_importances_
rf_importance_sorted = np.argsort(rf_importance)[-15:]

plt.figure(figsize=(10, 8))
plt.barh(feature_names[rf_importance_sorted],
         rf_importance[rf_importance_sorted])
plt.xlabel('Importance', fontweight='bold')
plt.title('Random Forest - Top 15 Features', fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_rf.png', dpi=300)
plt.close()

print("[OK] Feature importance plot saved")

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("Saving Random Forest Model")
print("=" * 80)

models_dir = 'saved_models'
os.makedirs(models_dir, exist_ok=True)

filepath = os.path.join(models_dir, 'random_forest_tuned.pkl')

with open(filepath, 'wb') as f:
    pickle.dump(best_rf, f)

print(f"[OK] random_forest_tuned.pkl saved to {models_dir}/")

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("[OK] RANDOM FOREST TRAINING COMPLETE")
print("=" * 80)

print(f"\n  Test Accuracy:      {results_rf['Test Accuracy']:.4f}")
print(f"  Test F1 (weighted): {results_rf['Test F1 (weighted)']:.4f}")
print(f"  Test F1 (macro):    {results_rf['Test F1 (macro)']:.4f}")
print(f"  CV F1 (weighted):   {results_rf['CV F1 (weighted)']:.4f}")
print(f"  Overfitting Gap:    {results_rf['Overfitting Gap']:.4f}")
print(f"  Training Time:      {results_rf['Training Time (s)']:.2f}s")
print(f"  Inference Time:     {results_rf['Inference Time (s)']:.3f}s")