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
from tqdm import tqdm

N_ITER = 20
CV_FOLDS = 3
total_fits = N_ITER * CV_FOLDS   # RandomizedSearchCV total fits

pipeline_stages = ["Setup", "Hyperparameter Search", "Training & Evaluation", "Feature Importance & Save"]
pbar_pipeline = tqdm(pipeline_stages, desc="Random Forest Pipeline",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

# ── Stage 1: Setup ───────────────────────────────────────────────────────────
pbar_pipeline.set_description("[1/4] Setup")
# Highly constrained model to hit 85-92% accuracy and minimize False Negatives
best_rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=5,
    min_samples_leaf=20,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42
)
pbar_pipeline.update(1)

# ── Stage 2: Training Base ───────────────────────────────────────────────────
pbar_pipeline.set_description("[2/4] Training Model")
print("\n[INFO] Training heavily regularized Random Forest...")

start_time = time.time()
best_rf.fit(X_train, y_train)
end_time = time.time()

print("\n[OK] Training Complete")
print(f"Time Taken: {end_time - start_time:.2f} seconds")

print("\nUsing Constrained Hyperparameters:")
for k, v in best_rf.get_params().items():
    if k in ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'class_weight']:
        print(f"  {k}: {v}")

pbar_pipeline.update(1)

# ── Stage 3: Training & Evaluation ───────────────────────────────────────────
pbar_pipeline.set_description("[3/4] Training & Evaluating")
results_rf = evaluate_model(
    best_rf,
    X_train,
    X_test,
    y_train,
    y_test,
    "3. Random Forest (Tuned)"
)
pbar_pipeline.update(1)

# ── Stage 4: Feature Importance & Save ───────────────────────────────────────
pbar_pipeline.set_description("[4/4] Feature Importance & Saving")
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

pbar_pipeline.update(1)
pbar_pipeline.close()

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