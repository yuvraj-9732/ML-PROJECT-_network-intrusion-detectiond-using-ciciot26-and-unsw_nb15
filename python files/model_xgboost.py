"""
═══════════════════════════════════════════════════════════════════════════════
  MODEL 4: XGBOOST (Regularized + Tuned + Early Stopping)
  Network Intrusion Detection System
═══════════════════════════════════════════════════════════════════════════════
"""

from data_setup import (
    X_train, X_test,
    y_train, y_test,
    evaluate_model, imbalance_ratio, X
)

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import time
from tqdm import tqdm

pipeline_stages = ["Setup & Best Parameters", "Final Training", "Evaluation & Save"]
pbar_pipeline = tqdm(pipeline_stages, desc="XGBoost Pipeline",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

# ── Stage 1: Base Model Setup & Best Parameters ───────────────────────────────
pbar_pipeline.set_description("[1/3] Setup & Best Parameters")

print("\n[INFO] Using pre-tuned best hyperparameters...")
best_params = {
    'subsample': 0.5,
    'reg_lambda': 10,
    'reg_alpha': 1,
    'n_estimators': 100,
    'min_child_weight': 10,
    'max_depth': 3,
    'learning_rate': 0.01,
    'gamma': 5,
    'colsample_bytree': 0.5
}

best_xgb = xgb.XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    tree_method='hist',
    device='cuda:0',  # will fallback to CPU if GPU not available
    random_state=42,
    **best_params
)

pbar_pipeline.update(1)

# ── Stage 2: Final Training (Early Stopping) ─────────────────────────────────
pbar_pipeline.set_description("[2/3] Final Training (Early Stopping)")
print("\n[INFO] Training with Early Stopping...")

n_estimators = best_xgb.get_params().get('n_estimators', 200)
pbar_trees = tqdm(total=n_estimators, desc="  XGBoost trees", unit="tree", leave=True,
                  bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} trees [{elapsed}<{remaining}]")

class TqdmTreeCallback(xgb.callback.TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        pbar_trees.update(1)
        loss = list(list(evals_log.values())[0].values())[0][-1] if evals_log else None
        if loss is not None:
            pbar_trees.set_postfix(val_loss=f"{loss:.4f}")
        return False

best_xgb.set_params(
    early_stopping_rounds=20,
    callbacks=[TqdmTreeCallback()]
)
best_xgb.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=True
)
pbar_trees.close()
pbar_pipeline.update(1)

# ── Stage 3: Evaluation & Save ───────────────────────────────────────────────
# Remove early_stopping_rounds & tqdm callback before evaluate_model —
# evaluate_model calls fit() without an eval_set, which would crash early stopping.
pbar_pipeline.set_description("[3/3] Evaluation & Save")
best_xgb.set_params(early_stopping_rounds=None, callbacks=[])
results_xgb = evaluate_model(
    best_xgb,
    X_train,
    X_test,
    y_train,
    y_test,
    "4. XGBoost (Tuned + Regularized)"
)

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("Feature Importance Analysis - XGBoost")
print("=" * 80)

feature_names = X.columns
xgb_importance = best_xgb.feature_importances_
xgb_importance_sorted = np.argsort(xgb_importance)[-15:]

plt.figure(figsize=(10, 8))
plt.barh(feature_names[xgb_importance_sorted],
         xgb_importance[xgb_importance_sorted])
plt.xlabel('Importance', fontweight='bold')
plt.title('XGBoost - Top 15 Features', fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_xgb.png', dpi=300)
plt.close()

print("[OK] Feature importance plot saved")
pbar_pipeline.update(1)
pbar_pipeline.close()

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("Saving XGBoost Model")
print("=" * 80)

models_dir = 'saved_models'
os.makedirs(models_dir, exist_ok=True)

filepath = os.path.join(models_dir, 'xgboost_tuned.pkl')

with open(filepath, 'wb') as f:
    pickle.dump(best_xgb, f)

print(f"[OK] xgboost_tuned.pkl saved to {models_dir}/")

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("[OK] XGBOOST TRAINING COMPLETE")
print("=" * 80)

print(f"\n  Test Accuracy:      {results_xgb['Test Accuracy']:.4f}")
print(f"  Test F1 (weighted): {results_xgb['Test F1 (weighted)']:.4f}")
print(f"  Test F1 (macro):    {results_xgb['Test F1 (macro)']:.4f}")
print(f"  CV F1 (weighted):   {results_xgb['CV F1 (weighted)']:.4f}")
print(f"  Overfitting Gap:    {results_xgb['Overfitting Gap']:.4f}")
print(f"  Training Time:      {results_xgb['Training Time (s)']:.2f}s")
print(f"  Inference Time:     {results_xgb['Inference Time (s)']:.3f}s")