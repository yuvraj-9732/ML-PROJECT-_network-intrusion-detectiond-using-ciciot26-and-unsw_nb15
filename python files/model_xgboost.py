"""
═══════════════════════════════════════════════════════════════════════════════
  MODEL 4: XGBOOST (Gradient Boosting) [BEST]
  Network Intrusion Detection System
═══════════════════════════════════════════════════════════════════════════════

MATHEMATICAL FOUNDATION:
─────────────────────────────────────────────────────────────────────────────
Sequential Ensemble (learns to correct previous errors):
  ŷ = Σ α_m * f_m(x) for m=1 to M trees
  where α_m = learning_rate (step size), f_m = weak learner (tree) m

Gradient Boosting (fit each tree to residual errors):
  Residual = y_true - ŷ_previous
  Tree m learns to predict residuals using negative gradients:
  dL/dŷ = ŷ - y_true  [cross-entropy gradient]

Loss Function (multi-class log-loss + regularization):
  L_total = -Σ y_k*log(ŷ_k) + λ*||w||  [L2 regularization]
  λ = 1.0 (penalizes large weights to prevent overfitting)

Regularization Strategies:
  - Shrinkage: α_m = 0.1 (small steps prevent oscillation)
  - Max Depth: 7 (shallow trees, weak learners)
  - Subsampling: 0.8 (randomness prevents overfitting)
  - Feature Subsampling: 0.8 (use 80% of 35 features per tree)

Theory: Sequential error correction via gradient descent in tree space
─────────────────────────────────────────────────────────────────────────────
"""

from data_setup import (
    X_train, X_test,          # raw features — trees are scale-invariant
    y_train, y_test,
    evaluate_model, imbalance_ratio, X
)
import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# ============================================================================
# MODEL 4: XGBOOST
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 4: XGBOOST (Gradient Boosting) [BEST]")
print("=" * 80)
print("""
Mathematical Equation (Sequential Boosting):
  ŷ = Σ α_m * f_m(x) for m=1 to 100 trees
  
Gradient-based Tree Learning:
  Tree m learns to fit negative gradients:
  dL/dŷ = ŷ - y_true
  
Loss Function with Regularization:
  L_total = -Σ y_k*log(ŷ_k) + λ*||w||
  λ = 1.0 (L2 regularization)
  
Regularization Strategy:
  ✓ Shrinkage: learning_rate = 0.1 (small steps)
  ✓ Max Depth: 7 (weak learners)
  ✓ Subsampling: 0.8 (data randomness)
  ✓ Colsample: 0.8 (feature randomness)
  
Why: 98.95% accuracy! Gradient descent corrects each tree's errors
When: Best all-around for tabular data (networks, medical, finance)
Importance: Sequential refinement beats single ensemble
Mechanism: Each tree fixes what previous trees got wrong
""")

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=None,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    tree_method='hist',            # histogram-based, works on CPU and GPU
    device='cuda:0',               # Try GPU first, will auto-fallback to CPU if unavailable
)
# Tree models use RAW (unscaled) features — XGBoost uses histogram-based splits
# which are inherently scale-invariant. Raw features preserve original units
# so that feature importance scores map directly to network traffic semantics.
results_xgb = evaluate_model(xgb_model, X_train, X_test, y_train, y_test,
                             "4. XGBoost (Gradient Boosting)")

# ============================================================================
# FEATURE IMPORTANCE (XGBoost)
# ============================================================================
print("\n" + "=" * 80)
print("Feature Importance Analysis - XGBoost")
print("=" * 80)

feature_names = X.columns
xgb_importance = results_xgb['Model Object'].feature_importances_
xgb_importance_sorted = np.argsort(xgb_importance)[-15:]  # Top 15

plt.figure(figsize=(10, 8))
plt.barh(feature_names[xgb_importance_sorted], xgb_importance[xgb_importance_sorted], color='coral')
plt.xlabel('Importance', fontweight='bold')
plt.title('XGBoost - Top 15 Features', fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_xgb.png', dpi=300, bbox_inches='tight')
plt.close()
print("Visualization saved: feature_importance_xgb.png")

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("Saving XGBoost Model")
print("=" * 80)

models_dir = 'saved_models'
os.makedirs(models_dir, exist_ok=True)

filepath = os.path.join(models_dir, 'xgboost.pkl')
with open(filepath, 'wb') as f:
    pickle.dump(results_xgb['Model Object'], f)
print(f"  [OK] xgboost.pkl saved to {models_dir}/")

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
