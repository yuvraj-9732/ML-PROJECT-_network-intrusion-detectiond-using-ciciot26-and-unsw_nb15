"""
═══════════════════════════════════════════════════════════════════════════════
  MODEL 1: LOGISTIC REGRESSION (Linear Baseline)
  Network Intrusion Detection System
═══════════════════════════════════════════════════════════════════════════════

MATHEMATICAL FOUNDATION:
─────────────────────────────────────────────────────────────────────────────
Sigmoid Function (outputs probability between 0 and 1):
  sigma(z) = 1 / (1 + e^(-z))
  where z = w^T * x + b (linear combination of features)

Multi-class Softmax (for 36 attack classes):
  P(class k | x) = e^(w_k^T * x) / Sum(e^(w_j^T * x)) for j=1 to 36

Loss Function (Cross-Entropy):
  L = -Sum(y_k * log(y_pred_k)) for all samples and classes
  Minimized via gradient descent:
  dL/dw = (1/n) * Sum(y_pred_i - y_i) * x_i

Theory: Learns linear decision boundaries using probabilistic outputs
─────────────────────────────────────────────────────────────────────────────
"""

from data_setup import (
    X_train_scaled, X_test_scaled, y_train, y_test,
    evaluate_model, imbalance_ratio, X
)
from sklearn.linear_model import LogisticRegression
import pickle
import os

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 1: LOGISTIC REGRESSION (Linear Baseline)")
print("=" * 80)
print("""
Mathematical Equation:
  sigma(z) = 1 / (1 + e^(-z))
  P(class k | x) = e^(w_k^T * x) / Sum(e^(w_j^T * x))
  
Loss Function:
  L = -Sum y_k * log(y_pred_k)
  Optimizer: L-BFGS (quasi-Newton method)
  
Why: Linear decision boundaries, fast, interpretable
When: Use for baselines and when linear patterns exist
Importance: Establishes baseline accuracy, shows feature weights
""")

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
results_lr = evaluate_model(lr_model, X_train_scaled, X_test_scaled, y_train, y_test,
                            "1. Logistic Regression (Linear Baseline)")

# ============================================================================
# SAVE MODEL
# ============================================================================
print("\n" + "=" * 80)
print("Saving Logistic Regression Model")
print("=" * 80)

models_dir = 'saved_models'
os.makedirs(models_dir, exist_ok=True)

filepath = os.path.join(models_dir, 'logistic_regression.pkl')
with open(filepath, 'wb') as f:
    pickle.dump(results_lr['Model Object'], f)
print(f"  [OK] logistic_regression.pkl saved to {models_dir}/")

print("\n" + "=" * 80)
print("[OK] LOGISTIC REGRESSION TRAINING COMPLETE")
print("=" * 80)
print(f"\n  Test Accuracy:      {results_lr['Test Accuracy']:.4f}")
print(f"  Test F1 (weighted): {results_lr['Test F1 (weighted)']:.4f}")
print(f"  Test F1 (macro):    {results_lr['Test F1 (macro)']:.4f}")
print(f"  CV F1 (weighted):   {results_lr['CV F1 (weighted)']:.4f}")
print(f"  Overfitting Gap:    {results_lr['Overfitting Gap']:.4f}")
print(f"  Training Time:      {results_lr['Training Time (s)']:.2f}s")
print(f"  Inference Time:     {results_lr['Inference Time (s)']:.3f}s")
