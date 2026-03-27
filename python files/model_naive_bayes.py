"""
═══════════════════════════════════════════════════════════════════════════════
  MODEL 2: NAIVE BAYES (Probabilistic Baseline)
  Network Intrusion Detection System
═══════════════════════════════════════════════════════════════════════════════

MATHEMATICAL FOUNDATION:
─────────────────────────────────────────────────────────────────────────────
Bayes' Theorem (probabilistic classification):
  P(class k | x) = P(x | class k) * P(class k) / P(x)

Naive Assumption (feature independence):
  P(x | class k) = prod P(x_j | class k) for j=1 to 35

Gaussian Likelihood (each feature assumed normal):
  P(x_j | class k) = (1 / sqrt(2*pi*sigma_k^2)) * exp(-(x_j - mu_k)^2 / (2*sigma_k^2))

Decision Rule (argmax posterior):
  class* = argmax_k P(class k | x) ∝ P(x | class k) * P(class k)

Theory: Uses Bayes' theorem with independence assumption for fast inference
─────────────────────────────────────────────────────────────────────────────
"""

from data_setup import (
    X_train_scaled, X_test_scaled, y_train, y_test,
    evaluate_model, imbalance_ratio, X
)
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import pickle
import os

# ============================================================================
# MODEL 2: NAIVE BAYES
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 2: NAIVE BAYES (Probabilistic Baseline)")
print("=" * 80)
print("""
Mathematical Equation (Bayes' Theorem):
  P(class k | x) = P(x | class k) * P(class k) / P(x)
  P(x | class k) = prod P(x_j | class k)  [independence assumption]
  
Gaussian Likelihood:
  P(x_j | class k) = (1 / sqrt(2*pi*sigma_k^2)) * exp(-(x_j - mu_k)^2 / (2*sigma_k^2))
  
Decision Rule:
  class* = argmax_k P(class k | x) * P(x | class k)
  
Why: Fast Bayesian inference, probabilistic uncertainty
When: Limited data, need speed, baseline comparison
Importance: Simple baseline shows impact of independence assumptions
Weakness: Assumes features independent (they're not!)
""")

pipeline_stages = ["Setup", "Training & Evaluation", "Saving model"]
pbar_pipeline = tqdm(pipeline_stages, desc="Naive Bayes Pipeline",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

# Stage 1: Setup
pbar_pipeline.set_description("[1/3] Setup")
nb_model = GaussianNB(var_smoothing=1e-5)
pbar_pipeline.update(1)

# Stage 2: Training & Evaluation
pbar_pipeline.set_description("[2/3] Training & Evaluating")
results_nb = evaluate_model(nb_model, X_train_scaled, X_test_scaled, y_train, y_test,
                            "2. Naive Bayes (Probabilistic Baseline)")
pbar_pipeline.update(1)

# Stage 3: Save
pbar_pipeline.set_description("[3/3] Saving model")
models_dir = 'saved_models'
os.makedirs(models_dir, exist_ok=True)

filepath = os.path.join(models_dir, 'naive_bayes.pkl')
with open(filepath, 'wb') as f:
    pickle.dump(results_nb['Model Object'], f)
pbar_pipeline.update(1)
pbar_pipeline.close()

print("\n" + "=" * 80)
print("Saving Naive Bayes Model")
print("=" * 80)
print(f"  [OK] naive_bayes.pkl saved to {models_dir}/")

print("\n" + "=" * 80)
print("[OK] NAIVE BAYES TRAINING COMPLETE")
print("=" * 80)
print(f"\n  Test Accuracy:      {results_nb['Test Accuracy']:.4f}")
print(f"  Test F1 (weighted): {results_nb['Test F1 (weighted)']:.4f}")
print(f"  Test F1 (macro):    {results_nb['Test F1 (macro)']:.4f}")
print(f"  CV F1 (weighted):   {results_nb['CV F1 (weighted)']:.4f}")
print(f"  Overfitting Gap:    {results_nb['Overfitting Gap']:.4f}")
print(f"  Training Time:      {results_nb['Training Time (s)']:.2f}s")
print(f"  Inference Time:     {results_nb['Inference Time (s)']:.3f}s")
