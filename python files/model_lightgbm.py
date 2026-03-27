"""
═══════════════════════════════════════════════════════════════════════════════
  MODEL 5: LIGHTGBM (Fast Gradient Boosting)
  Network Intrusion Detection System
═══════════════════════════════════════════════════════════════════════════════

MATHEMATICAL FOUNDATION:
─────────────────────────────────────────────────────────────────────────────
Leaf-wise Tree Growth (different from XGBoost level-wise):
  While Gain > min_gain_to_split:
      Find leaf with max ImpurityReduction
      Split that leaf on best feature

Leaf Score Optimization (Gradient Descent):
  Gain = (n_L * n_R)/(n_L + n_R) * (Δȳ_L - Δȳ_R)²
  where Δȳ = gradient residuals, n = sample count

Fast Information Gain Computation:
  Uses histogram-based bucketing (256 bins) instead of exact splits
  Reduces memory by 10-50x while maintaining accuracy

Sequential Boosting (same as XGBoost):
  ŷ = Σ α_m * f_m(x)  [m=1 to 100 trees]
  Each tree corrects previous residuals

Theory: Speed-optimized gradient boosting via leaf-wise growth
Note: Underperformed (25.79%) due to aggressive splitting on imbalanced data
─────────────────────────────────────────────────────────────────────────────
"""

from data_setup import (
    X_train, X_test,          # raw features — trees are scale-invariant
    y_train, y_test,
    evaluate_model, imbalance_ratio, X
)
import lightgbm as lgb
from tqdm import tqdm
import pickle
import os

# ============================================================================
# MODEL 5: LIGHTGBM
# ============================================================================
print("\n" + "=" * 80)
print("MODEL 5: LIGHTGBM (Fast Gradient Boosting)")
print("=" * 80)
print("""
Mathematical Equation (Leaf-wise Growth):
  Gain = (n_L * n_R)/(n_L + n_R) * (Δȳ_L - Δȳ_R)²
  While max(Gain) > min_gain_to_split:
      Find deepest leaf with max gain
      Split that leaf
  
Speed Optimization:
  Uses histogram-based bucketing (256 bins)
  Reduces memory by 10-50x vs exact splits
  Parallelizable across leaves
  
Sequential Boosting (like XGBoost):
  ŷ = Σ α_m * f_m(x) [m=1 to 100]
  
Why: Faster training than XGBoost (39s vs 145s)
When: Need speed, have hyperparameter tuning time
Note: Underperformed here (25.79%) - leaf-wise growth too aggressive on 5810x imbalance
Fix: Would improve with min_child_samples=100, reg_lambda=10.0
""")

N_ESTIMATORS = 100

pipeline_stages = ["Setup", "Training & Evaluation", "Saving model"]
pbar_pipeline = tqdm(pipeline_stages, desc="LightGBM Pipeline",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

# ── Stage 1: Setup ───────────────────────────────────────────────────────────
pbar_pipeline.set_description("[1/3] Setup")

# LightGBM per-tree tqdm callback
pbar_trees = tqdm(total=N_ESTIMATORS, desc="  LightGBM trees", unit="tree", leave=True,
                  bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} trees [{elapsed}<{remaining}]")

class TqdmLGBCallback:
    """LightGBM callback that updates a tqdm bar after each boosting round."""
    def __init__(self, pbar):
        self.pbar = pbar
        self.order = 10
        self.before_iteration = False

    def __call__(self, env):
        self.pbar.update(1)
        if env.evaluation_result_list:
            name, _, value, _ = env.evaluation_result_list[0]
            self.pbar.set_postfix({name: f"{value:.4f}"})

lgb_model = lgb.LGBMClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced',
    reg_lambda=10.0,
    # GPU acceleration disabled due to imbalanced data
    # device_type='gpu',
    min_child_samples=100,  # Prevent overfitting with GPU on imbalanced data
    callbacks=[TqdmLGBCallback(pbar_trees)],
)
pbar_pipeline.update(1)

# ── Stage 2: Training & Evaluation ───────────────────────────────────────────
pbar_pipeline.set_description("[2/3] Training & Evaluating")
# Tree models use RAW (unscaled) features — LightGBM's histogram binning
# is scale-invariant. Raw features keep feature importance interpretable.
results_lgb = evaluate_model(
    lgb_model, X_train, X_test, y_train, y_test,
    "5. LightGBM (Fast Gradient Boosting)",
)
pbar_trees.close()
pbar_pipeline.update(1)

# ── Stage 3: Save ─────────────────────────────────────────────────────────────────
pbar_pipeline.set_description("[3/3] Saving model")

models_dir = 'saved_models'
os.makedirs(models_dir, exist_ok=True)

filepath = os.path.join(models_dir, 'lightgbm.pkl')
with open(filepath, 'wb') as f:
    pickle.dump(results_lgb['Model Object'], f)

pbar_pipeline.update(1)
pbar_pipeline.close()

print("\n" + "=" * 80)
print("Saving LightGBM Model")
print("=" * 80)
print(f"  [OK] lightgbm.pkl saved to {models_dir}/")

print("\n" + "=" * 80)
print("[OK] LIGHTGBM TRAINING COMPLETE")
print("=" * 80)
print(f"\n  Test Accuracy:      {results_lgb['Test Accuracy']:.4f}")
print(f"  Test F1 (weighted): {results_lgb['Test F1 (weighted)']:.4f}")
print(f"  Test F1 (macro):    {results_lgb['Test F1 (macro)']:.4f}")
print(f"  CV F1 (weighted):   {results_lgb['CV F1 (weighted)']:.4f}")
print(f"  Overfitting Gap:    {results_lgb['Overfitting Gap']:.4f}")
print(f"  Training Time:      {results_lgb['Training Time (s)']:.2f}s")
print(f"  Inference Time:     {results_lgb['Inference Time (s)']:.3f}s")
