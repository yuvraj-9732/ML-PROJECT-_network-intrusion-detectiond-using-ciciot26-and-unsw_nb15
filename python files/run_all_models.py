"""
═══════════════════════════════════════════════════════════════════════════════
  Run All 5 Models — GPU-Accelerated Training + Per-Model TXT Reports
═══════════════════════════════════════════════════════════════════════════════

  GPU Usage:
    XGBoost  → device='cuda:0'   (NVIDIA GPU via CUDA — auto-fallback to CPU)
    LightGBM → device_type='gpu' (NVIDIA GPU via OpenCL — auto-fallback to CPU)
    Random Forest, Logistic Regression, Naive Bayes → CPU only (sklearn)

  Output:
    results/<model>_report.txt   — full metrics + confusion matrix + cl. report
    results/all_models_summary.txt — side-by-side comparison of all 5 models
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import io
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── Change working dir so parquet & saved_models resolve correctly ────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)   # ML_project root
os.chdir(project_dir)
sys.path.insert(0, script_dir)              # so data_setup can be imported

# ── Shared data (loads once for all 5 models) ────────────────────────────────
print("=" * 80)
print("Loading shared data via data_setup.py ...")
print("=" * 80)

from data_setup import (
    X_train, X_test,
    X_train_scaled, X_test_scaled,
    y_train, y_test,
    evaluate_model, imbalance_ratio, X, scaler
)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import pickle
import time

# ── Output directory ──────────────────────────────────────────────────────────
results_dir = os.path.join(project_dir, 'results')
os.makedirs(results_dir, exist_ok=True)
models_dir  = os.path.join(project_dir, 'saved_models')
os.makedirs(models_dir, exist_ok=True)

# ═════════════════════════════════════════════════════════════════════════════
# Helper: save per-model txt report
# ═════════════════════════════════════════════════════════════════════════════

def save_model_report(result, y_test, filepath):
    """Write accuracy, confusion matrix, and full classification report to txt."""
    y_pred = result['Predictions']
    labels  = np.unique(np.concatenate([y_test, y_pred]))

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"  MODEL REPORT: {result['Model']}\n")
        f.write("=" * 80 + "\n\n")

        f.write("── PERFORMANCE METRICS ────────────────────────────────────────────────────\n")
        f.write(f"  Train Accuracy          : {result['Train Accuracy']:.6f}  ({result['Train Accuracy']*100:.4f}%)\n")
        f.write(f"  Test  Accuracy          : {result['Test Accuracy']:.6f}  ({result['Test Accuracy']*100:.4f}%)\n")
        f.write(f"  Train F1 (weighted)     : {result['Train F1 (weighted)']:.6f}\n")
        f.write(f"  Test  F1 (weighted)     : {result['Test F1 (weighted)']:.6f}\n")
        f.write(f"  Test  F1 (macro)        : {result['Test F1 (macro)']:.6f}\n")
        f.write(f"  CV    F1 (weighted, 3k) : {result['CV F1 (weighted)']:.6f}\n")
        f.write(f"  Overfitting Gap         : {result['Overfitting Gap']:.6f}\n")
        f.write(f"  Training Time           : {result['Training Time (s)']:.3f}s\n")
        f.write(f"  Inference Time          : {result['Inference Time (s)']:.3f}s\n")

        # Overfitting verdict
        gap = result['Overfitting Gap']
        if gap < 0.02:
            verdict = "✓ EXCELLENT generalization (gap < 2%)"
        elif gap < 0.05:
            verdict = "✓ GOOD generalization (gap 2-5%)"
        elif gap < 0.10:
            verdict = "⚠ MILD overfitting (gap 5-10%)"
        else:
            verdict = "✗ OVERFITTING DETECTED (gap > 10%)"
        f.write(f"  Overfitting Verdict     : {verdict}\n\n")

        # ── Confusion Matrix ──────────────────────────────────────────────────
        f.write("── CONFUSION MATRIX ───────────────────────────────────────────────────────\n")
        f.write("  (rows = True class, columns = Predicted class)\n")
        f.write("  Only classes present in the test set are shown.\n\n")

        cm = confusion_matrix(y_test, y_pred, labels=labels)

        # Header row
        header  = "        " + "".join(f"{int(l):>6}" for l in labels)
        f.write(header + "\n")
        f.write("  " + "-" * (len(header) - 2) + "\n")
        for i, row in enumerate(cm):
            f.write(f"  {int(labels[i]):>4} | " + "".join(f"{v:>6}" for v in row) + "\n")
        f.write("\n")

        # Per-class diagonal (correct predictions)
        f.write("  Per-class correct predictions (diagonal):\n")
        for i, lbl in enumerate(labels):
            total = cm[i].sum()
            correct = cm[i, i]
            pct = 100 * correct / total if total > 0 else 0
            f.write(f"    Class {int(lbl):>3}: {correct:>8,} / {total:>8,}  ({pct:>6.2f}%)\n")
        f.write("\n")

        # ── Classification Report ─────────────────────────────────────────────
        f.write("── CLASSIFICATION REPORT ──────────────────────────────────────────────────\n")
        f.write(classification_report(y_test, y_pred, labels=labels, zero_division=0))
        f.write("\n")

    print(f"  [OK] Saved → {filepath}")


# ═════════════════════════════════════════════════════════════════════════════
# Model Definitions
# ═════════════════════════════════════════════════════════════════════════════

# GPU detection note
print("\n" + "=" * 80)
print("GPU CONFIGURATION (NVIDIA GeForce RTX 3050 Laptop)")
print("=" * 80)
print("  XGBoost  → device='cuda:0'          [CUDA — auto-fallback to CPU]")
print("  LightGBM → device_type='gpu'         [OpenCL — auto-fallback to CPU]")
print("  RF / LR / NB → CPU only (sklearn has no GPU backend)")
print()

MODELS = [
    {
        "name": "1_logistic_regression",
        "label": "1. Logistic Regression (Linear Baseline)",
        "pkl":   "logistic_regression.pkl",
        "model": LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
        ),
        "X_tr": X_train_scaled,   # SCALED — required for gradient-based optimizer
        "X_te": X_test_scaled,
        "gpu":  False,
        "gpu_note": "N/A — sklearn LR has no GPU backend",
    },
    {
        "name": "2_naive_bayes",
        "label": "2. Naive Bayes (Probabilistic Baseline)",
        "pkl":   "naive_bayes.pkl",
        "model": GaussianNB(),
        "X_tr": X_train_scaled,   # SCALED — Gaussian likelihood benefits from normalization
        "X_te": X_test_scaled,
        "gpu":  False,
        "gpu_note": "N/A — sklearn NB has no GPU backend",
    },
    {
        "name": "3_random_forest",
        "label": "3. Random Forest (Tree Ensemble)",
        "pkl":   "random_forest.pkl",
        "model": RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample',
        ),
        "X_tr": X_train,   # RAW — trees are scale-invariant
        "X_te": X_test,
        "gpu":  False,
        "gpu_note": "N/A — sklearn RF has no GPU backend (use cuML/RAPIDS for GPU RF)",
    },
    {
        "name": "4_xgboost",
        "label": "4. XGBoost (Gradient Boosting)",
        "pkl":   "xgboost.pkl",
        "model": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            tree_method='hist',   # histogram-based — works on both CPU & GPU
            device='cuda:0',      # NVIDIA GPU (auto-fallback to CPU if unavailable)
        ),
        "X_tr": X_train,   # RAW — XGBoost histogram splits are scale-invariant
        "X_te": X_test,
        "gpu":  True,
        "gpu_note": "CUDA via device='cuda:0' — RTX 3050 supported",
    },
    {
        "name": "5_lightgbm",
        "label": "5. LightGBM (Fast Gradient Boosting)",
        "pkl":   "lightgbm.pkl",
        "model": lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            device_type='gpu',    # NVIDIA GPU via OpenCL (auto-fallback to CPU)
            min_child_samples=5,
            verbose=-1,
        ),
        "X_tr": X_train,   # RAW — LightGBM histogram binning is scale-invariant
        "X_te": X_test,
        "gpu":  True,
        "gpu_note": "OpenCL via device_type='gpu' — RTX 3050 supported (needs GPU-enabled LGB build)",
    },
]

# ═════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═════════════════════════════════════════════════════════════════════════════

all_results = []

for cfg in MODELS:
    print("\n" + "=" * 80)
    print(f"TRAINING: {cfg['label']}")
    print(f"  GPU: {'YES — ' + cfg['gpu_note'] if cfg['gpu'] else 'NO  — ' + cfg['gpu_note']}")
    print("=" * 80)

    # If GPU model fails (driver/library issue), retry on CPU
    try:
        result = evaluate_model(
            cfg["model"],
            cfg["X_tr"], cfg["X_te"],
            y_train, y_test,
            cfg["label"],
        )
    except Exception as e:
        print(f"\n  [!] GPU training failed ({e})")
        print(f"  [*] Retrying on CPU ...")

        # Rebuild model without GPU settings
        if cfg["name"] == "4_xgboost":
            cpu_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=7, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, eval_metric='mlogloss', tree_method='hist',
            )
        elif cfg["name"] == "5_lightgbm":
            cpu_model = lgb.LGBMClassifier(
                n_estimators=100, max_depth=7, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, device_type='cpu', min_child_samples=5, verbose=-1,
            )
        else:
            raise  # non-GPU models shouldn't fail
        result = evaluate_model(
            cpu_model,
            cfg["X_tr"], cfg["X_te"],
            y_train, y_test,
            cfg["label"] + " [CPU fallback]",
        )

    # Save pkl
    pkl_path = os.path.join(models_dir, cfg["pkl"])
    with open(pkl_path, 'wb') as f:
        pickle.dump(result['Model Object'], f)
    print(f"  [OK] Model saved → saved_models/{cfg['pkl']}")

    # Save per-model txt report
    txt_path = os.path.join(results_dir, f"{cfg['name']}_report.txt")
    save_model_report(result, y_test, txt_path)

    all_results.append(result)


# ═════════════════════════════════════════════════════════════════════════════
# Combined Summary txt
# ═════════════════════════════════════════════════════════════════════════════

summary_path = os.path.join(results_dir, 'all_models_summary.txt')

with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 90 + "\n")
    f.write("  ALL MODELS — COMPARISON SUMMARY\n")
    f.write("  Dataset: CICIoT2023 + UNSW-NB15 (merged_clean.parquet)\n")
    f.write("  Features: 35 predictors → label (36 attack classes, integer 0-35)\n")
    f.write("  GPU: XGBoost (CUDA) + LightGBM (OpenCL) | RF/LR/NB: CPU only\n")
    f.write("=" * 90 + "\n\n")

    # Header
    col = "{:<50} {:>10} {:>12} {:>12} {:>10} {:>10} {:>10}\n"
    f.write(col.format(
        "Model", "Test Acc", "F1-Weighted", "F1-Macro", "CV-F1", "Gap", "Train(s)"
    ))
    f.write("-" * 90 + "\n")

    sorted_results = sorted(all_results, key=lambda r: r['Test Accuracy'], reverse=True)
    for r in sorted_results:
        name = r['Model'][:49]
        f.write(col.format(
            name,
            f"{r['Test Accuracy']:.4f}",
            f"{r['Test F1 (weighted)']:.4f}",
            f"{r['Test F1 (macro)']:.4f}",
            f"{r['CV F1 (weighted)']:.4f}",
            f"{r['Overfitting Gap']:+.4f}",
            f"{r['Training Time (s)']:.1f}s",
        ))

    f.write("\n" + "=" * 90 + "\n")
    best = sorted_results[0]
    f.write(f"\n  BEST MODEL : {best['Model']}\n")
    f.write(f"  Test Accuracy : {best['Test Accuracy']:.4f} ({best['Test Accuracy']*100:.2f}%)\n")
    f.write(f"  F1 Weighted   : {best['Test F1 (weighted)']:.4f}\n")
    f.write(f"  F1 Macro      : {best['Test F1 (macro)']:.4f}\n")
    f.write(f"  Overfitting   : {best['Overfitting Gap']:+.6f}\n")
    f.write("\n" + "=" * 90 + "\n")

print(f"\n  [OK] Summary  → {summary_path}")

# ═════════════════════════════════════════════════════════════════════════════
# Console summary
# ═════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("ALL MODELS TRAINED — FINAL SUMMARY")
print("=" * 80)
print(f"\n{'Model':<50} {'TestAcc':>8} {'F1-W':>8} {'F1-M':>8} {'CV-F1':>8} {'Gap':>8}")
print("-" * 80)
for r in sorted_results:
    print(f"{r['Model'][:49]:<50} {r['Test Accuracy']:>8.4f} "
          f"{r['Test F1 (weighted)']:>8.4f} {r['Test F1 (macro)']:>8.4f} "
          f"{r['CV F1 (weighted)']:>8.4f} {r['Overfitting Gap']:>+8.4f}")

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)
for cfg in MODELS:
    print(f"  results/{cfg['name']}_report.txt")
print(f"  results/all_models_summary.txt")
print()
