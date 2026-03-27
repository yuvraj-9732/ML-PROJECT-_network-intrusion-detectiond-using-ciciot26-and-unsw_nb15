"""
═══════════════════════════════════════════════════════════════════════════════
  Shared Data Setup & Utilities
  Used by: model_logistic_regression.py, model_naive_bayes.py,
           model_random_forest.py, model_xgboost.py, model_lightgbm.py
═══════════════════════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix, classification_report
)
import time
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# ============================================================================
# 1. LOAD & EXPLORE DATA
# ============================================================================
print("=" * 80)
print("STEP 1: Loading and Analyzing Data")
print("=" * 80)

# Load cleaned data
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parquet_path = os.path.join(script_dir, 'merged_clean.parquet')
df_full = pd.read_parquet(parquet_path)
print(f"\nFull dataset shape: {df_full.shape}")
print(f"Features: {df_full.shape[1] - 1}, Samples: {df_full.shape[0]:,}")

# Sample 1M rows for efficient training while preserving class distribution
print("\n[*] Sampling 1M rows from full dataset for computational efficiency...")
# Ensure stratify works: if a class in df_full has only 1 sample, stratify will crash.
# Our dataset has a minimum class size of 25, so stratify=df_full['label'] is safe for the full set.
df_sample, _ = train_test_split(df_full, test_size=0.88, random_state=42, stratify=df_full['label'])
df = df_sample.reset_index(drop=True)
print(f"Working with dataset shape: {df.shape}")
print(f"Features: {df.shape[1] - 1}, Samples: {df.shape[0]:,}")

# Check for data issues
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"Data types:\n{df.dtypes.value_counts()}")

# ============================================================================
# 2. CLASS DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Class Distribution Analysis")
print("=" * 80)

# Analyze label distribution
label_dist = df['label'].value_counts()
print(f"\nTotal classes: {len(label_dist)}")
print(f"\nClass distribution (top 10):")
print(label_dist.head(10))

imbalance_ratio = label_dist.max() / label_dist.min()
print(f"\nImbalance Ratio: {imbalance_ratio:.2f}x")
print(f"(Highest class: {label_dist.idxmax()} = {label_dist.max():,} samples)")
print(f"(Lowest class: {label_dist.idxmin()} = {label_dist.min():,} samples)")

# Visualize class distribution
plt.figure(figsize=(14, 6))
label_dist.plot(kind='bar', color='steelblue')
plt.title('Class Distribution Across 36 Attack Types', fontsize=14, fontweight='bold')
plt.xlabel('Attack Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("\nVisualization saved: class_distribution.png")

# ============================================================================
# 3. DATA PREPARATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Data Preparation (Train/Test Split & Normalization)")
print("=" * 80)

# ── Feature / Target split ──────────────────────────────────────────────────
# PREDICTORS (X): all 35 network-traffic feature columns
# TARGET     (y): 'label' — integer-encoded attack class (int64, values 0-35)
#   • 0  = Normal traffic
#   • 1-35 = 35 distinct attack / DoS / DDoS / Recon categories
#   NOTE: 'attack_cat' (coarser grouping) was dropped at feature-selection time
#         to prevent target leakage.
X = df.drop('label', axis=1)   # shape: (n_samples, 35)
y = df['label']                 # shape: (n_samples,)  dtype: int64

# Safety guard: label is already int64 in merged_clean.parquet.
# This branch only fires if the parquet is regenerated with string labels.
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"Target encoded to numeric (0-{len(le.classes_)-1})")
else:
    print(f"Target 'label': dtype={y.dtype}, classes={y.nunique()} (values {y.min()}-{y.max()})")

# Stratified train/test split (80/20). If a class has <2 samples in `df`, stratify will fail.
# We fall back to random split if stratification is impossible (e.g., when testing with small samples).
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"\nTrain set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set:  {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Verify stratification
print(f"\nClass distribution preserved:")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Standardize features (REQUIRED for Logistic Regression & Naive Bayes;
# optional but harmless for tree models — trees are scale-invariant).
# Scaler is fit ONLY on X_train, then applied to X_test to prevent leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform training set
X_test_scaled  = scaler.transform(X_test)        # transform only — no refit

print(f"\nFeatures standardized (mean=0, std=1)")
print(f"Feature statistics (scaled train set):")
print(f"  Mean: {X_train_scaled.mean():.6f}")
print(f"  Std:  {X_train_scaled.std():.6f}")
print(f"\nExported sets:")
print(f"  X_train        (raw) : {X_train.shape}  — use for tree models")
print(f"  X_train_scaled       : {X_train_scaled.shape}  — use for LR / NB")
print(f"  X_test         (raw) : {X_test.shape}")
print(f"  X_test_scaled        : {X_test_scaled.shape}")

# ============================================================================
# 4. MODEL TRAINING & EVALUATION FUNCTION
# ============================================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train model and evaluate with F1 (weighted/macro) + Accuracy metrics.
    Displays a tqdm progress bar across 4 pipeline stages.
    """
    print(f"\n{'-'*60}")
    print(f"Training: {model_name}")
    print(f"{'-'*60}")

    stages = ["Training model", "Generating predictions", "Cross-validation (3-fold)", "Inference timing"]
    pbar = tqdm(stages, desc=f"[{model_name}]", unit="stage",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} stages [{elapsed}<{remaining}]")

    # ── Stage 1: Training ────────────────────────────────────────────────────
    pbar.set_description(f"[{model_name}] 1/4 Training")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    pbar.update(1)

    # ── Stage 2: Predictions ─────────────────────────────────────────────────
    pbar.set_description(f"[{model_name}] 2/4 Predicting")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    pbar.update(1)

    # Metrics - Training
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_f1_macro = f1_score(y_train, y_train_pred, average='macro', zero_division=0)

    # Metrics - Testing
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

    # ── Stage 3: Cross-validation ────────────────────────────────────────────
    # 3-fold stratified CV on a 30k-row stratified subsample of X_train.
    # Full train set is ~800k rows; subsampling keeps CV fast while preserving
    # class proportions across all 36 attack classes.
    pbar.set_description(f"[{model_name}] 3/4 Cross-validation")
    CV_SAMPLE = 30_000
    import pandas as pd
    import numpy as _np
    from sklearn.base import clone as _clone

    _y_arr = _np.array(y_train) if not isinstance(y_train, _np.ndarray) else y_train
    _y_pos = pd.Series(_y_arr)

    # Guarantee at least 3 samples per class so StratifiedKFold(n_splits=3) never fails,
    # and XGBoost/LightGBM never see a truncated class list in any fold.
    _pos_idx_list = []
    for c, g in _y_pos.groupby(_y_pos):
        # We need at least 3 samples for 3-fold CV. If a class has < 3 in the entire
        # train set, we oversample it (replace=True) just for the CV subset.
        n_req = max(3, int(CV_SAMPLE * len(g) / len(_y_pos)))
        if len(g) >= n_req:
            sampled = g.sample(n_req, random_state=42)
        else:
            sampled = g.sample(n_req, replace=True, random_state=42)
        _pos_idx_list.extend(sampled.index.values)
    
    _pos_idx = _np.array(_pos_idx_list)
    _np.random.seed(42)
    _np.random.shuffle(_pos_idx)
    # Ensure it's roughly CV_SAMPLE (might be slightly larger if many rare classes)
    if len(_pos_idx) > CV_SAMPLE:
        _pos_idx = _pos_idx[:CV_SAMPLE]
        
    # Final safety check: if the random shuffle slice dropped a rare class below 3,
    # we just use the un-sliced list (it's only a few rows larger).
    if _y_pos.iloc[_pos_idx].value_counts().min() < 3:
        _pos_idx = _np.array(_pos_idx_list)

    if hasattr(X_train, 'iloc'):
        X_cv = X_train.iloc[_pos_idx]
    else:
        X_cv = X_train[_pos_idx]
    y_cv = _y_arr[_pos_idx]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores_list = []
    with tqdm(cv.split(X_cv, y_cv), total=3, desc="  CV folds", unit="fold", leave=False,
              bar_format="  {l_bar}{bar}| fold {n_fmt}/{total_fmt} [{elapsed}]") as fold_bar:
        for train_idx, val_idx in fold_bar:
            _yf_tr = y_cv[train_idx]
            if hasattr(X_cv, 'iloc'):
                _Xf_tr, _Xf_val = X_cv.iloc[train_idx], X_cv.iloc[val_idx]
                # LightGBM/XGBoost crash if _yf_tr is missing classes present in the full dataset.
                # Find missing classes (out of 36) and append 1 valid sample of each to the training fold.
                _yf_tr_series = _yf_tr if isinstance(_yf_tr, pd.Series) else pd.Series(_yf_tr)
                missing_classes = set(range(36)) - set(_yf_tr_series.unique())
                if missing_classes:
                    missing_X = []
                    missing_y = []
                    for c in missing_classes:
                        # find one instance of class c in the full X_train/y_train
                        idx = _y_arr == c
                        if idx.sum() > 0:
                            missing_X.append(X_train[idx].iloc[0:1] if hasattr(X_train, 'iloc') else X_train[idx][0:1])
                            missing_y.append(c)
                    if missing_X:
                        # append missing to _Xf_tr, _yf_tr
                        if hasattr(_Xf_tr, 'iloc'):
                            _Xf_tr = pd.concat([_Xf_tr] + missing_X, ignore_index=True)
                        else:
                            _Xf_tr = _np.vstack([_Xf_tr] + missing_X)
                        if isinstance(_yf_tr, pd.Series):
                            _yf_tr = pd.concat([_yf_tr, pd.Series(missing_y)], ignore_index=True)
                        else:
                            _yf_tr = _np.append(_yf_tr, missing_y)
            else:
                _Xf_tr, _Xf_val = X_cv[train_idx], X_cv[val_idx]
                _yf_tr_series = _yf_tr
                missing_classes = set(range(36)) - set(_np.unique(_yf_tr_series))
                if missing_classes:
                    missing_X = []
                    missing_y = []
                    for c in missing_classes:
                        idx = _y_arr == c
                        if idx.sum() > 0:
                            missing_X.append(X_train[idx][0:1])
                            missing_y.append(c)
                    if missing_X:
                        _Xf_tr = _np.vstack([_Xf_tr] + missing_X)
                        _yf_tr = _np.append(_yf_tr, missing_y)

            _yf_tr, _yf_val = _np.array(_yf_tr), _np.array(y_cv[val_idx])
            # Use a fresh clone each fold — avoids class-mismatch errors in
            # XGBoost/LightGBM when a previously-fitted model is re-fitted on a
            # fold subset that has a different set of classes (rare classes can
            # drop out of a 3-fold split of a 30k stratified subsample).
            _fold_model = _clone(model)
            _fold_model.fit(_Xf_tr, _yf_tr)
            _pred = _fold_model.predict(_Xf_val)
            cv_scores_list.append(f1_score(_yf_val, _pred, average='weighted', zero_division=0))
            fold_bar.set_postfix(f1=f"{cv_scores_list[-1]:.4f}")
    cv_scores = _np.array(cv_scores_list)
    pbar.update(1)

    # ── Stage 4: Inference timing ─────────────────────────────────────────────
    pbar.set_description(f"[{model_name}] 4/4 Inference timing")
    start = time.time()
    _ = model.predict(X_test)
    inference_time = time.time() - start
    pbar.update(1)
    pbar.close()

    # Results dictionary
    results = {
        'Model': model_name,
        'Train Accuracy': train_acc,
        'Test Accuracy': test_acc,
        'Train F1 (weighted)': train_f1_weighted,
        'Test F1 (weighted)': test_f1_weighted,
        'Test F1 (macro)': test_f1_macro,
        'CV F1 (weighted)': cv_scores.mean(),
        'Overfitting Gap': train_acc - test_acc,
        'Training Time (s)': train_time,
        'Inference Time (s)': inference_time,
        'Model Object': model,
        'Predictions': y_test_pred,
    }

    print(f"\nTRAINING METRICS:")
    print(f"  Accuracy:      {train_acc:.4f}")
    print(f"  F1 (weighted): {train_f1_weighted:.4f}")
    print(f"  F1 (macro):    {train_f1_macro:.4f}")

    print(f"\nTEST METRICS (PRIMARY):")
    print(f"  Accuracy:      {test_acc:.4f} [BEST]")
    print(f"  F1 (weighted): {test_f1_weighted:.4f} [BEST]")
    print(f"  F1 (macro):    {test_f1_macro:.4f}")

    print(f"\nCROSS-VALIDATION (3-fold):")
    print(f"  F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print(f"\nOVERFITTING CHECK:")
    print(f"  Train - Test Accuracy Gap: {results['Overfitting Gap']:.4f}")
    if results['Overfitting Gap'] > 0.1:
        print(f"  [!] Potential overfitting detected (gap > 0.10)")
    else:
        print(f"  [OK] Good generalization (gap < 0.10)")

    print(f"\nTIMING:")
    print(f"  Training:   {train_time:.3f}s")
    print(f"  Inference:  {inference_time:.3f}s")

    return results
