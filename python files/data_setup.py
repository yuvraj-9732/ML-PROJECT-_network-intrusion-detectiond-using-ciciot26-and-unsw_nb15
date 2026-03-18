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

# ============================================================================
# 1. LOAD & EXPLORE DATA
# ============================================================================
print("=" * 80)
print("STEP 1: Loading and Analyzing Data")
print("=" * 80)

# Load cleaned data
df_full = pd.read_parquet('merged_clean.parquet')
print(f"\nFull dataset shape: {df_full.shape}")
print(f"Features: {df_full.shape[1] - 1}, Samples: {df_full.shape[0]:,}")

# Sample 1M rows for efficient training while preserving class distribution
print("\n[*] Sampling 1M rows from full dataset for computational efficiency...")
# Use stratified sampling to preserve class proportions
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

# Stratified train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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
    Train model and evaluate with F1 (weighted/macro) + Accuracy metrics
    """
    print(f"\n{'-'*60}")
    print(f"Training: {model_name}")
    print(f"{'-'*60}")
    
    # Training time
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics - Training
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_f1_macro = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
    
    # Metrics - Testing
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    
    # Cross-validation: 3-fold stratified on a STRATIFIED SUBSAMPLE of training data.
    # IMPORTANT: CV runs only on X_train/y_train — never on X_test/y_test.
    #
    # Why subsample (30k rows)?  Full train set is ~800k rows.  Running 3-fold CV
    # on the full set triples the training cost (e.g., LR alone would take ~20 min).
    # A 30k stratified sample keeps all 36 classes represented and produces
    # reliable F1 estimates (Law of Large Numbers) in seconds instead of hours.
    CV_SAMPLE = 30_000
    import pandas as pd
    import numpy as _np

    # Build a fresh 0-based series (avoids original DataFrame label mis-indexing)
    _y_arr = _np.array(y_train) if not isinstance(y_train, _np.ndarray) else y_train
    _y_pos = pd.Series(_y_arr)   # clean 0-based positional index

    # Stratified sample: pick proportional rows from each class
    _pos_idx = _y_pos.groupby(_y_pos).apply(
        lambda g: g.sample(
            min(len(g), max(1, CV_SAMPLE * len(g) // len(_y_pos))),
            random_state=42,
        )
    ).index.get_level_values(1).values           # positional ints, guaranteed 0-based
    _pos_idx = _pos_idx[:CV_SAMPLE]              # cap at CV_SAMPLE

    # Index into X_train (works for both numpy arrays and DataFrames)
    if hasattr(X_train, 'iloc'):
        X_cv = X_train.iloc[_pos_idx]
    else:
        X_cv = X_train[_pos_idx]
    y_cv = _y_arr[_pos_idx]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_cv, y_cv, cv=cv, scoring='f1_weighted')
    
    # Inference time on test set
    start = time.time()
    _ = model.predict(X_test)
    inference_time = time.time() - start
    
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
