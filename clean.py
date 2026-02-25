import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load encoded dataset ──────────────────────────────────────────────────────
print("Loading merged_encoded.parquet ...")
df = pd.read_parquet("merged_encoded.parquet")
print(f"Original shape: {df.shape[0]:,} rows  x  {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Correlation-based feature selection
#   Strategy:
#   - Compute full correlation matrix
#   - For any pair of features with |corr| > THRESHOLD, drop the one
#     with the LOWER mean absolute correlation to the rest of the dataset
#     (i.e. keep the more "informative" one of the pair)
#   - Always protect the target column 'label'
# ─────────────────────────────────────────────────────────────────────────────
THRESHOLD  = 0.90          # drop one feature if |corr| with another >= this
TARGET_COL = "label"

print(f"\n[1] Computing correlation matrix (threshold = {THRESHOLD}) ...")
corr_matrix = df.drop(columns=[TARGET_COL]).corr().abs()

# Upper triangle mask
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Mean absolute correlation of each feature (proxy for "informativeness")
mean_corr = corr_matrix.mean()

to_drop = set()
for col in upper.columns:
    if col in to_drop:
        continue
    # Features highly correlated with this column
    highly_correlated = upper[col][upper[col] >= THRESHOLD].index.tolist()
    for partner in highly_correlated:
        if partner in to_drop:
            continue
        # Drop whichever has lower mean absolute correlation
        if mean_corr[col] >= mean_corr[partner]:
            to_drop.add(partner)
        else:
            to_drop.add(col)
            break   # col is dropped; move to next col

print(f"\n    Features before : {df.shape[1] - 1}  (excl. target)")
print(f"    Features to drop: {len(to_drop)}")
print(f"    Features kept   : {df.shape[1] - 1 - len(to_drop)}")
print(f"\n    Dropped columns:\n      {sorted(to_drop)}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Also drop near-zero variance features
# ─────────────────────────────────────────────────────────────────────────────
VAR_THRESHOLD = 1e-5
low_var_cols = [
    c for c in df.columns
    if c != TARGET_COL and c not in to_drop and df[c].var() < VAR_THRESHOLD
]
print(f"\n[2] Near-zero variance columns (var < {VAR_THRESHOLD}): {low_var_cols}")
to_drop.update(low_var_cols)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Drop and save
# ─────────────────────────────────────────────────────────────────────────────
df_clean = df.drop(columns=list(to_drop))
print(f"\n[3] Final shape after cleaning: {df_clean.shape[0]:,} rows  x  {df_clean.shape[1]} columns")
print(f"    Remaining features:\n      {[c for c in df_clean.columns if c != TARGET_COL]}")

out_path = "merged_clean.parquet"
df_clean.to_parquet(out_path, index=False)
print(f"\n✅ Saved → {out_path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Heatmap of the KEPT features to confirm low redundancy
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Plotting correlation heatmap of kept features ...")
kept_features = [c for c in df_clean.columns if c != TARGET_COL]
corr_clean = df_clean[kept_features].corr()

fig, ax = plt.subplots(figsize=(max(12, len(kept_features) * 0.55),
                                max(10, len(kept_features) * 0.5)))
mask = np.triu(np.ones_like(corr_clean, dtype=bool))
sns.heatmap(
    corr_clean,
    mask=mask,
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    annot=len(kept_features) <= 30,   # annotate only if compact enough
    fmt=".2f",
    linewidths=0.4,
    cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    ax=ax,
)
ax.set_title(f"Correlation Heatmap — {len(kept_features)} Kept Features (threshold={THRESHOLD})",
             fontsize=14, pad=12)
plt.tight_layout()
plt.savefig("heatmap_clean.png", dpi=150)
plt.close()
print("Saved → heatmap_clean.png")

print("\n✅ All done!")
print(f"   Original features : {df.shape[1] - 1}")
print(f"   Dropped           : {len(to_drop)}")
print(f"   Final features    : {df_clean.shape[1] - 1}  (+ target '{TARGET_COL}')")
