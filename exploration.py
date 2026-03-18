import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load encoded dataset ──────────────────────────────────────────────────────
# First, just want to look at the data and understand what we're dealing with
print("Loading merged_encoded.parquet ...")
print("    (this should be the preprocessed data from data loading step)")
df = pd.read_parquet("merged_encoded.parquet")
print(f"Shape: {df.shape[0]:,} rows  x  {df.shape[1]} columns")

# Make pandas output wider so we can see everything
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  BASIC INFO - Just see what's in the data
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Columns & dtypes ──────────────────────────────────────────")
print(df.dtypes.to_string())

print("\n── Null counts (check for missing values) ────────────────────")
null_counts = df.isnull().sum()
print(null_counts[null_counts > 0].to_string() if null_counts.any() else "  No nulls — clean dataset ✔")

print("\n── Descriptive statistics (min/max/mean/etc) ──────────────────")
print(df.describe().to_string())

print("\n── First 5 rows (what does the actual data look like?) ──────")
print(df.head(5).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 2.  CORRELATION HEATMAP - See which features are redundant
# ─────────────────────────────────────────────────────────────────────────────
print("\nComputing correlation matrix ...")
print("    (this might take a moment with all the features...)")
corr = df.corr()

print(f"[*] Creating huge heatmap ({df.shape[1]}x{df.shape[1]})")
fig, ax = plt.subplots(figsize=(28, 24))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    annot=False,   # set to True if you want values on heatmap (but it's slow)
    linewidths=0.3,
    cbar_kws={"shrink": 0.6, "label": "Pearson r"},
    ax=ax,
)
ax.set_title("Correlation Heatmap — Encoded Dataset", fontsize=18, pad=16)
plt.tight_layout()
plt.savefig("heatmap_encoded.png", dpi=150)
plt.close()
print("Saved → heatmap_encoded.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  PAIR PLOT  (top 8 features by mean absolute correlation)
# ─────────────────────────────────────────────────────────────────────────────
mean_abs_corr = corr.abs().mean().sort_values(ascending=False)
top_features  = mean_abs_corr.head(8).index.tolist()
print(f"\nTop 8 features for pair plot: {top_features}")

SAMPLE_N  = 2000
label_col = "label"

n_classes = df[label_col].nunique()
per_class = max(1, SAMPLE_N // n_classes)
frames = []
for cls, grp in df[top_features + [label_col]].groupby(label_col):
    frames.append(grp.sample(min(len(grp), per_class), random_state=42))
sample = pd.concat(frames).sample(frac=1, random_state=42).head(SAMPLE_N)
print(f"Pair plot sample: {len(sample):,} rows  |  {sample[label_col].nunique()} classes")

print("Rendering pair plot ...")
g = sns.pairplot(
    sample,
    hue=label_col,
    vars=top_features,
    plot_kws={"alpha": 0.4, "s": 15},
    diag_kind="kde",
    corner=True,
    palette="tab10",
)
g.figure.suptitle("Pair Plot — Top 8 Features (Encoded Dataset)", y=1.01, fontsize=16)
g.figure.set_size_inches(18, 18)
plt.tight_layout()
plt.savefig("pairplot_encoded.png", dpi=120)
plt.close()
print("Saved → pairplot_encoded.png")

print("\n✅ All done! Check heatmap_encoded.png & pairplot_encoded.png")
