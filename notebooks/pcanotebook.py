# %% [markdown]
# ## SECOM – Principal Component Analysis (PCA)
# 
# In this phase, we will perform a basic principal component analysis. Principal Component Analysis (PCA) is a natural next step to address redundancy. PCA shows that a small subset of principal components captures most of the dataset variance, reducing dimensionality while preserving information.
# 
# ### Key Steps:
# 
# 1. **Standardize features**
# 
# 2. **Apply PCA**
# 
# 3. **Plot cumulative explained variance**
# 
# 
# By projecting correlated features into a smaller set of orthogonal principal components, PCA captures the maximum variance in fewer dimensions while reducing noise. This step preserves the underlying structure of the data, enabling more effective classification of the Pass/Fail target.
# 

# %% [markdown]
# Import libraries

# %%
import importlib.util
import sys
from pathlib import Path

# CHANGE THIS PATH to where dpf.py actually lives
dpf_path = Path("/home/theodorescottwillis/Documents/GitHub/SECOM-Process-Sensor-Analysis/dpf.py")

spec = importlib.util.spec_from_file_location("dpf", dpf_path)
dpf = importlib.util.module_from_spec(spec)
sys.modules["dpf"] = dpf
spec.loader.exec_module(dpf)

# Now test
dpf.Check


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import dpf
from sklearn.decomposition import PCA

# better looking plots
plt.style.use('seaborn-v0_8-whitegrid')
%matplotlib inline

# %% [markdown]
# Load cleaned data

# %%
cleaned_df = pd.read_csv("secom_cleaned.csv")

# %% [markdown]
# Drop non-numeric columns

# %%
numeric_df = cleaned_df.select_dtypes(include='number')

# %%
dpf.Check(numeric_df)

# %% [markdown]
# Standardize features

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(numeric_df.drop(columns=['Pass/Fail']) 
                               if 'Pass/Fail' in numeric_df.columns 
                               else numeric_df)

print("Scaled data shape:", X_scaled.shape)

# %% [markdown]
# Apply PCA

# %%
# FULL PCA (to understand variance distribution)
pca_full = PCA()
pca_full.fit(X_scaled)

# Explained variance per component
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

n_features = X_scaled.shape[1]

# %% [markdown]
# SCREE PLOT + CUMULATIVE VARIANCE

# %%
plt.figure(figsize=(14, 6))

# Scree plot (left)
plt.subplot(1, 2, 1)
plt.plot(range(1, len(explained_var) + 1), explained_var, 'o-', color='teal')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')
plt.grid(True)

# Cumulative (right)
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'o-', color='darkorange')
plt.axhline(y=0.80, color='gray', linestyle='--', alpha=0.7, label='80% threshold')
plt.axhline(y=0.90, color='gray', linestyle=':', alpha=0.7, label='90% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
# How many components for common thresholds?

# %%
thresholds = [0.70, 0.80, 0.85, 0.90, 0.95]

for thresh in thresholds:
    n_comp = np.argmax(cumulative_var >= thresh) + 1
    actual_var = cumulative_var[n_comp-1]
    print(f"To keep ≥ {thresh:.0%} variance → {n_comp} components "
          f"({actual_var:.3%} actual)")

# Also show what 80 fixed components would give
if n_features >= 80:
    print(f"\nFirst 80 components explain: {cumulative_var[79]:.3%}")

# %% [markdown]
# CHOOSE & RUN TARGET PCA

# %%
# Recommended: variance-based (change percentage here)
variance_target = 0.85

pca = PCA(n_components=variance_target)
X_pca = pca.fit_transform(X_scaled)

n_components_kept = pca.n_components_
explained_total = pca.explained_variance_ratio_.sum()

print(f"Kept {n_components_kept} components → {explained_total:.1%} variance")
print(f"Shape of transformed data: {X_pca.shape}")

# %% [markdown]
# CREATE PCA DATAFRAME

# %%
pc_columns = [f"PC{i+1}" for i in range(n_components_kept)]

pca_df = pd.DataFrame(
    X_pca,
    columns=pc_columns,
    index=numeric_df.index
)


# %%
dpf.Check(pca_df)

# %% [markdown]
# If 'Pass/Fail' column exists, add it back to the PCA DataFrame

# %%
# Bring back the target variable
if 'Pass/Fail' in numeric_df.columns:
    pca_df['Pass/Fail'] = numeric_df['Pass/Fail']
else:
    print("Warning: 'Pass/Fail' not found in numeric_df")

print("PCA DataFrame shape:", pca_df.shape)
display(pca_df.head(6))   # use display() in Jupyter for nicer table

print(pca_df['Pass/Fail'].value_counts(normalize=True))



# %% [markdown]
# Confirm class distribution

# %%
print("\nClass distribution (normalized):")
display(pca_df['Pass/Fail'].value_counts(normalize=True).round(4) * 100)

print("\nRaw counts:")
display(pca_df['Pass/Fail'].value_counts())

# %% [markdown]
# Extract & Interpret Loadings

# %%
# Loadings = how much each original feature contributes to each PC
feature_names = numeric_df.drop(columns=['Pass/Fail']).columns \
    if 'Pass/Fail' in numeric_df.columns else numeric_df.columns

loadings = pd.DataFrame(
    pca.components_.T,                     # shape: (n_features, n_components)
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=feature_names
)

print("\nLoadings DataFrame shape:", loadings.shape)

# Show absolute top 8 contributors per PC (most influential sensors)
print("\nTop contributing features per PC (by absolute loading):")
for pc in loadings.columns[:6]:           # look at first 6 PCs
    top = loadings[pc].abs().sort_values(ascending=False).head(8)
    print(f"\n{pc}:")
    print(top.round(4))

# Optional: save loadings for later inspection
# loadings.to_csv("secom_pca_loadings.csv")

# %% [markdown]
# Biplot – Loadings + Scores (very useful for SECOM)

# %%
# ─── Debug: Check exact column names in pca_df ───────────────────────
print("pca_df columns:")
print(pca_df.columns.tolist())

print("\nFirst few rows to confirm:")
display(pca_df.head(3))

print("\nDoes 'PC1' exist?", 'PC1' in pca_df.columns)
print("Does 'PC1' (case sensitive) exist?", any(col == 'PC1' for col in pca_df.columns))
print("Sample PC-like columns:", [col for col in pca_df.columns if 'PC' in col or 'pc' in col][:6])

# %%
def plot_biplot(pc_x=1, pc_y=2, n_arrows=15):
    """Simple biplot: top contributing variables as arrows + points colored by class"""
    fig, ax = plt.subplots(figsize=(11, 9))
    
    # Scores (transformed data points)
    sns.scatterplot(
        data=pca_df,
        x=f'PC{pc_x}', y=f'PC{pc_y}',
        hue='Pass/Fail',
        style='Pass/Fail',
        palette={-1: '#1f77b4', 1: '#ff4444'},   # blue=pass, red=fail
        alpha=0.5,
        s=60,
        ax=ax
    )
    
    # Loadings arrows – only show top contributors to avoid clutter
    top_features = loadings[f'PC{pc_x}'].abs().add(loadings[f'PC{pc_y}'].abs()).nlargest(n_arrows).index
    
    for feat in top_features:
        lx, ly = loadings.loc[feat, f'PC{pc_x}'], loadings.loc[feat, f'PC{pc_y}']
        ax.arrow(0, 0, lx, ly,
                 color='darkgreen', alpha=0.7, head_width=0.015, length_includes_head=True)
        ax.text(lx*1.12, ly*1.12, feat,
                color='darkgreen', ha='center', va='center', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    ax.set_xlabel(f'PC{pc_x} ({explained_var[pc_x-1]:.2%} variance)')
    ax.set_ylabel(f'PC{pc_y} ({explained_var[pc_y-1]:.2%} variance)')
    ax.set_title(f'PCA Biplot: PC{pc_x} vs PC{pc_y} (top {n_arrows} loading vectors)')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    plt.legend(title='Pass/Fail')
    plt.tight_layout()
    plt.show()

# Plot first few pairs
plot_biplot(1, 2, n_arrows=12)
plot_biplot(1, 3, n_arrows=10)   # optional

# %% [markdown]
# Score Plots – Looking for separation by class

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (ax, pc_x, pc_y) in enumerate(zip(axes, [1,1,2], [2,3,3])):
    sns.scatterplot(
        data=pca_df,
        x=f'PC{pc_x}', y=f'PC{pc_y}',
        hue='Pass/Fail',
        style='Pass/Fail',
        palette={-1: '#1f77b4', 1: '#ff4444'},
        alpha=0.6,
        s=50,
        ax=ax
    )
    ax.set_title(f'PC{pc_x} vs PC{pc_y}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# Variance explained by the selected components

# %%
print(f"Number of components kept: {pca.n_components_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3%}")

plt.figure(figsize=(10,5))
plt.plot(range(1, pca.n_components_+1), np.cumsum(pca.explained_variance_ratio_), 'o-')
plt.axhline(y=0.85, color='gray', ls='--', label='85% target')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Variance captured by selected PCA')
plt.grid(True)
plt.legend()
plt.show()

# %% [markdown]
# Clustering in low-dimensional PCA space (exploratory)

# %%
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# %% [markdown]
# Use top 3–5 PCs (adjust based on your cumulative variance)

# %%
n_cluster_pcs = min(5, pca.n_components_)
X_for_clust = pca_df[[f'PC{i+1}' for i in range(n_cluster_pcs)]]

# %% [markdown]
# Try 2–6 clusters

# %%
sil_scores = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_for_clust)
    sil = silhouette_score(X_for_clust, labels)
    sil_scores.append(sil)
    print(f"k={k} → silhouette={sil:.3f}")

# %% [markdown]
# Pick best (highest silhouette) or domain choice (e.g. k=2 or 3)

# %%
best_k = np.argmax(sil_scores) + 2
print(f"\nBest silhouette at k={best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
pca_df['Cluster'] = kmeans.fit_predict(X_for_clust)

# %% [markdown]
# Visualize

# %%
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=pca_df,
    x='PC1', y='PC2',
    hue='Cluster',
    palette='tab10',
    style='Pass/Fail',
    alpha=0.7,
    s=70
)
plt.title(f'PCA Space Clusters (k={best_k}) – colored by cluster, shaped by Pass/Fail')
plt.grid(True, alpha=0.3)
plt.show()

# %% [markdown]
# Crosstab: how clusters relate to actual Pass/Fail

# %%
print("\nCluster vs Pass/Fail crosstab:")
display(pd.crosstab(pca_df['Cluster'], pca_df['Pass/Fail'], normalize='index').round(3) * 100)

# %% [markdown]
# SAVE DATAFRAMES FOR LATER:

# %%
# 1. Original cleaned features — most important for feature selection
numeric_df.to_csv("secom_numeric_cleaned_475features.csv", index=False)
print("Saved: secom_numeric_cleaned_475features.csv")

# 2. PCA-transformed version — for direct modeling comparison
pca_df.to_csv("secom_pca_transformed_85pct.csv", index=False)
print("Saved: secom_pca_transformed_85pct.csv")

# 3. Loadings (for interpretation & possible manual selection)
loadings.to_csv("secom_pca_loadings.csv")
print("Saved: secom_pca_loadings.csv")


# Save the fitted scaler (useful if you ever need to transform new data)
import joblib
joblib.dump(scaler, "secom_scaler.pkl")
print("Saved scaler: secom_scaler.pkl")

# Save the fitted PCA object itself (can transform new data later)
joblib.dump(pca, "secom_pca_model_85pct.pkl")
print("Saved PCA object: secom_pca_model_85pct.pkl")

# %% [markdown]
# Optional next steps / reminders

# %%
print("""
Next logical steps after this PCA exploration:
  • Use top PCs or top-loading original features → train classifier (e.g. RandomForest, XGBoost, LogisticRegression with class_weight='balanced')
  • Try anomaly detection (Isolation Forest, Autoencoder) instead of classification — failures are rare
  • Compare model performance before/after PCA
  • Investigate top-loading sensors → domain interpretation (which process steps matter most?)
  For Boruta / RFE / XGBoost importance → need numeric_df (original features)
  For PCA-based modeling baseline → need pca_df
  For interpretation / storytelling → loadings
""")

# %% [markdown]
# ### Key PCA Insights
# 
# - Dimensionality reduced from 475 features to X components while retaining Y% variance → substantial compression due to sensor redundancy.
# - Cumulative variance curve shows rapid early gains → most information in first ~50–100 directions.
# - Exploratory clustering in PCA space reveals **two regimes**:
#   - Large "normal" cluster (93% pass).
#   - Smaller high-risk cluster (25% fail rate) → potential anomalous process state.
# - Takeaway: PCA effectively captures variance and subgroup structure, but linear separation of fails is limited → motivates supervised feature selection or non-linear methods next.


