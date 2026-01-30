# %% [markdown]
# # 05 – SECOM: Supervised Feature Selection & Modeling Baselines
# 
# **Previous phase**: PCA reduced dimensionality from 475 features to ~109 components explaining 85% variance, but showed limited linear separation of fails.
# 
# **This phase goals**:
# - Perform supervised feature selection to identify sensors most relevant to defect prediction (Pass/Fail)
# - Address redundancy more interpretably than PCA alone
# - Build baseline models on:
#   - Original features
#   - Selected features (Boruta / RFE)
#   - PCA-transformed features
# - Compare performance with focus on minority class (fails): recall, F1, PR-AUC
# 
# **Key methods**:
# - Boruta (all-relevant selection)
# - RFE with XGBoost
# - Baseline classifiers: XGBoost, Random Forest (imbalance-aware)
# 
# **Why this matters**: In semiconductor manufacturing, identifying key defect-driving sensors enables targeted process monitoring and yield improvement.

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

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from boruta import BorutaPy

import warnings
warnings.filterwarnings('ignore')

# Better looking plots
plt.style.use('seaborn-v0_8-whitegrid')
# %matplotlib inline

print("Imports complete.")


# %% [markdown]
# Load cleaned data

# %%
# 1. Original cleaned features (main source for feature selection)
df = pd.read_csv("secom_numeric_cleaned_475features.csv")
print("Loaded original cleaned data:", df.shape)

# Separate features and target
X = df.drop(columns=['Pass/Fail'])
feature_names = X.columns.tolist()  # keep for later reference
y = df['Pass/Fail']                 # -1 / +1

# Map to 0/1 for modeling (Boruta and many libs prefer this)
y_binary = y.map({-1: 0, 1: 1})

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts(normalize=True).round(3) * 100}")

# 2. PCA-transformed version (for comparison)
pca_df = pd.read_csv("secom_pca_transformed_85pct.csv")
X_pca = pca_df.drop(columns=['Pass/Fail'])
y_pca = pca_df['Pass/Fail']  # keep original -1/+1 for now

print("PCA data loaded:", X_pca.shape)

# %% [markdown]
# BORUTA STARTER – ALL-RELEVANT FEATURE SELECTION

# %%
# Base estimator: Random Forest with imbalance handling
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

# Boruta setup
print("Starting BorutaPy...")
boruta_selector = BorutaPy(
    rf,
    n_estimators='auto',
    verbose=2,
    random_state=42,
    max_iter=150,         # adjust higher if needed (can take 15–60 min)
    perc=90               # confidence threshold (90% is standard)
)

# Fit Boruta (uses numpy arrays)
boruta_selector.fit(X.values, y_binary.values)

# Extract results
confirmed_mask = boruta_selector.support_
tentative_mask = boruta_selector.support_weak_

confirmed_features = [feature_names[i] for i in range(len(feature_names)) if confirmed_mask[i]]
tentative_features = [feature_names[i] for i in range(len(feature_names)) if tentative_mask[i]]
rejected_features = [feature_names[i] for i in range(len(feature_names)) 
                     if not confirmed_mask[i] and not tentative_mask[i]]

print("\nBoruta Results:")
print(f"  Confirmed important features: {len(confirmed_features)}")
print(f"  Tentative features:           {len(tentative_features)}")
print(f"  Rejected features:            {len(rejected_features)}")

# Save confirmed features list
pd.Series(confirmed_features).to_csv("secom_boruta_confirmed_features.csv", index=False)
print("Saved confirmed features to: secom_boruta_confirmed_features.csv")

# Quick preview
print("\nTop 15 confirmed features:")
print(confirmed_features[:15])

# Optional: save full support masks for later use
pd.DataFrame({
    'Feature': feature_names,
    'Confirmed': confirmed_mask,
    'Tentative': tentative_mask
}).to_csv("secom_boruta_support.csv", index=False)

# %% [markdown]
# Save & Preview Confirmed + Tentative Features

# %%
# Combine confirmed + tentative for modeling
selected_features = confirmed_features + tentative_features
print(f"Total selected (confirmed + tentative): {len(selected_features)}")

# Save full list
pd.Series(selected_features).to_csv("secom_boruta_selected_features.csv", index=False)
print("Saved selected features to: secom_boruta_selected_features.csv")

# Quick look at the selected ones
print("\nSelected features (first 20):")
print(selected_features[:20])

# %% [markdown]
# Quick Model Comparison: Full vs Boruta vs PCA

# %%
def evaluate_model(X_tr, X_te, y_tr, y_te, name):
    model = XGBClassifier(
        scale_pos_weight=14,
        eval_metric='aucpr',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    
    print(f"\n=== {name} ===")
    print(classification_report(y_te, y_pred, target_names=['Pass (0)', 'Fail (1)']))
    
    precision, recall, _ = precision_recall_curve(y_te, y_prob)
    pr_auc = auc(recall, precision)
    print(f"PR-AUC: {pr_auc:.4f}")
    
    return model

# Stratified split – use the same indices for fair comparison
X_train_full, X_test_full, y_train, y_test = train_test_split(
    X, y_binary,
    test_size=0.3,
    stratify=y_binary,
    random_state=42
)

# 1. Full original features
print("Evaluating Full Features...")
evaluate_model(X_train_full, X_test_full, y_train, y_test, "Full (474 features)")

# 2. Boruta selected features
X_selected = X[selected_features]  # DataFrame
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(
    X_selected, y_binary,
    test_size=0.3,
    stratify=y_binary,
    random_state=42
)
print("Evaluating Boruta Selected...")
evaluate_model(X_train_sel, X_test_sel, y_train_sel, y_test_sel, "Boruta Selected (~71 features)")

# 3. PCA transformed features
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(
    X_pca, y_pca.map({-1: 0, 1: 1}),
    test_size=0.3,
    stratify=y_binary,
    random_state=42
)
print("Evaluating PCA...")
evaluate_model(X_pca_train, X_pca_test, y_pca_train, y_pca_test, "PCA Transformed")


