# SECOM Process Sensor Analysis
This project uses the SECOM semiconductor manufacturing sensor data. The purpose of this data science project is to seperate relevant features from noise and irrelevant features.

Interview at  mon feb 2nd 2 30 pm...

So, let me walk you through the full pipeline I followed for the SECOM project.  and why each step was in that order.

PREPROCESSING CLEANING IMPUTATION REMOVING CONSTANT FEATURES EDA, PCA

First, I started with *median imputation* to fill in missing values, because sensor data in fabs often has gaps from failed measurements or downtime. Median is robust to outliers, which is important in noisy industrial data. I also *dropped constant features*.  Then, I did exporatory data analysis to look for patterns. I could have gone further here with domain experts to identify which near constant process-specific constants might be worth keeping, but that was my initial cleaning/EDA pass.
Next, I applied *Principal Component Analysis (PCA)*.
I created *scree plots and cumulative explained variance plots* to choose how many components to keep.  I targeted 85% variance, which gave us 109 components from the original 475 features.
I then visualized loadings and score plots to understand clustering between principal components.
The goal here was to reduce dimensionality, remove redundant features, and maximize captured variance in an exploratory way.
PCA is more about understanding overall structure and noise than directly selecting predictive features.  so it's exploratory rather than predictive. I could have gone much deeper with domain experts (e.g., "Does PC1 align with temperature/pressure in deposition?") to interpret loadings in process terms.
After PCA, I moved to supervised feature selection using two wrapper methods:

*Boruta*  an all-relevant selection algorithm that compares real features against randomized shadow features using Random Forest importance. It gave us 64 confirmed + 7 tentative features.
RFE (Recursive Feature Elimination) with XGBoost as the estimator.  recursively removes the least important features until 80 remain.

FEATURE SELECTION BORUTA RFE

I noticed good overlap between *Boruta and RFE* (e.g., Feature_32, Feature_34, Feature_60 appear in both top lists).  that's a strong validation signal.
This step was critical because PCA is unsupervised and variance-focused; Boruta/RFE are supervised and target-specific.  they identify features that actually help predict fails, not just explain variance.

IMBALANCE SMOTE

Before modeling, I addressed the extreme imbalance (~93.4% pass vs 6.6% fail) using SMOTE (Synthetic Minority Over-sampling Technique). SMOTE generates synthetic fail examples by interpolating between real ones and their nearest neighbors.  applied only on training data to avoid leakage.

BASE MODEL TRAINING/TESTING/EVALUATION XGBOOST

Finally, I trained my base model: *XGBoost*, a gradient-boosted tree algorithm that's excellent for tabular, heterogeneous data like sensor readings. It handles missing values natively, provides feature importance, and works well with scale_pos_weight for imbalance.

LOW RECALL, USED CROSS VALIDATION

Even after all this, recall on fails was still low in initial runs (~0.03–0.06 without balancing, 0.19–0.29 with SMOTE).

UNSUPERVISED ANOMALY DETECTION ISOLATION FOREST ALGORITHM INSTEAD OF CLASSIFICATION, DETECT FAILS AS ANOMALIES

So I pivoted to Isolation Forest, an unsupervised anomaly detection algorithm based on random partitioning trees. It treats fails as anomalies.  no labels needed.  and achieved 0.226 recall at 10% contamination.
Looking at the CV results (stratified 5-fold), the strongest performer was Boruta-selected features + SMOTE, with average fail recall of 0.396 ± 0.100.  meaning ~40% of defects detected on average across folds.

CROSS VALIDATION, BEST RESULT HAS 33-40% RECALL, STILL MEANINGFUL, ESPECIALLY FOR BASE RUN

Even at 33–40% recall, that's meaningful in manufacturing: catching one-third more fails early avoids scrap, rework, and customer returns.  potentially saving thousands per batch and millions annually in a fab. With more data, senior data scientists, and domain experts (process engineers, physicists, materials scientists), I could fine-tune further: interpret top features in process context, try hybrid sampling (SMOTEENN), add cost-sensitive learning, or incorporate temporal trends.

The *key lesson:* in real semiconductor data, the signal is subtle and buried in noise. No single algorithm solves it.  it's iterative refinement, collaboration with domain experts, and choosing the right tool (supervised vs anomaly) for the business goal: early defect detection to optimize yield and reduce costs.


