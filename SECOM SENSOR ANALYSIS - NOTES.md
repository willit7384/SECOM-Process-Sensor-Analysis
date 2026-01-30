SECOM SENSOR ANALYSIS - NOTES
THEODORE WILLIS


High-dimensional sensor data (~590 features originally)
Severe class imbalance (~93–94% pass / -1, ~6–7% fail / +1)
Heavy focus on missing values, redundancy, dimensionality reduction (PCA/feature selection), and handling rarity of defects
Goal: Often predictive modeling for defect detection (classification or anomaly detection), interpretability (which sensors/process steps matter?), and yield improvement insights

1\. SECOM - DATA UNDERSTANDING PHASE...DONE.

Dataset source: UCI Machine Learning Repository (SECOM – Semiconductor Manufacturing Process)
1567 instances, 591 columns (590 anonymous sensor measurements + 1 time column + 1 Pass/Fail target)
Target: Pass/Fail (-1 = pass/good wafer, +1 = fail/defective)
Severe imbalance observed (~93.4% pass, ~6.6% fail)
High missing values in many sensors (some >50%)
Features are real-valued, mostly continuous sensor readings at different process stages
Business context: Predict failures early to improve yield in semiconductor wafer fabrication
Key challenges identified: High dimensionality, multicollinearity/redundancy among sensors, class imbalance, noisy/missing data
Tools used: pandas profiling, basic descriptive stats, missing value heatmaps

2\. SECOM - DATA CLEANING PHASE...DONE.

Removed columns with excessive missing values (e.g., >55–60% threshold, common practice)
Imputed remaining missing values (median/mean for numeric, or KNN imputation in some approaches)
Handled outliers (winsorization or removal where extreme)
Dropped constant/variance-near-zero columns
Converted 'Pass/Fail' to numeric if needed (-1/1 or 0/1)
Resulting cleaned dataset: ~cleaned_df.csv~ with reduced features (~400–500 remaining after aggressive cleaning)
Validated: No missing values left, distributions checked for sanity
Output file: secom_cleaned.csv

3\. SECOM - EXPLORATORY DATA ANALYSIS PHASE...DONE.

Univariate: Histograms, boxplots for key sensors (after cleaning)
Bivariate: Correlations (heatmap showed high multicollinearity in sensor groups)
Class-specific analysis: Compared sensor distributions between pass vs fail (limited separation visible due to rarity of fails)
Time trends: Checked if any sensors drift over time (some do)
Missing value patterns: Confirmed many sensors missing in blocks (possibly failed sensors or process stages)
Key insights: Strong correlations among sensors → redundancy likely; fails are rare and subtle
Visuals created: Correlation matrix, pairplots (sampled), class balance pie/bar

4\. SECOM - PRINCIPAL COMPONENT ANALYSIS PHASE---IN PROGRESS.

**Multivariate Statistics** is where I will learn more about the principal component analysis (PCA) process, and that class typically comes after all three prerequisites:



Calculus III - Must come first (multivariable calculus is foundational)

Linear Algebra - Usually a prerequisite or co-requisite

Probability Theory - Must come first (statistical foundations)

Typical sequence:



Calculus I, II, III

Linear Algebra (often taken concurrently with or after Calc III)

Probability Theory

Statistics/Inferential Statistics

Then: Multivariate Statistics

Multivariate Statistics builds on all of these because it requires:



Understanding of vectors and matrices (Linear Algebra)

Partial derivatives and optimization (Calculus III)

Probability distributions and inference (Probability Theory)

Basic statistical concepts (Statistics)

So Multivariate Statistics is typically a junior/senior level course taken after 2-3 years of foundational mathematics


Standardized features (StandardScaler)
Full PCA to understand variance: Scree plot + cumulative variance plot created
Selected n_components for ~85% variance retention (typically 80–150 components depending on cleaning aggressiveness; your run kept a specific number for 85%)
Created pca_df with PC1–PCn + Pass/Fail
Extracted loadings: Top contributing sensors per PC identified (top 8 per first 6 PCs printed)
Biplots implemented: PC1 vs PC2, PC1 vs PC3 (arrows for top loadings, points colored by Pass/Fail)
Current observations:
Diminishing returns after moderate number of components (elbow visible)
Class separation in PCA space is weak/moderate (typical for SECOM – fails not linearly separable in low dims)
Loadings show groups of correlated sensors (potential process stage clusters)

**Next sub-tasks (to complete phase):**
Write 3–5 bullet insights from biplots/loadings (which sensors dominate PC1/PC2?)
Add more score plots (PC2 vs PC3, etc.)
Decide if PCA is final reduction or if feature selection (e.g., Boruta, loadings-based) is better for modeling

Progress: ~90% – ready for modeling input

**5\. SECOM - REDUNDANCY STORY PHASE...INCOMPLETE.**

Goal: Tell the "story" of multicollinearity/redundancy and how to reduce it meaningfully (beyond just PCA)
Planned steps:
Correlation-based clustering of sensors (e.g., hierarchical clustering on correlation matrix)
Group correlated features (e.g., sensors measuring similar aspects of the same process step)
Loadings interpretation: Identify redundant groups from PCA (variables pointing same direction/close angles)
Compare PCA vs other methods: Boruta, Recursive Feature Elimination (RFE), correlation thresholding, or variance thresholding
Domain story: Link back to semiconductor process if possible (etch, deposition, lithography sensors often cluster)

Deliverable: Markdown section + visuals (dendrogram, correlation clusters, top redundant pairs)
Why important: PCA mixes interpretability; redundancy removal keeps original features for better explainability in manufacturing

6\. SECOM - DATA MODELING PHASE...INCOMPLETE.

Planned approaches (common for SECOM):
Baseline models on cleaned original features (after feature selection)
Models on PCA-transformed data
Anomaly detection (Isolation Forest, One-Class SVM, Autoencoder) – often better for rare fails
Supervised classification: Logistic Regression, Random Forest, XGBoost, LightGBM (with class_weight='balanced' or SMOTE/oversampling)
Hyperparameter tuning (GridSearchCV or Optuna)
Cross-validation: StratifiedKFold due to imbalance

Metrics focus: Recall for fails (minimize missed defects), F1-score, Precision-Recall AUC, Cost-sensitive evaluation (false negatives expensive in manufacturing)
Compare: Original features vs PCA vs selected features

7\. SECOM - MODEL EVALUATION PHASE...INCOMPLETE.

Planned:
Confusion matrices, classification reports, PR curves
Feature importance (from tree models or permutation importance)
SHAP or LIME explanations on top models (which sensors drive predictions?)
Compare models: Which handles imbalance best? Which is most interpretable?
Error analysis: Look at misclassified fails (patterns in sensors?)

Deliverable: Table of model performances + key insights

8\. SECOM - LIMITATIONS/DELIVERABLES...INCOMPLETE.

Limitations to cover:
Anonymized features → hard domain interpretation
Severe imbalance → models biased toward majority class
Missing values imputation may introduce bias
PCA loses direct interpretability
No temporal modeling (time column ignored in most analyses)
Generalization: Dataset from one fab/line?

Deliverables:
Final notebook/report with all phases
Best model artifact (pickled)
Recommendations: Top 20–50 influential sensors, suggested monitoring, potential process improvements
Optional: Deployment sketch (real-time monitoring pipeline)

### MY PLAN: ###
**TONIGHT: FINISH PCA + BEGIN REDUNDANCY REDUCTION RESEARCH**
**Friday:** Finish redundancy section + feature selection comparison.
**Saturday:** Model baselines + metrics plots.
**Sunday:** Polish notebook, rehearse explanations (record yourself 2–3 times).
**Monday AM:** Final run-through.

**RIGHT NOW:**

**Next sub-tasks (to complete phase):**

1. Write 3–5 bullet insights from biplots/loadings (which sensors dominate PC1/PC2?)
2. Add more score plots (PC2 vs PC3, etc.)
3. Decide if PCA is final reduction or if feature selection (e.g., Boruta, loadings-based) is better for modeling

**THEN**

**BEGIN REDUNDANCY REDUCTION**

**Goal:** Tell the "story" of multicollinearity/redundancy and how to reduce it meaningfully (beyond just PCA)

**Planned steps:**
1. Correlation-based clustering of sensors (e.g., hierarchical clustering on correlation matrix)
2. Group correlated features (e.g., sensors measuring similar aspects of the same process step)
3. Loadings interpretation: Identify redundant groups from PCA (variables pointing same direction/close angles)
4. Compare PCA vs other methods: Boruta, Recursive Feature Elimination (RFE), correlation thresholding, or variance thresholding

**Domain story:** Link back to semiconductor process if possible (etch, deposition, lithography sensors often cluster)

**Deliverable:** Markdown section + visuals (dendrogram, correlation clusters, top redundant pairs)

**Why important:** PCA mixes interpretability; redundancy removal keeps original features for better explainability in manufacturing


### Conclusion & Interview Takeaways

- PCA showed high redundancy but no clear fail separation.
- Supervised selection (Boruta/RFE) reduced features to ~70–80 while maintaining or improving performance.
- SMOTE boosted recall significantly in classification.
- Isolation Forest at contamination=0.10 achieved the highest fail recall (22.6%) — validates anomaly detection for rare manufacturing defects.
- Next steps: Tune Isolation Forest further, apply SHAP for explainability on top models, and link key features back to potential process steps (e.g., etch/deposition sensors).

This project demonstrates end-to-end thinking: from cleaning → dimensionality reduction → feature selection → imbalance-aware modeling → anomaly detection.