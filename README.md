# Bit-coin-trend-Machine-Learning-Analysis
Author: Yi-Chen Kuo

Make sure to download the "BTCUSDT" dataset via:
https://www.kaggle.com/datasets/kaanxtr/btc-price-1m

## Minimal Full Pipeline

To regenerate all results in the report:

  1. Run feature engineering
  
  2. Run regression
  
  3. Run all classifiers and comparison:
  
  - In `Classification.py` set `mode = "ALL"`, then run
  - set `mode = "COMPARE"`, then run
  
  4. Run Clustering_RoleMining.py


# How to Run the Code

This project uses **four main Python scripts**.  
All scripts assume that `BTCUSDT.csv` is in the same folder.

---

## 1. Feature_Engineer.py

**Purpose:**  
Load the raw BTCUSDT data, create engineered features, and save a cached aggregated DataFrame (`agg_df_cache.parquet`).

**How to run:**


This script will:

- Read `BTCUSDT.csv`
- Create the aggregated DataFrame (`agg_df`)
- Save it to `agg_df_cache.parquet` for reuse by other scripts

You should run this **once before** the other three scripts.

---

## 2. Regression.py  (Phase II: Regression)

**Purpose:**  
Run the regression analysis (Phase II) and generate regression results and plots.

**How to run:**


This script will:

- Load `BTCUSDT.csv`
- Load or create `agg_df` using `load_or_create_agg_df` from `Feature_Engineer.py`
- Build lag features and fit an OLS regression model
- Print the regression summary and errors
- Save the regression plots (actual vs predicted)

---

## 3. Classification.py  (Phase III: Classification)

**Purpose:**  
Run the classification analysis (Phase III) with different classifiers and optionally compare them.

At the bottom of `Classification.py` you will see:


You must **change `mode` in the `main` block** to select what you want to run.

### 3.1 Run all classifiers and comparison

Set:mode = "ALL", then run

This will:

- Run all classifiers (LDA, Decision Tree, Logistic Regression, KNN, SVM, Naive Bayes, Random Forest, Ensemble, MLP)
- Save metrics for each classifier to `classification_results_<Name>.csv`
- Create the overall comparison plot `Classification_Comparison.png`
- Save the combined table to `classification_results_ALL.csv`

### 3.2 Only regenerate the comparison from existing CSVs

If you have already run all classifiers before, and only want to rebuild the comparison table and bar chart (no retraining), set:mode = "COMPARE", then run

This will:

- Load all `classification_results_*.csv`
- Print the comparison table
- Recreate `Classification_Comparison.png` and `classification_results_ALL.csv`

---

## 4. Clustering_RoleMining.py  (Phase IV: Clustering & Association)

**Purpose:**  
Run Phase IV: K-Means, DBSCAN, and Apriori association rule mining.

**How to run:**


This script will:

- Load `BTCUSDT.csv`
- Load or create `agg_df` using `load_or_create_agg_df`
- Run K-Means with different K values (elbow and silhouette)
- Run DBSCAN
- Run Apriori and print the top 10 association rules
- Save the related plots:
  - `KMeans_Elbow.png`
  - `KMeans_Silhouette.png`
  - `KMeans_Clusters.png`
  - `DBSCAN_Clusters.png`
  - `Apriori_Scatter.png`

---
