import pandas as pd
import numpy as np
import os
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = "12"
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 5)
RANDOM = 5805
CACHE_PATH = "agg_df_cache.parquet"

FEATURES = [
    'volatility',
    'volume',
    'number_of_trades',
    'taker_buy_quote_asset_volume'
]


def feature_engineering_eda(df):
    print("\n===== Phase I: Feature Engineering & EDA =====\n")

    print("Original shape:", df.shape)

    print("\nMissing value ratio per column:")
    print(df.isnull().mean())

    print("\nNumber of duplicated rows:", df.duplicated().sum())
    df = df.drop_duplicates()
    print("Shape after duplicates removal:", df.shape)

    df = df.dropna()
    print("Shape after dropping NaN:", df.shape)

    df["volatility"] = (df["high"] - df["low"]) / df["open"]
    df["price_trend"] = ((df["close"] - df["open"]).apply(
        lambda x: "up" if x > 0 else ("down" if x < 0 else "flat")))

    print("\n--- Outlier Detection ---")
    for col in ['volatility']:
        q25, q75 = np.percentile(df[col].dropna(), [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        before = df.shape[0]
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        print(f"Removed {before - df.shape[0]} outlier rows in {col}.")

    print("\n--- Correlation and Covariance Analysis ---")
    num_df = df.select_dtypes(include=[np.number])

    corr = num_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    cov = num_df.cov()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cov, annot=False, cmap='Blues')
    plt.title("Covariance Heatmap")
    plt.tight_layout()
    plt.show()

    print("\n--- Target Variable Distribution ---")
    print("Price trend distribution:\n", df['price_trend'].value_counts(normalize=True))

    if "ignore" in df.columns:
        df = df.drop(columns=["ignore"])

    return df


def vif_analysis(df):
    print("\n===== VIF Analysis for Multicollinearity =====\n")

    X = df[FEATURES].dropna()

    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    print(vif_data)

    return vif_data


def aggregation(df):
    print("\n===== Data Aggregation =====\n")

    freq_list = ['5min', '15min', '1h', '1d']
    result_dict = {}

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    for freq in freq_list:
        agg_df = df.resample(freq).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
            'volume': 'sum', 'quote_asset_volume': 'sum',
            'number_of_trades': 'sum', 'volatility': 'mean'
        }).dropna()

        X = agg_df[['open', 'high', 'low', 'volume', 'quote_asset_volume',
                    'number_of_trades', 'volatility']]
        y = agg_df['close']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        result_dict[freq] = round(mse, 3)

    best_freq = min(result_dict, key=result_dict.get)
    print(f"Best aggregation frequency: {best_freq}, MSE = {result_dict[best_freq]}")

    agg_df = df.resample('5min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last',
        'volume': 'sum', 'quote_asset_volume': 'sum',
        'number_of_trades': 'sum', 'volatility': 'mean',
        'taker_buy_base_asset_volume': 'sum', 'taker_buy_quote_asset_volume': 'sum'
    }).dropna()

    print(f"Shape after aggregation: {agg_df.shape}")

    agg_df["price_trend"] = ((agg_df["close"] - agg_df["open"]).apply(
        lambda x: "up" if x > 0 else ("down" if x < 0 else "flat")))
    agg_df["volume_bucket"] = pd.qcut(agg_df["volume"], q=3, labels=["low", "medium", "high"])

    print("\nPrice trend distribution:")
    print(agg_df['price_trend'].value_counts(normalize=True))

    return agg_df


def feature_importance_analysis(agg_df):
    print("\n===== Random Forest Feature Importance Analysis =====\n")

    X = agg_df[FEATURES].dropna()
    y = agg_df.loc[X.index, 'price_trend']

    rf = RandomForestClassifier(n_estimators=20, random_state=RANDOM, n_jobs=-1)
    rf.fit(X, y)

    importances = rf.feature_importances_
    print("Feature Importance:")
    for feat, imp in zip(FEATURES, importances):
        print(f"  {feat}: {imp:.4f}")

    plt.figure(figsize=(8, 5))
    plt.barh(FEATURES, importances, color='skyblue')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.show()

    return importances


def Elbow_Silhouette(agg_df):
    print("\n===== Elbow and Silhouette Analysis =====\n")

    df = agg_df.copy()
    X_raw = df[FEATURES].dropna()
    X = StandardScaler().fit_transform(X_raw)

    inertia = []
    K_range = range(2, 16)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertia, marker='o', color='blue')
    plt.xlabel('Number of clusters: K')
    plt.ylabel('Inertia (Sum of squared distances)')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("Elbow Method.png", dpi=300, bbox_inches='tight')
    plt.show()

    silhouette_scores = []
    sample_size = min(len(X), 5000)
    idx = np.random.choice(len(X), sample_size, replace=False)
    X_sample = X[idx]
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM)
        sample_labels = kmeans.fit_predict(X_sample)
        score = silhouette_score(X_sample, sample_labels)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, silhouette_scores, marker='o', color='green')
    plt.xlabel('Number of clusters: K')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Optimal K (Sampled)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("Silhouette Analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


def clustering(agg_df, optimal_k):
    print(f"\n===== K-Means Clustering (K={optimal_k}) =====\n")

    df = agg_df.copy()
    X_raw = df[FEATURES].dropna()
    X = StandardScaler().fit_transform(X_raw)
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM)
    clusters = kmeans.fit_predict(X)
    df.loc[X_raw.index, 'cluster'] = clusters

    print("Mean value of each feature in every cluster group:")
    print(df.groupby('cluster')[FEATURES].mean())

    return df


def classification(agg_df):
    print("\n===== Phase III: Classification Analysis =====\n")

    X = agg_df[FEATURES].dropna()
    y = agg_df.loc[X.index, 'price_trend']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False, random_state=RANDOM)

    print("--- Random Forest Classifier ---")
    rf = RandomForestClassifier(random_state=RANDOM)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_rf))
    print("Overall accuracy:", accuracy_score(y_test, y_pred_rf))

    cm_rf = confusion_matrix(y_test, y_pred_rf, labels=rf.classes_)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                xticklabels=rf.classes_, yticklabels=rf.classes_)
    plt.title("Confusion Matrix - Random Forest")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    # plt.savefig("CM_RandomForest.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("\n--- Logistic Regression Classifier ---")
    logreg = LogisticRegression(max_iter=1000, random_state=RANDOM)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_lr))
    print("Overall accuracy:", accuracy_score(y_test, y_pred_lr))

    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=logreg.classes_)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens',
                xticklabels=logreg.classes_, yticklabels=logreg.classes_)
    plt.title("Confusion Matrix - Logistic Regression")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    # plt.savefig("CM_LogisticRegression.png", dpi=300, bbox_inches='tight')
    plt.show()


def load_or_create_agg_df(df):
    if os.path.exists(CACHE_PATH):
        print(f"\nLoaded cached aggregate DataFrame from: {CACHE_PATH}")
        agg_df = pd.read_parquet(CACHE_PATH)
    else:
        agg_df = aggregation(df)
        agg_df.to_parquet(CACHE_PATH)
        print(f"\nAggregate DataFrame cached to: {CACHE_PATH}")
    return agg_df


if __name__ == '__main__':
    df = pd.read_csv("BTCUSDT.csv", low_memory=False)

    df = feature_engineering_eda(df)

    agg_df = load_or_create_agg_df(df)

    vif_analysis(agg_df)

    feature_importance_analysis(agg_df)

    print("\n===== Analysis Complete =====")
