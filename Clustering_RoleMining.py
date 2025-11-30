from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

RANDOM = 5805
np.random.seed(RANDOM)


def clustering_and_association(agg_df):
    print("\n===== Phase IV: Clustering and Association =====\n")

    # === 1. Prepare Data for Clustering ===
    features = ['volatility', 'volume', 'number_of_trades', 'taker_buy_quote_asset_volume']
    X = agg_df[features].dropna()

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Subsample for silhouette/DBSCAN (because they are slow on large data)
    sample_size = 66000
    idx = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample = X_scaled[idx]

    print(f"Using {sample_size} samples for clustering analysis...")

    # === 2. K-Means Clustering ===
    print("\n--- K-Means Analysis ---")

    wcss = []
    sil_scores = []
    k_range = range(2, 10)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=RANDOM, n_init=10)
        kmeans.fit(X_sample)
        wcss.append(kmeans.inertia_)

        labels = kmeans.predict(X_sample)
        score = silhouette_score(X_sample, labels)
        sil_scores.append(score)
        print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={score:.4f}")

    # Plot WCSS (Elbow Method)
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, wcss, marker='o')
    plt.title('K-Means: Within-Cluster Variation (Elbow Method)')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.grid(True)
    # plt.savefig("KMeans_Elbow.png")
    plt.show()

    # Plot Silhouette Scores
    plt.figure(figsize=(10, 5))
    plt.plot(k_range, sil_scores, marker='s', color='orange')
    plt.title('K-Means: Silhouette Analysis')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    # plt.savefig("KMeans_Silhouette.png")
    plt.show()

    best_k = k_range[np.argmax(sil_scores)]
    print(f"Best K based on Silhouette: {best_k}")

    # Fit Best K-Means
    best_kmeans = KMeans(n_clusters=best_k, init='k-means++', random_state=RANDOM)
    clusters = best_kmeans.fit_predict(X_sample)

    # Visualize Clusters (Volatility vs Volume)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_sample[:, 0], X_sample[:, 1], c=clusters, cmap='viridis', alpha=0.5, s=10)
    plt.title(f'K-Means Clustering (K={best_k}): Volatility vs Volume')
    plt.xlabel('Volatility (Scaled)')
    plt.ylabel('Volume (Scaled)')
    # plt.savefig("KMeans_Clusters.png")
    plt.show()

    # === 3. DBSCAN Algorithm ===
    print("\n--- DBSCAN Analysis ---")
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    db_clusters = dbscan.fit_predict(X_sample)

    n_clusters_db = len(set(db_clusters)) - (1 if -1 in db_clusters else 0)
    n_noise = list(db_clusters).count(-1)

    print(f"DBSCAN: Estimated clusters: {n_clusters_db}")
    print(f"DBSCAN: Noise points: {n_noise}")

    plt.figure(figsize=(8, 6))
    unique_labels = set(db_clusters)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]

        class_member_mask = (db_clusters == k)
        xy = X_sample[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=6, alpha=0.6)

    plt.title(f'DBSCAN Clustering (eps=0.5, min_samples=10)')
    plt.xlabel('Volatility (Scaled)')
    plt.ylabel('Volume (Scaled)')
    # plt.savefig("DBSCAN_Clusters.png")
    plt.show()

    # === 4. Apriori Algorithm ===
    print("\n--- Apriori Association Rule Mining ---")

    # Prepare categorical data for Apriori
    # 'Low', 'Medium', 'High'
    df_apriori = agg_df[features + ['price_trend']].dropna().copy()

    for col in features:
        df_apriori[col] = pd.qcut(df_apriori[col], q=3, labels=['Low', 'Medium', 'High'])

    df_onehot = pd.get_dummies(df_apriori)

    frequent_itemsets = apriori(df_onehot, min_support=0.05, use_colnames=True)

    # Generate Rules
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

    # Sort
    rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])

    print(f"\nFound {len(rules)} association rules.")
    print("\nTop 10 Rules by Lift:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).to_string())

    # rules.head(10).to_csv("Apriori_Rules.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(rules['support'], rules['confidence'], alpha=0.5, c=rules['lift'], cmap='coolwarm')
    plt.colorbar(label='Lift')
    plt.title('Association Rules: Support vs Confidence')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    # plt.savefig("Apriori_Scatter.png")
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv("BTCUSDT.csv", low_memory=False)

    from Feature_Engineer import load_or_create_agg_df
    agg_df = load_or_create_agg_df(df)

    clustering_and_association(agg_df)
