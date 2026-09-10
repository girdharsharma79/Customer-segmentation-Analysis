"""
Customer Segmentation Analysis
===============================
Project 2: Segments mall customers into groups based on income and
spending behavior using K-Means clustering, so a business can see which
customer types to target.

Pipeline:
    1. Load    - read the customer dataset (Customer ID, Age, Gender,
                  Annual Income, Spending Score)
    2. Clean   - handle missing values, drop duplicates
    3. Explore - look at the distribution of income and spending score
    4. Cluster - apply K-Means (with an elbow-method scan to pick k)
    5. Visualize & interpret - scatter plot of clusters + plain-English
       labels for each segment

Author: <your name here>
Dataset: Kaggle Mall Customer Segmentation Dataset (or the synthetic
generator in gen_customer_data.py)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RAW_CSV = "raw_customer_data.csv"
CLEAN_CSV = "clean_customer_data.csv"
CLUSTERED_CSV = "customer_segments.csv"
ELBOW_PLOT = "elbow_plot.png"
CLUSTER_PLOT = "customer_clusters.png"


# ---------------------------------------------------------------------------
# STEP 1 — LOAD
# ---------------------------------------------------------------------------
def load_data(path: str = RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# STEP 2 — CLEAN & PREPROCESS
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["CustomerID"], keep="first")

    # Fill missing income with the median (robust to outliers)
    df["Annual Income (k$)"] = df["Annual Income (k$)"].fillna(
        df["Annual Income (k$)"].median()
    )

    # Normalize gender text
    df["Gender"] = df["Gender"].str.strip().str.title()

    df = df.dropna(subset=["Age", "Spending Score (1-100)"])
    df = df.reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# STEP 3 — EXPLORE
# ---------------------------------------------------------------------------
def explore(df: pd.DataFrame):
    print("Income distribution:\n", df["Annual Income (k$)"].describe(), "\n")
    print("Spending score distribution:\n", df["Spending Score (1-100)"].describe(), "\n")
    print("Gender split:\n", df["Gender"].value_counts(), "\n")


# ---------------------------------------------------------------------------
# STEP 4 — CLUSTER (K-Means, with an elbow scan to justify k)
# ---------------------------------------------------------------------------
def find_optimal_k(X_scaled, k_range=range(2, 9)) -> list:
    """Returns inertia (within-cluster sum of squares) for each k, and
    saves an elbow plot so the choice of k is visible/justified."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(list(k_range), inertias, marker="o", color="#2F7A6E")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.title("Elbow Method for Optimal k")
    plt.tight_layout()
    plt.savefig(ELBOW_PLOT, dpi=150)
    plt.close()
    return inertias


def apply_kmeans(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    features = df[["Annual Income (k$)", "Spending Score (1-100)"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    find_optimal_k(X_scaled)

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df = df.copy()
    df["Cluster"] = km.fit_predict(X_scaled)
    return df


# ---------------------------------------------------------------------------
# STEP 5 — VISUALIZE & INTERPRET
# ---------------------------------------------------------------------------
def label_clusters(df: pd.DataFrame) -> dict:
    """Turns numeric cluster IDs into plain-English business labels based
    on each cluster's average income and spending score, measured against
    the upper/lower quartiles of the whole (clean) dataset so a genuinely
    "average" cluster is labeled as such rather than forced into a corner."""
    summary = df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean()

    income_q25, income_q75 = df["Annual Income (k$)"].quantile([0.25, 0.75])
    spend_q25, spend_q75 = df["Spending Score (1-100)"].quantile([0.25, 0.75])

    labels = {}
    for cluster_id, row in summary.iterrows():
        income, spend = row["Annual Income (k$)"], row["Spending Score (1-100)"]
        income_high = income >= income_q75
        income_low = income <= income_q25
        spend_high = spend >= spend_q75
        spend_low = spend <= spend_q25

        if income_high and spend_high:
            labels[cluster_id] = "High Income - High Spending (Target)"
        elif income_high and spend_low:
            labels[cluster_id] = "High Income - Low Spending (Careful)"
        elif income_low and spend_high:
            labels[cluster_id] = "Low Income - High Spending (Impulsive)"
        elif income_low and spend_low:
            labels[cluster_id] = "Low Income - Low Spending (Budget)"
        else:
            labels[cluster_id] = "Average Income - Average Spending"
    return labels


def visualize_clusters(df: pd.DataFrame, labels: dict):
    plt.figure(figsize=(7, 5))
    colors = plt.cm.tab10.colors
    for cluster_id in sorted(df["Cluster"].unique()):
        subset = df[df["Cluster"] == cluster_id]
        plt.scatter(
            subset["Annual Income (k$)"],
            subset["Spending Score (1-100)"],
            s=45,
            color=colors[cluster_id % len(colors)],
            label=labels[cluster_id],
            alpha=0.8,
        )
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (1-100)")
    plt.title("Customer Segments")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(CLUSTER_PLOT, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    print("STEP 1: Loading data...")
    raw_df = load_data()
    print(f"  Loaded {raw_df.shape[0]} rows")

    print("\nSTEP 2: Cleaning & preprocessing...")
    clean_df = clean_data(raw_df)
    clean_df.to_csv(CLEAN_CSV, index=False)
    print(f"  Clean dataset: {clean_df.shape[0]} rows -> {CLEAN_CSV}")

    print("\nSTEP 3: Exploring the data...")
    explore(clean_df)

    print("STEP 4: Applying K-Means clustering...")
    clustered_df = apply_kmeans(clean_df, k=5)
    print(f"  Elbow plot saved -> {ELBOW_PLOT}")

    print("\nSTEP 5: Interpreting & visualizing clusters...")
    labels = label_clusters(clustered_df)
    clustered_df["Segment"] = clustered_df["Cluster"].map(labels)
    clustered_df.to_csv(CLUSTERED_CSV, index=False)
    visualize_clusters(clustered_df, labels)
    print(f"  Cluster plot saved -> {CLUSTER_PLOT}")
    print(f"  Labeled data saved -> {CLUSTERED_CSV}")

    print("\nSegment summary:")
    summary = clustered_df.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        Avg_Income=("Annual Income (k$)", "mean"),
        Avg_Spending=("Spending Score (1-100)", "mean"),
    ).round(1)
    print(summary.to_string())


if __name__ == "__main__":
    main()
