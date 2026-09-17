"""
Store Clustering & KNN Peer Benchmarking Pipeline.

Executes unsupervised clustering (KMeans) and nearest-neighbor peer retrieval (KNN)
to segment retail apparel stores into operational archetypes and provide
comparative benchmarking for executive command center decision-making.

Features:
- Markdown Dependency: sum(promo_markdowns) / sum(net_sales)
- Sell-Through Velocity: sum(net_sales) / sum(beginning_inv_retail + purchases_retail)
- Historical Shrinkage Rate: sum(shrinkage) / sum(net_sales)
- Normalized GMROI: sum(gross_margin) / mean(ending_inv_cost)

Archetypes:
- Cluster 0: Capital-Efficient Flagships
- Cluster 1: Balanced Regional Performers
- Cluster 2: Discount-Addicted Outliers
- Cluster 3: High-Risk Shrinkage Anomalies
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

# Attempt to import scikit-learn modules; if blocked by OS security policy
# (e.g. Windows Smart App Control / WDAC on newly released Python DLLs),
# provide an exact pure NumPy drop-in implementation matching the sklearn API.
try:
    from sklearn.cluster import KMeans as _SklearnKMeans
    from sklearn.neighbors import NearestNeighbors as _SklearnNearestNeighbors
    from sklearn.preprocessing import StandardScaler as _SklearnStandardScaler
    SKLEARN_AVAILABLE = True
except (ImportError, Exception):
    SKLEARN_AVAILABLE = False


class _NumPyStandardScaler:
    """Pure NumPy StandardScaler matching sklearn API."""

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "_NumPyStandardScaler":
        data = np.asarray(X, dtype=float)
        self.mean_ = np.mean(data, axis=0)
        std = np.std(data, axis=0, ddof=0)
        self.scale_ = np.where(std == 0.0, 1.0, std)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler is not fitted.")
        data = np.asarray(X, dtype=float)
        return (data - self.mean_) / self.scale_

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.fit(X).transform(X)


class _NumPyNearestNeighbors:
    """Pure NumPy NearestNeighbors implementation matching sklearn API with Euclidean metric."""

    def __init__(self, n_neighbors: int = 4, metric: str = "euclidean") -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self._fit_X: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "_NumPyNearestNeighbors":
        self._fit_X = np.asarray(X, dtype=float)
        return self

    def kneighbors(
        self, X: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._fit_X is None:
            raise RuntimeError("NearestNeighbors is not fitted.")

        query_X = self._fit_X if X is None else np.asarray(X, dtype=float)
        # Pairwise Euclidean distances: shape (n_queries, n_samples)
        # Using broadcasting: ||A - B||
        diff = query_X[:, np.newaxis, :] - self._fit_X[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=-1)

        # Sort indices by distance
        sorted_indices = np.argsort(distances, axis=1)[:, : self.n_neighbors]
        sorted_distances = np.take_along_axis(
            distances, sorted_indices, axis=1
        )

        return sorted_distances, sorted_indices


class _NumPyKMeans:
    """Pure NumPy KMeans implementation with k-means++ seeding and Lloyd's iterations."""

    def __init__(
        self,
        n_clusters: int = 4,
        random_state: Optional[int] = 42,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
    ) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = float("inf")

    def _init_kmeans_plus_plus(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        n_samples, n_features = X.shape
        centers = np.empty((self.n_clusters, n_features), dtype=float)

        # First center uniformly at random
        first_idx = rng.integers(0, n_samples)
        centers[0] = X[first_idx]

        # Remaining centers chosen with probability proportional to D(x)^2
        for c_idx in range(1, self.n_clusters):
            dist_sq = np.min(
                [np.sum((X - centers[i]) ** 2, axis=1) for i in range(c_idx)],
                axis=0,
            )
            probs = dist_sq / np.sum(dist_sq)
            next_idx = rng.choice(n_samples, p=probs)
            centers[c_idx] = X[next_idx]

        return centers

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> "_NumPyKMeans":
        data = np.asarray(X, dtype=float)
        n_samples = data.shape[0]
        rng = np.random.default_rng(self.random_state)

        best_inertia = float("inf")
        best_centers = None
        best_labels = None

        for _ in range(self.n_init):
            centers = self._init_kmeans_plus_plus(data, rng)
            labels = np.zeros(n_samples, dtype=int)

            for _ in range(self.max_iter):
                # Assign to nearest center
                dists = np.linalg.norm(
                    data[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=-1
                )
                new_labels = np.argmin(dists, axis=1)

                # Recompute centers
                new_centers = np.empty_like(centers)
                for k in range(self.n_clusters):
                    cluster_members = data[new_labels == k]
                    if len(cluster_members) > 0:
                        new_centers[k] = np.mean(cluster_members, axis=0)
                    else:
                        new_centers[k] = data[rng.integers(0, n_samples)]

                # Check convergence
                center_shift = np.sum((new_centers - centers) ** 2)
                centers = new_centers
                labels = new_labels
                if center_shift < self.tol:
                    break

            # Calculate inertia
            inertia = float(
                np.sum([
                    np.sum((data[labels == k] - centers[k]) ** 2)
                    for k in range(self.n_clusters)
                ])
            )

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        self.fit(X)
        return self.labels_


# Expose unified classes (sklearn if available, otherwise NumPy drop-in)
if SKLEARN_AVAILABLE:
    StandardScaler = _SklearnStandardScaler
    NearestNeighbors = _SklearnNearestNeighbors
    KMeans = _SklearnKMeans
else:
    StandardScaler = _NumPyStandardScaler
    NearestNeighbors = _NumPyNearestNeighbors
    KMeans = _NumPyKMeans


# Feature names used for clustering and KNN indexing
FEATURE_COLS = [
    "markdown_dependency",
    "sell_through_velocity",
    "shrinkage_rate",
    "gmroi",
]

ARCHETYPE_LABELS = {
    0: "Capital-Efficient Flagships",
    1: "Balanced Regional Performers",
    2: "Discount-Addicted Outliers",
    3: "High-Risk Shrinkage Anomalies",
}


def load_and_aggregate_store_data(
    data_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Loads monthly RIM data (CSV or Parquet) and computes store-level aggregated metrics:
    1. Markdown Dependency: sum(promo_markdowns) / sum(net_sales)
    2. Sell-Through Velocity: sum(net_sales) / sum(beginning_inv_retail + purchases_retail)
    3. Historical Shrinkage Rate: sum(shrinkage) / sum(net_sales)
    4. Normalized GMROI: sum(gross_margin) / mean(ending_inv_cost)
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Retail dataset not found at: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format '{path.suffix}'. Use CSV or Parquet.")

    # Calculate Total Goods Available at Retail for each period if not present
    if "total_goods_retail" not in df.columns:
        df["total_goods_retail"] = (
            df["beginning_inv_retail"] + df["purchases_retail"]
        )

    # Store-level aggregation across all 24 monthly periods
    agg_df = df.groupby(["store_id", "region", "store_tier"]).agg(
        total_net_sales=("net_sales", "sum"),
        total_markdowns=("promo_markdowns", "sum"),
        total_shrinkage=("shrinkage", "sum"),
        total_goods_retail=("total_goods_retail", "sum"),
        total_gross_margin=("gross_margin", "sum"),
        mean_ending_inv_cost=("ending_inv_cost", "mean"),
    ).reset_index()

    # Feature Engineering
    agg_df["markdown_dependency"] = (
        agg_df["total_markdowns"] / agg_df["total_net_sales"]
    ).round(4)

    agg_df["sell_through_velocity"] = (
        agg_df["total_net_sales"] / agg_df["total_goods_retail"]
    ).round(4)

    agg_df["shrinkage_rate"] = (
        agg_df["total_shrinkage"] / agg_df["total_net_sales"]
    ).round(4)

    agg_df["gmroi"] = (
        agg_df["total_gross_margin"] / agg_df["mean_ending_inv_cost"]
    ).round(2)

    return agg_df


class StoreBenchmarkKNN:
    """
    Nearest Neighbors retrieval engine for operational peer-group discovery.
    Identifies the top-k most similar stores within the standardized feature space.
    """

    def __init__(self, n_neighbors: int = 4, metric: str = "euclidean") -> None:
        """
        Initialize KNN search engine.
        n_neighbors=4 retrieves the target store itself (distance 0.0) plus its top 3 peers.
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.knn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.store_features_df: Optional[pd.DataFrame] = None
        self.scaler: Optional[StandardScaler] = None
        self.scaled_features: Optional[np.ndarray] = None

    def fit(
        self,
        store_features_df: pd.DataFrame,
        scaler: StandardScaler,
    ) -> "StoreBenchmarkKNN":
        """Fit the KNN index on scaled store operational features."""
        self.store_features_df = store_features_df.copy().reset_index(drop=True)
        self.scaler = scaler
        self.scaled_features = self.scaler.transform(
            self.store_features_df[FEATURE_COLS]
        )
        self.knn_model.fit(self.scaled_features)
        return self

    def find_peer_group(self, target_store_id: str) -> Dict[str, Any]:
        """
        Finds the operational peer group for a given store:
        Returns the target store's assigned cluster, operational metrics,
        and its top 3 most similar peer locations with Euclidean distances.
        """
        if self.store_features_df is None or self.scaler is None:
            raise RuntimeError("StoreBenchmarkKNN model is not fitted.")

        store_match = self.store_features_df[
            self.store_features_df["store_id"] == target_store_id
        ]
        if store_match.empty:
            raise ValueError(f"Store ID '{target_store_id}' not found in store index.")

        target_idx = store_match.index[0]
        target_row = store_match.iloc[0]

        target_vector = self.scaled_features[target_idx].reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(target_vector)

        # Peer stores (excluding the query store itself at index 0)
        peers: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0][1:], indices[0][1:]):
            peer_row = self.store_features_df.iloc[idx]
            peers.append({
                "peer_store_id": peer_row["store_id"],
                "peer_region": peer_row["region"],
                "peer_tier": peer_row["store_tier"],
                "peer_cluster_id": int(peer_row["cluster_id"]),
                "peer_cluster_label": peer_row["cluster_label"],
                "euclidean_distance": float(round(dist, 4)),
                "markdown_dependency": float(peer_row["markdown_dependency"]),
                "sell_through_velocity": float(peer_row["sell_through_velocity"]),
                "shrinkage_rate": float(peer_row["shrinkage_rate"]),
                "gmroi": float(peer_row["gmroi"]),
            })

        return {
            "target_store_id": target_store_id,
            "region": target_row["region"],
            "store_tier": target_row["store_tier"],
            "cluster_id": int(target_row["cluster_id"]),
            "cluster_label": target_row["cluster_label"],
            "metrics": {
                "markdown_dependency": float(target_row["markdown_dependency"]),
                "sell_through_velocity": float(target_row["sell_through_velocity"]),
                "shrinkage_rate": float(target_row["shrinkage_rate"]),
                "gmroi": float(target_row["gmroi"]),
            },
            "nearest_peers": peers,
        }


def train_clustering_pipeline(
    store_df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, StandardScaler, KMeans, StoreBenchmarkKNN]:
    """
    Executes the complete clustering pipeline:
    1. Standardizes operational features.
    2. Fits KMeans (k=4).
    3. Maps clusters deterministically to the 4 business archetypes:
       - Cluster 0: Capital-Efficient Flagships
       - Cluster 1: Balanced Regional Performers
       - Cluster 2: Discount-Addicted Outliers
       - Cluster 3: High-Risk Shrinkage Anomalies
    4. Trains the KNN index for peer retrieval.
    """
    df = store_df.copy()

    # 1. Feature Standardization
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(df[FEATURE_COLS])

    # 2. KMeans Clustering
    kmeans = KMeans(n_clusters=4, random_state=random_state, n_init=10)
    raw_clusters = kmeans.fit_predict(scaled_matrix)
    df["raw_cluster"] = raw_clusters

    # 3. Deterministic Cluster Mapping to Business Archetypes
    # Compute centroids in original feature space to classify clusters reliably
    centroids = df.groupby("raw_cluster")[FEATURE_COLS].mean()

    # Cluster with highest markdown rate -> Cluster 2: Discount-Addicted Outliers
    discount_cluster = int(centroids["markdown_dependency"].idxmax())

    # Cluster with highest shrinkage rate -> Cluster 3: High-Risk Shrinkage Anomalies
    shrink_cluster = int(centroids["shrinkage_rate"].idxmax())

    # Remaining clusters
    remaining = [c for c in centroids.index if c not in (discount_cluster, shrink_cluster)]

    # Out of remaining, highest GMROI / sell-through -> Cluster 0: Capital-Efficient Flagships
    flagship_cluster = int(centroids.loc[remaining, "gmroi"].idxmax())

    # Remainder -> Cluster 1: Balanced Regional Performers
    balanced_cluster = [c for c in remaining if c != flagship_cluster][0]

    # Map raw clusters to standardized 0..3 IDs
    raw_to_archetype_id = {
        flagship_cluster: 0,
        balanced_cluster: 1,
        discount_cluster: 2,
        shrink_cluster: 3,
    }

    df["cluster_id"] = df["raw_cluster"].map(raw_to_archetype_id)
    df["cluster_label"] = df["cluster_id"].map(ARCHETYPE_LABELS)
    df = df.drop(columns=["raw_cluster"])

    # 4. Train KNN Benchmark Engine
    knn_engine = StoreBenchmarkKNN(n_neighbors=4)
    knn_engine.fit(df, scaler)

    return df, scaler, kmeans, knn_engine


def save_pipeline(
    scaler: StandardScaler,
    kmeans: KMeans,
    knn_engine: StoreBenchmarkKNN,
    store_summary_df: pd.DataFrame,
    output_dir: Union[str, Path] = "models",
) -> Path:
    """Serializes the fitted pipeline bundle into a joblib artifact."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    bundle_path = out_path / "store_cluster_pipeline.joblib"

    pipeline_bundle = {
        "scaler": scaler,
        "kmeans": kmeans,
        "knn_engine": knn_engine,
        "store_summary_df": store_summary_df,
        "feature_cols": FEATURE_COLS,
        "archetype_labels": ARCHETYPE_LABELS,
    }

    joblib.dump(pipeline_bundle, bundle_path)
    return bundle_path


def load_pipeline(bundle_path: Union[str, Path] = "models/store_cluster_pipeline.joblib") -> Dict[str, Any]:
    """Loads a previously serialized pipeline bundle."""
    path = Path(bundle_path)
    if not path.exists():
        raise FileNotFoundError(f"Serialized model pipeline not found at {path}")
    return joblib.load(path)


def export_store_summary(
    df: pd.DataFrame,
    output_csv_path: Union[str, Path] = "store_clusters_summary.csv",
) -> Path:
    """Exports the store-level clustering summary table to CSV."""
    out_path = Path(output_csv_path)
    export_cols = [
        "store_id",
        "region",
        "store_tier",
        "cluster_id",
        "cluster_label",
        "markdown_dependency",
        "sell_through_velocity",
        "shrinkage_rate",
        "gmroi",
    ]
    df[export_cols].to_csv(out_path, index=False)
    return out_path


def print_executive_clustering_report(
    store_df: pd.DataFrame,
    knn_engine: StoreBenchmarkKNN,
    target_sample_store: str = "STORE_104",
) -> None:
    """Displays an executive ASCII summary table of cluster distribution, centroids, and KNN lookup."""
    print("=" * 80)
    print("      APPAREL PRE-SALES COMMAND CENTER: STORE CLUSTERING & FORENSICS")
    print("=" * 80)

    # Cluster Distribution
    print("OPERATIONAL CLUSTER DISTRIBUTION (50 STORE FLEET):")
    counts = store_df.groupby(["cluster_id", "cluster_label"]).size().reset_index(name="store_count")
    for _, row in counts.iterrows():
        print(f"  * Cluster {row['cluster_id']}: {row['cluster_label']:<32} | {row['store_count']:>2} stores")
    print("-" * 80)

    # Mean Metrics by Cluster
    print("MEAN OPERATIONAL BENCHMARKS BY CLUSTER ARCHETYPE:")
    cluster_means = store_df.groupby(["cluster_id", "cluster_label"]).agg(
        store_count=("store_id", "count"),
        markdown_dep=("markdown_dependency", "mean"),
        sell_through=("sell_through_velocity", "mean"),
        shrink_rate=("shrinkage_rate", "mean"),
        mean_gmroi=("gmroi", "mean"),
    ).reset_index()

    print(cluster_means.to_string(
        index=False,
        formatters={
            "markdown_dep": "{:.1%}".format,
            "sell_through": "{:.3f}x".format,
            "shrink_rate": "{:.2%}".format,
            "mean_gmroi": "{:.2f}x".format,
        },
    ))
    print("-" * 80)

    # Sample KNN Peer Lookup
    print(f"PRE-SALES PEER GROUP LOOKUP BENCHMARK: [{target_sample_store}]")
    try:
        lookup = knn_engine.find_peer_group(target_sample_store)
        m = lookup["metrics"]
        print(f"Target Store: {lookup['target_store_id']} ({lookup['region']} - {lookup['store_tier']})")
        print(f"Archetype   : Cluster {lookup['cluster_id']} - {lookup['cluster_label']}")
        print(f"Metrics     : Markdown Dep: {m['markdown_dependency']:.1%} | Sell-Through: {m['sell_through_velocity']:.3f}x | Shrink: {m['shrinkage_rate']:.2%} | GMROI: {m['gmroi']:.2f}x")
        print("\nTop-3 Closest Operational Peers:")
        peer_df = pd.DataFrame(lookup["nearest_peers"])
        print(peer_df[[
            "peer_store_id", "peer_region", "peer_tier", "peer_cluster_label",
            "euclidean_distance", "markdown_dependency", "sell_through_velocity", "shrinkage_rate", "gmroi"
        ]].to_string(
            index=False,
            formatters={
                "euclidean_distance": "{:.4f}".format,
                "markdown_dependency": "{:.1%}".format,
                "sell_through_velocity": "{:.3f}x".format,
                "shrinkage_rate": "{:.2%}".format,
                "gmroi": "{:.2f}x".format,
            }
        ))
    except Exception as e:
        print(f"Could not perform lookup for {target_sample_store}: {e}")
    print("=" * 80)


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    data_csv = root_dir / "synthetic_apparel_rim_data.csv"
    data_parquet = root_dir / "synthetic_apparel_rim_data.parquet"

    # Prefer parquet if available, otherwise CSV
    data_file = data_parquet if data_parquet.exists() else data_csv

    print(f"Aggregating 24-month multi-period store metrics from {data_file.name}...")
    store_agg = load_and_aggregate_store_data(data_file)
    print(f"[OK] Aggregated operational metrics for {len(store_agg)} store locations.")

    print("Training KMeans clustering (k=4) and KNN peer benchmarking engine...")
    store_clustered, scaler, kmeans, knn_engine = train_clustering_pipeline(store_agg, random_state=42)
    print("[OK] Clustering and archetype mapping complete.")

    # Export summary CSV
    summary_path = export_store_summary(store_clustered, root_dir / "store_clusters_summary.csv")
    print(f"[OK] Exported store cluster summary: {summary_path.name}")

    # Serialize pipeline
    bundle_path = save_pipeline(scaler, kmeans, knn_engine, store_clustered, root_dir / "models")
    print(f"[OK] Serialized model pipeline bundle: {bundle_path.relative_to(root_dir)}")

    # Print executive report
    print_executive_clustering_report(store_clustered, knn_engine, target_sample_store="STORE_104")


if __name__ == "__main__":
    main()
