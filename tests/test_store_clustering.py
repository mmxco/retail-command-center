"""
Unit tests for the Store Clustering and KNN Peer Benchmarking Pipeline.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from store_clustering import (
    ARCHETYPE_LABELS,
    FEATURE_COLS,
    StoreBenchmarkKNN,
    load_and_aggregate_store_data,
    load_pipeline,
    save_pipeline,
    train_clustering_pipeline,
)


class TestStoreClusteringPipeline(unittest.TestCase):
    """Test suite for feature aggregation, KMeans clustering, and KNN benchmarking."""

    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.data_file = cls.root_dir / "synthetic_apparel_rim_data.parquet"
        if not cls.data_file.exists():
            cls.data_file = cls.root_dir / "synthetic_apparel_rim_data.csv"

        cls.store_agg = load_and_aggregate_store_data(cls.data_file)
        (
            cls.store_clustered,
            cls.scaler,
            cls.kmeans,
            cls.knn_engine,
        ) = train_clustering_pipeline(cls.store_agg, random_state=42)

    def test_feature_aggregation_shape_and_columns(self):
        """Validates that aggregation correctly yields 50 stores with proper metrics."""
        self.assertEqual(len(self.store_agg), 50)
        expected_cols = [
            "store_id",
            "region",
            "store_tier",
            "markdown_dependency",
            "sell_through_velocity",
            "shrinkage_rate",
            "gmroi",
        ]
        for col in expected_cols:
            self.assertIn(col, self.store_agg.columns)

        # Check values are strictly positive and finite
        self.assertTrue((self.store_agg["markdown_dependency"] > 0).all())
        self.assertTrue((self.store_agg["sell_through_velocity"] > 0).all())
        self.assertTrue((self.store_agg["shrinkage_rate"] > 0).all())
        self.assertTrue((self.store_agg["gmroi"] > 0).all())

    def test_clustering_archetypes_and_distribution(self):
        """Validates that KMeans partitions into the 4 defined archetypes matching injected data."""
        cluster_counts = self.store_clustered["cluster_id"].value_counts().to_dict()

        # All 4 clusters present
        self.assertEqual(len(cluster_counts), 4)
        for cluster_id in range(4):
            self.assertIn(cluster_id, cluster_counts)

        # Check archetype labels
        for _, row in self.store_clustered.iterrows():
            self.assertEqual(
                row["cluster_label"], ARCHETYPE_LABELS[row["cluster_id"]]
            )

        # Verify cluster behavioral characteristics:
        centroids = self.store_clustered.groupby("cluster_id")[FEATURE_COLS].mean()

        # Cluster 0 (Flagships): Highest GMROI, lowest markdowns
        self.assertGreater(centroids.loc[0, "gmroi"], centroids.loc[1, "gmroi"])
        self.assertLess(centroids.loc[0, "markdown_dependency"], 0.12)

        # Cluster 2 (Discount-Addicted): Markdowns > 35%
        self.assertGreater(centroids.loc[2, "markdown_dependency"], 0.35)

        # Cluster 3 (Shrinkage Anomalies): Shrinkage > 4.0%
        self.assertGreater(centroids.loc[3, "shrinkage_rate"], 0.04)

        # Cluster counts match ground truth injected in Phase 3
        # 5 Flagships, 4 Discount, 3 Shrink, 38 Normal
        self.assertEqual(cluster_counts[0], 5)
        self.assertEqual(cluster_counts[1], 38)
        self.assertEqual(cluster_counts[2], 4)
        self.assertEqual(cluster_counts[3], 3)

    def test_knn_peer_lookup(self):
        """Tests KNN peer group retrieval for target store STORE_104."""
        target_id = "STORE_104"
        lookup = self.knn_engine.find_peer_group(target_id)

        self.assertEqual(lookup["target_store_id"], target_id)
        self.assertIn("cluster_id", lookup)
        self.assertIn("cluster_label", lookup)
        self.assertIn("metrics", lookup)

        peers = lookup["nearest_peers"]
        self.assertEqual(len(peers), 3)

        # Verify peers do not include target store
        peer_ids = [p["peer_store_id"] for p in peers]
        self.assertNotIn(target_id, peer_ids)

        # Verify distances are sorted ascending and non-negative
        distances = [p["euclidean_distance"] for p in peers]
        self.assertTrue(all(d >= 0 for d in distances))
        self.assertEqual(distances, sorted(distances))

    def test_knn_invalid_store_error(self):
        """Tests that querying a nonexistent store raises ValueError."""
        with self.assertRaises(ValueError):
            self.knn_engine.find_peer_group("NONEXISTENT_STORE")

    def test_model_serialization_and_deserialization(self):
        """Tests that pipeline can be serialized to disk and loaded with full inference capability."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = save_pipeline(
                self.scaler,
                self.kmeans,
                self.knn_engine,
                self.store_clustered,
                output_dir=tmpdir,
            )
            self.assertTrue(bundle_path.exists())

            loaded = load_pipeline(bundle_path)
            self.assertIn("scaler", loaded)
            self.assertIn("kmeans", loaded)
            self.assertIn("knn_engine", loaded)
            self.assertIn("store_summary_df", loaded)

            loaded_knn = loaded["knn_engine"]
            res = loaded_knn.find_peer_group("STORE_101")
            self.assertEqual(res["target_store_id"], "STORE_101")
            self.assertEqual(len(res["nearest_peers"]), 3)


if __name__ == "__main__":
    unittest.main()
