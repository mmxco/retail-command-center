"""
Unit tests for the Retail Inventory Method (RIM) Calculation Engine and Forensics Module.
"""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from rim_engine import RIMEngine, compute_period_rim


class TestRIMEngineCalculations(unittest.TestCase):
    """Tests deterministic single-period and edge case RIM equations."""

    def test_compute_period_rim_textbook(self):
        """
        Validates exact textbook RIM numbers:
        Beginning: Cost 100,000, Retail 200,000
        Purchases: Cost 50,000, Retail 100,000
        Net Markups: 0
        Goods Available: Cost 150,000, Retail 300,000 -> Cost-to-Retail Ratio = 0.50 (50%)
        Net Sales: 120,000
        Markdowns: 20,000
        Shrinkage: 5,000
        Total Reductions: 145,000
        Ending Inventory Retail: 300,000 - 145,000 = 155,000
        Ending Inventory Cost: 155,000 * 0.50 = 77,500
        Gross COGS: 150,000 - 77,500 = 72,500
        Gross Margin $: 120,000 - 72,500 = 47,500
        Gross Margin %: 47,500 / 120,000 = 39.5833%
        Average Inventory Cost: (100,000 + 77,500) / 2 = 88,750
        GMROI: 47,500 / 88,750 = 0.5352
        """
        result = compute_period_rim(
            beg_cost=100_000.0,
            beg_retail=200_000.0,
            purchases_cost=50_000.0,
            purchases_retail=100_000.0,
            net_sales=120_000.0,
            markdowns=20_000.0,
            shrinkage=5_000.0,
            net_markups=0.0,
        )

        self.assertAlmostEqual(result["goods_avail_cost"], 150_000.0, places=2)
        self.assertAlmostEqual(result["goods_avail_retail"], 300_000.0, places=2)
        self.assertAlmostEqual(result["cost_to_retail_ratio"], 0.50, places=4)
        self.assertAlmostEqual(result["total_reductions"], 145_000.0, places=2)
        self.assertAlmostEqual(result["ending_inv_retail"], 155_000.0, places=2)
        self.assertAlmostEqual(result["ending_inv_cost"], 77_500.0, places=2)
        self.assertAlmostEqual(result["gross_cogs"], 72_500.0, places=2)
        self.assertAlmostEqual(result["gross_margin"], 47_500.0, places=2)
        self.assertAlmostEqual(result["gross_margin_pct"], 39.5833, places=2)
        self.assertAlmostEqual(result["avg_inv_cost"], 88_750.0, places=2)
        self.assertAlmostEqual(result["gmroi"], 0.5352, places=4)
        self.assertAlmostEqual(result["markdown_pct_sales"], 16.6667, places=2)
        self.assertAlmostEqual(result["shrink_pct_sales"], 4.1667, places=2)

    def test_with_net_markups(self):
        """Verifies that upward net markups properly expand Goods Available Retail."""
        result = compute_period_rim(
            beg_cost=40_000.0,
            beg_retail=90_000.0,
            purchases_cost=20_000.0,
            purchases_retail=45_000.0,
            net_sales=50_000.0,
            markdowns=5_000.0,
            shrinkage=1_000.0,
            net_markups=5_000.0,  # 90k + 45k + 5k = 140k
        )
        self.assertEqual(result["goods_avail_retail"], 140_000.0)
        self.assertEqual(result["goods_avail_cost"], 60_000.0)
        expected_ratio = 60_000.0 / 140_000.0
        self.assertAlmostEqual(result["cost_to_retail_ratio"], expected_ratio, places=5)

    def test_zero_division_guards(self):
        """Validates zero division behavior when inventory or sales are zero."""
        # 0 goods available retail
        res1 = compute_period_rim(
            beg_cost=0.0,
            beg_retail=0.0,
            purchases_cost=0.0,
            purchases_retail=0.0,
            net_sales=0.0,
            markdowns=0.0,
            shrinkage=0.0,
        )
        self.assertEqual(res1["cost_to_retail_ratio"], 0.0)
        self.assertEqual(res1["ending_inv_cost"], 0.0)
        self.assertEqual(res1["gross_margin_pct"], 0.0)
        self.assertEqual(res1["gmroi"], 0.0)

        # 0 sales but positive inventory
        res2 = compute_period_rim(
            beg_cost=50_000.0,
            beg_retail=100_000.0,
            purchases_cost=0.0,
            purchases_retail=0.0,
            net_sales=0.0,
            markdowns=0.0,
            shrinkage=0.0,
        )
        self.assertEqual(res2["gross_margin_pct"], 0.0)
        self.assertEqual(res2["markdown_pct_sales"], 0.0)

    def test_negative_retail_stockout_floor(self):
        """Ensures that if reductions exceed goods available, ending cost is floored at 0.0."""
        result = compute_period_rim(
            beg_cost=10_000.0,
            beg_retail=20_000.0,
            purchases_cost=0.0,
            purchases_retail=0.0,
            net_sales=25_000.0,  # exceeds retail available (20k)
            markdowns=2_000.0,
            shrinkage=1_000.0,
        )
        self.assertEqual(result["ending_inv_retail"], -8_000.0)
        self.assertEqual(result["ending_inv_cost"], 0.0)  # Max(0.0, ...) floor enforced


class TestRIMEnginePipeline(unittest.TestCase):
    """Tests the high-level RIMEngine class methods and multi-period roll-forwards."""

    def setUp(self):
        self.engine = RIMEngine(tolerance=0.02)
        self.parquet_path = (
            Path(__file__).resolve().parent.parent / "synthetic_apparel_rim_data.parquet"
        )

    def test_rollforward_store_continuity(self):
        """Validates that rollforward_store strictly maintains Month T Ending == Month T+1 Beginning."""
        activity_data = pd.DataFrame({
            "period_date": ["2025-01-01", "2025-02-01", "2025-03-01"],
            "purchases_cost": [40_000.0, 50_000.0, 45_000.0],
            "purchases_retail": [100_000.0, 120_000.0, 110_000.0],
            "net_sales": [60_000.0, 70_000.0, 80_000.0],
            "promo_markdowns": [8_000.0, 10_000.0, 9_000.0],
            "shrinkage": [1_200.0, 1_400.0, 1_600.0],
        })

        rollforward_df = self.engine.rollforward_store(
            store_activity_df=activity_data,
            initial_cost=80_000.0,
            initial_retail=200_000.0,
        )

        self.assertEqual(len(rollforward_df), 3)

        # Check Month 0 Ending == Month 1 Beginning
        self.assertAlmostEqual(
            rollforward_df.loc[0, "ending_inv_cost"],
            rollforward_df.loc[1, "beginning_inv_cost"],
            places=2,
        )
        self.assertAlmostEqual(
            rollforward_df.loc[0, "ending_inv_retail"],
            rollforward_df.loc[1, "beginning_inv_retail"],
            places=2,
        )

        # Check Month 1 Ending == Month 2 Beginning
        self.assertAlmostEqual(
            rollforward_df.loc[1, "ending_inv_cost"],
            rollforward_df.loc[2, "beginning_inv_cost"],
            places=2,
        )
        self.assertAlmostEqual(
            rollforward_df.loc[1, "ending_inv_retail"],
            rollforward_df.loc[2, "beginning_inv_retail"],
            places=2,
        )

    def test_audit_dataset_full_reconciliation(self):
        """Audits the synthetic apparel dataset to ensure 100% mathematical integrity."""
        if not self.parquet_path.exists():
            self.skipTest(f"Dataset {self.parquet_path} not found.")

        df = pd.read_parquet(self.parquet_path)
        audited_df, summary = self.engine.audit_dataframe(df)

        self.assertEqual(summary["total_rows_audited"], 1200)
        self.assertEqual(summary["temporal_rollforward_breaks_count"], 0)
        for delta_name, count in summary["discrepancies"].items():
            self.assertEqual(
                count, 0, f"Discrepancies found in {delta_name}: {count}"
            )
        self.assertTrue(summary["passed"])

    def test_anomaly_forensics_classification(self):
        """Verifies that the forensic diagnostic module correctly classifies store anomalies."""
        if not self.parquet_path.exists():
            self.skipTest(f"Dataset {self.parquet_path} not found.")

        df = pd.read_parquet(self.parquet_path)
        diagnostics = self.engine.diagnose_anomalies(df)

        counts = diagnostics["detected_profile"].value_counts().to_dict()
        self.assertEqual(counts.get("Discount-Addicted", 0), 4)
        self.assertEqual(counts.get("Shrinkage Anomaly", 0), 3)
        self.assertEqual(counts.get("High-Turn Flagship", 0), 5)
        self.assertEqual(counts.get("Normal", 0), 38)


if __name__ == "__main__":
    unittest.main()
