"""
tests/test_empirical_stress.py
Empirical stress-testing harness for core domain calculation engines.
Tests:
1. Pure Domain Isolation (Headless execution and zero Streamlit dependencies)
2. RIM Accounting Stress Tests (Edge cases, zero divisions, negative retail floors, Monte Carlo)
3. Clustering and KNN Stress Tests (STORE_104, boundary stores, unknown store IDs, joblib serialization parity)
4. Value Realization Stress Tests (Boundary fleet sizes, extreme parameters, financial integrity invariants)
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from core.rim_engine import RIMEngine, compute_period_rim
from core.store_clustering import (
    ARCHETYPE_LABELS,
    FEATURE_COLS,
    StoreBenchmarkKNN,
    load_and_aggregate_store_data,
    load_pipeline,
    save_pipeline,
    train_clustering_pipeline,
)
from core.value_engine import (
    ValueCalculatorInputs,
    ValueRealizationOutput,
    calculate_value_realization,
)


class TestPureDomainIsolation(unittest.TestCase):
    """Verifies core calculation engines can be imported and executed headlessly without Streamlit."""

    def test_headless_import_isolation_in_isolated_process(self):
        """Executes a pristine python subprocess to assert Streamlit is never imported by core modules."""
        python_exe = sys.executable
        isolation_script = (
            "import sys; "
            "import core.rim_engine; "
            "import core.store_clustering; "
            "import core.value_engine; "
            "assert 'streamlit' not in sys.modules, f'Streamlit leaked into sys.modules: {sys.modules.get(\"streamlit\")}'; "
            "print('ISOLATION_CONFIRMED')"
        )
        proc = subprocess.run(
            [python_exe, "-c", isolation_script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"Pure domain isolation failed. Stdout: {proc.stdout}, Stderr: {proc.stderr}",
        )
        self.assertIn("ISOLATION_CONFIRMED", proc.stdout)

    def test_headless_execution_in_isolated_process(self):
        """Executes calculation functions in a pristine subprocess without Streamlit."""
        python_exe = sys.executable
        execution_script = (
            "import sys; "
            "from core.rim_engine import compute_period_rim; "
            "from core.value_engine import calculate_value_realization, ValueCalculatorInputs; "
            "res_rim = compute_period_rim(100, 200, 50, 100, 120, 20, 5, 0); "
            "assert res_rim['ending_inv_cost'] == 77.5; "
            "res_val = calculate_value_realization(ValueCalculatorInputs(store_count=10)); "
            "assert res_val.total_fleet_sales == 32000000.0; "
            "assert 'streamlit' not in sys.modules; "
            "print('HEADLESS_EXECUTION_CONFIRMED')"
        )
        proc = subprocess.run(
            [python_exe, "-c", execution_script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"Headless execution failed. Stdout: {proc.stdout}, Stderr: {proc.stderr}",
        )
        self.assertIn("HEADLESS_EXECUTION_CONFIRMED", proc.stdout)


class TestRIMAccountingStress(unittest.TestCase):
    """Rigorous stress tests for RIM accounting formulas, zero divisions, and boundary conditions."""

    def setUp(self):
        self.engine = RIMEngine(tolerance=0.02)

    def test_zero_goods_available(self):
        """Zero goods available for sale (0 cost, 0 retail)."""
        res = compute_period_rim(
            beg_cost=0.0,
            beg_retail=0.0,
            purchases_cost=0.0,
            purchases_retail=0.0,
            net_sales=0.0,
            markdowns=0.0,
            shrinkage=0.0,
            net_markups=0.0,
        )
        self.assertEqual(res["goods_avail_cost"], 0.0)
        self.assertEqual(res["goods_avail_retail"], 0.0)
        self.assertEqual(res["cost_to_retail_ratio"], 0.0)
        self.assertEqual(res["ending_inv_retail"], 0.0)
        self.assertEqual(res["ending_inv_cost"], 0.0)
        self.assertEqual(res["gross_cogs"], 0.0)
        self.assertEqual(res["gross_margin"], 0.0)
        self.assertEqual(res["gross_margin_pct"], 0.0)
        self.assertEqual(res["gmroi"], 0.0)
        self.assertEqual(res["markdown_pct_sales"], 0.0)
        self.assertEqual(res["shrink_pct_sales"], 0.0)

    def test_zero_sales_positive_inventory(self):
        """Zero sales with positive inventory roll-forward."""
        res = compute_period_rim(
            beg_cost=60_000.0,
            beg_retail=120_000.0,
            purchases_cost=40_000.0,
            purchases_retail=80_000.0,
            net_sales=0.0,
            markdowns=0.0,
            shrinkage=0.0,
            net_markups=0.0,
        )
        self.assertEqual(res["goods_avail_cost"], 100_000.0)
        self.assertEqual(res["goods_avail_retail"], 200_000.0)
        self.assertEqual(res["cost_to_retail_ratio"], 0.5)
        self.assertEqual(res["total_reductions"], 0.0)
        self.assertEqual(res["ending_inv_retail"], 200_000.0)
        self.assertEqual(res["ending_inv_cost"], 100_000.0)
        self.assertEqual(res["gross_cogs"], 0.0)
        self.assertEqual(res["gross_margin"], 0.0)
        self.assertEqual(res["gross_margin_pct"], 0.0)
        self.assertEqual(res["gmroi"], 0.0)
        self.assertEqual(res["markdown_pct_sales"], 0.0)
        self.assertEqual(res["shrink_pct_sales"], 0.0)

    def test_negative_retail_stockout_floor(self):
        """Acute stockouts where total reductions exceed goods available retail."""
        res = compute_period_rim(
            beg_cost=20_000.0,
            beg_retail=40_000.0,
            purchases_cost=10_000.0,
            purchases_retail=20_000.0,
            net_sales=70_000.0,
            markdowns=10_000.0,
            shrinkage=5_000.0,
            net_markups=0.0,
        )
        self.assertEqual(res["goods_avail_cost"], 30_000.0)
        self.assertEqual(res["goods_avail_retail"], 60_000.0)
        self.assertEqual(res["cost_to_retail_ratio"], 0.5)
        self.assertEqual(res["total_reductions"], 85_000.0)
        self.assertEqual(res["ending_inv_retail"], -25_000.0)
        # CRITICAL ACCOUNTING INVARIANT: Cost basis non-negativity floor
        self.assertGreaterEqual(res["ending_inv_cost"], 0.0)
        self.assertEqual(res["ending_inv_cost"], 0.0)
        # COGS equals full goods available cost since ending inventory is 0
        self.assertEqual(res["gross_cogs"], 30_000.0)
        # Gross margin = 70k - 30k = 40k
        self.assertEqual(res["gross_margin"], 40_000.0)

    def test_massive_upward_markups(self):
        """Extreme upward price revisions (markups) deflating cost complement."""
        res = compute_period_rim(
            beg_cost=50_000.0,
            beg_retail=100_000.0,
            purchases_cost=50_000.0,
            purchases_retail=100_000.0,
            net_sales=80_000.0,
            markdowns=5_000.0,
            shrinkage=2_000.0,
            net_markups=800_000.0,
        )
        self.assertEqual(res["goods_avail_cost"], 100_000.0)
        self.assertEqual(res["goods_avail_retail"], 1_000_000.0)
        self.assertAlmostEqual(res["cost_to_retail_ratio"], 0.10, places=4)
        self.assertEqual(res["total_reductions"], 87_000.0)
        self.assertEqual(res["ending_inv_retail"], 913_000.0)
        self.assertAlmostEqual(res["ending_inv_cost"], 91_300.0, places=2)
        self.assertAlmostEqual(res["gross_cogs"], 8_700.0, places=2)
        self.assertGreaterEqual(res["ending_inv_cost"], 0.0)

    def test_extreme_markdown_shocks(self):
        """Extreme markdown shocks where markdowns exceed net sales or goods retail."""
        res_shock = compute_period_rim(
            beg_cost=100_000.0,
            beg_retail=200_000.0,
            purchases_cost=50_000.0,
            purchases_retail=100_000.0,
            net_sales=100_000.0,
            markdowns=250_000.0,
            shrinkage=5_000.0,
        )
        self.assertEqual(res_shock["goods_avail_retail"], 300_000.0)
        self.assertEqual(res_shock["total_reductions"], 355_000.0)
        self.assertEqual(res_shock["ending_inv_retail"], -55_000.0)
        self.assertEqual(res_shock["ending_inv_cost"], 0.0)
        self.assertEqual(res_shock["gross_cogs"], 150_000.0)
        self.assertEqual(res_shock["gross_margin"], -50_000.0)
        self.assertEqual(res_shock["gross_margin_pct"], -50.0)
        self.assertGreaterEqual(res_shock["ending_inv_cost"], 0.0)

    def test_monte_carlo_cost_non_negativity_invariants(self):
        """Generates 1,000 randomized stress combinations; verifies ending cost >= 0 and finite metrics."""
        rng = np.random.default_rng(2026)
        n_trials = 1000

        beg_costs = rng.uniform(0.0, 500_000.0, n_trials)
        beg_retails = beg_costs * rng.uniform(1.0, 4.0, n_trials)
        purch_costs = rng.uniform(0.0, 500_000.0, n_trials)
        purch_retails = purch_costs * rng.uniform(1.0, 4.0, n_trials)
        net_sales = rng.uniform(0.0, 1_500_000.0, n_trials)
        markdowns = rng.uniform(0.0, 800_000.0, n_trials)
        shrinkages = rng.uniform(0.0, 50_000.0, n_trials)
        net_markups = rng.uniform(0.0, 200_000.0, n_trials)

        for i in range(n_trials):
            res = compute_period_rim(
                beg_cost=float(beg_costs[i]),
                beg_retail=float(beg_retails[i]),
                purchases_cost=float(purch_costs[i]),
                purchases_retail=float(purch_retails[i]),
                net_sales=float(net_sales[i]),
                markdowns=float(markdowns[i]),
                shrinkage=float(shrinkages[i]),
                net_markups=float(net_markups[i]),
            )
            self.assertGreaterEqual(
                res["ending_inv_cost"],
                0.0,
                f"Ending cost basis violated non-negativity floor on trial {i}",
            )
            self.assertGreaterEqual(
                res["cost_to_retail_ratio"],
                0.0,
                f"Cost to retail ratio negative on trial {i}",
            )
            self.assertFalse(
                np.isnan(res["ending_inv_cost"]),
                f"NaN detected in ending_inv_cost on trial {i}",
            )
            self.assertFalse(
                np.isinf(res["ending_inv_cost"]),
                f"Inf detected in ending_inv_cost on trial {i}",
            )

    def test_vectorized_dataframe_edge_cases(self):
        """Verifies RIMEngine.evaluate_dataframe handles edge rows gracefully."""
        test_df = pd.DataFrame({
            "beginning_inv_cost": [0.0, 50_000.0, 10_000.0],
            "beginning_inv_retail": [0.0, 100_000.0, 20_000.0],
            "purchases_cost": [0.0, 0.0, 5_000.0],
            "purchases_retail": [0.0, 0.0, 10_000.0],
            "net_sales": [0.0, 0.0, 50_000.0],
            "promo_markdowns": [0.0, 0.0, 10_000.0],
            "shrinkage": [0.0, 0.0, 2_000.0],
            "net_markups": [0.0, 5_000.0, 0.0],
        })
        evaluated = self.engine.evaluate_dataframe(test_df)
        self.assertEqual(len(evaluated), 3)
        self.assertEqual(evaluated.loc[0, "calc_ending_inv_cost"], 0.0)
        self.assertEqual(evaluated.loc[1, "calc_ending_inv_cost"], 50_000.0)
        self.assertEqual(evaluated.loc[2, "calc_ending_inv_cost"], 0.0)
        self.assertTrue((evaluated["calc_ending_inv_cost"] >= 0.0).all())


class TestStoreClusteringStress(unittest.TestCase):
    """Rigorous stress tests for KMeans clustering, KNN retrieval, and model serialization parity."""

    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent
        data_parquet = cls.root_dir / "synthetic_apparel_rim_data.parquet"
        data_csv = cls.root_dir / "synthetic_apparel_rim_data.csv"
        data_path = data_parquet if data_parquet.exists() else data_csv
        cls.store_df = load_and_aggregate_store_data(data_path)
        cls.clustered_df, cls.scaler, cls.kmeans, cls.knn_engine = (
            train_clustering_pipeline(cls.store_df, random_state=42)
        )

    def test_target_store_104_peer_retrieval(self):
        """Tests StoreBenchmarkKNN specifically with target store STORE_104."""
        target_id = "STORE_104"
        lookup = self.knn_engine.find_peer_group(target_id)

        self.assertEqual(lookup["target_store_id"], target_id)
        self.assertEqual(lookup["region"], "West")
        self.assertEqual(lookup["store_tier"], "Flagship Tier 1")
        self.assertIn(lookup["cluster_id"], range(4))
        self.assertIn(lookup["cluster_label"], ARCHETYPE_LABELS.values())

        peers = lookup["nearest_peers"]
        self.assertEqual(len(peers), 3, "Must return exactly 3 closest peers")

        # Assert target store itself is not in peers
        peer_ids = [p["peer_store_id"] for p in peers]
        self.assertNotIn(target_id, peer_ids)

        # Assert non-negative, monotonically non-decreasing Euclidean distances
        distances = [p["euclidean_distance"] for p in peers]
        for d in distances:
            self.assertGreaterEqual(d, 0.0)
        self.assertEqual(distances, sorted(distances))

    def test_boundary_stores_fleet_coverage(self):
        """Tests first store, last store, and all 50 stores in fleet across boundary conditions."""
        store_ids = self.clustered_df["store_id"].tolist()
        self.assertEqual(len(store_ids), 50)

        # Boundary: First store (STORE_101)
        first_store = store_ids[0]
        first_lookup = self.knn_engine.find_peer_group(first_store)
        self.assertEqual(first_lookup["target_store_id"], first_store)
        self.assertEqual(len(first_lookup["nearest_peers"]), 3)

        # Boundary: Last store (STORE_150)
        last_store = store_ids[-1]
        last_lookup = self.knn_engine.find_peer_group(last_store)
        self.assertEqual(last_lookup["target_store_id"], last_store)
        self.assertEqual(len(last_lookup["nearest_peers"]), 3)

        # Verify complete fleet coverage: every store yields valid peers
        for s_id in store_ids:
            res = self.knn_engine.find_peer_group(s_id)
            self.assertEqual(len(res["nearest_peers"]), 3)
            self.assertNotIn(s_id, [p["peer_store_id"] for p in res["nearest_peers"]])

    def test_unknown_store_ids_raise_value_error(self):
        """Asserts unknown store IDs raise ValueError with descriptive messaging."""
        invalid_ids = [
            "UNKNOWN_STORE",
            "STORE_999",
            "STORE_000",
            "",
            "   ",
            "None",
            "INVALID_FORMAT_STORE",
        ]
        for invalid_id in invalid_ids:
            with self.assertRaises(ValueError, msg=f"Failed to raise ValueError for '{invalid_id}'"):
                self.knn_engine.find_peer_group(invalid_id)

    def test_unfitted_knn_raises_runtime_error(self):
        """Asserts calling find_peer_group on an unfitted StoreBenchmarkKNN raises RuntimeError."""
        unfitted_knn = StoreBenchmarkKNN(n_neighbors=4)
        with self.assertRaises(RuntimeError):
            unfitted_knn.find_peer_group("STORE_104")

    def test_kmeans_and_knn_serialization_deserialization_parity(self):
        """Asserts 100% mathematical and behavioral parity between pre-save and loaded pipeline bundles."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle_path = save_pipeline(
                scaler=self.scaler,
                kmeans=self.kmeans,
                knn_engine=self.knn_engine,
                store_summary_df=self.clustered_df,
                output_dir=tmp_dir,
            )
            self.assertTrue(bundle_path.exists())

            loaded_bundle = load_pipeline(bundle_path)
            loaded_scaler = loaded_bundle["scaler"]
            loaded_kmeans = loaded_bundle["kmeans"]
            loaded_knn = loaded_bundle["knn_engine"]
            loaded_df = loaded_bundle["store_summary_df"]

            # 1. Scaler Parity
            np.testing.assert_allclose(self.scaler.mean_, loaded_scaler.mean_)
            np.testing.assert_allclose(self.scaler.scale_, loaded_scaler.scale_)

            # 2. KMeans Centers and Inertia Parity
            np.testing.assert_allclose(
                self.kmeans.cluster_centers_, loaded_kmeans.cluster_centers_
            )
            self.assertAlmostEqual(self.kmeans.inertia_, loaded_kmeans.inertia_, places=4)

            # 3. DataFrame Parity
            pd.testing.assert_frame_equal(self.clustered_df, loaded_df)

            # 4. KNN Peer Query Parity across multiple test stores
            test_stores = ["STORE_101", "STORE_104", "STORE_125", "STORE_150"]
            for s_id in test_stores:
                orig_res = self.knn_engine.find_peer_group(s_id)
                load_res = loaded_knn.find_peer_group(s_id)

                self.assertEqual(orig_res["cluster_id"], load_res["cluster_id"])
                self.assertEqual(orig_res["cluster_label"], load_res["cluster_label"])

                orig_peers = orig_res["nearest_peers"]
                load_peers = load_res["nearest_peers"]
                self.assertEqual(len(orig_peers), len(load_peers))

                for p_orig, p_load in zip(orig_peers, load_peers):
                    self.assertEqual(p_orig["peer_store_id"], p_load["peer_store_id"])
                    self.assertAlmostEqual(
                        p_orig["euclidean_distance"],
                        p_load["euclidean_distance"],
                        places=4,
                    )


class TestValueRealizationStress(unittest.TestCase):
    """Rigorous stress tests for value realization equations, scale limits, and financial integrity."""

    def test_zero_stores_boundary(self):
        """Boundary condition: 0 stores."""
        inputs = ValueCalculatorInputs(store_count=0)
        output = calculate_value_realization(inputs)

        self.assertEqual(output.total_fleet_sales, 0.0)
        self.assertEqual(output.annual_fleet_markdown_volume, 0.0)
        self.assertEqual(output.annual_fleet_shrinkage_loss, 0.0)
        self.assertEqual(output.annual_markdown_savings, 0.0)
        self.assertEqual(output.gross_margin_recovery, 0.0)
        self.assertEqual(output.annual_shrinkage_recovery, 0.0)
        self.assertEqual(output.hours_saved_annually, 0.0)
        self.assertEqual(output.annual_labor_savings, 0.0)
        self.assertEqual(output.annual_stockout_recapture, 0.0)
        self.assertEqual(output.total_annual_value, 0.0)

        # Software cost with 0 stores is just the base platform fee ($50k)
        self.assertEqual(output.annual_software_investment, 50_000.0)
        self.assertEqual(output.net_annual_benefit, -50_000.0)
        # Payback period guarded from division by zero
        self.assertEqual(output.payback_period_months, 0.0)
        # 3-Year ROI is -100% since value is 0
        self.assertAlmostEqual(output.net_3year_roi_pct, -100.0, places=2)

    def test_ten_thousand_stores_massive_scale(self):
        """Boundary condition: 10,000 stores ($32B fleet revenue)."""
        inputs = ValueCalculatorInputs(store_count=10_000)
        output = calculate_value_realization(inputs)

        expected_sales = 10_000 * 3_200_000.0  # $32,000,000,000
        self.assertEqual(output.total_fleet_sales, expected_sales)

        expected_software_cost = 50_000.0 + (10_000 * 1_200.0)  # $12,050,000
        self.assertEqual(output.annual_software_investment, expected_software_cost)

        self.assertGreater(output.total_annual_value, 0.0)
        self.assertGreater(output.net_annual_benefit, 0.0)
        self.assertGreater(output.net_3year_roi_pct, 0.0)
        self.assertGreater(output.payback_period_months, 0.0)
        # Check no float overflow or NaN/Inf
        self.assertTrue(np.isfinite(output.total_annual_value))
        self.assertTrue(np.isfinite(output.net_annual_benefit))
        self.assertTrue(np.isfinite(output.net_3year_roi_pct))
        self.assertTrue(np.isfinite(output.payback_period_months))

    def test_zero_markdown_shock_and_optimization(self):
        """Boundary condition: 0% markdown optimization lift."""
        inputs = ValueCalculatorInputs(target_markdown_opt_pct=0.0)
        output = calculate_value_realization(inputs)

        self.assertEqual(output.annual_markdown_savings, 0.0)
        self.assertEqual(output.gross_margin_recovery, 0.0)
        # Remaining pillars still generate positive value
        self.assertGreater(output.annual_shrinkage_recovery, 0.0)
        self.assertGreater(output.annual_labor_savings, 0.0)
        self.assertGreater(output.annual_stockout_recapture, 0.0)
        self.assertGreater(output.total_annual_value, 0.0)

    def test_hundred_percent_markdown_shock_and_optimization(self):
        """Boundary condition: 100% baseline markdown and 100% optimization."""
        inputs = ValueCalculatorInputs(
            store_count=10,
            avg_store_sales=1_000_000.0,
            baseline_markdown_pct=100.0,
            target_markdown_opt_pct=100.0,
            cost_to_retail_ratio=0.5,
        )
        output = calculate_value_realization(inputs)

        self.assertEqual(output.total_fleet_sales, 10_000_000.0)
        self.assertEqual(output.annual_fleet_markdown_volume, 10_000_000.0)
        self.assertEqual(output.annual_markdown_savings, 10_000_000.0)
        self.assertEqual(output.gross_margin_recovery, 5_000_000.0)

    def test_extreme_store_metrics(self):
        """Extreme parameter inputs: zero sales, zero hourly rate, zero incidents, zero fees."""
        # Scenario A: Zero wages and zero stockout incidents
        inputs_a = ValueCalculatorInputs(
            store_mgr_hourly_rate=0.0,
            annual_stockout_incidents=0,
        )
        output_a = calculate_value_realization(inputs_a)
        self.assertEqual(output_a.annual_labor_savings, 0.0)
        self.assertEqual(output_a.annual_stockout_recapture, 0.0)

        # Scenario B: Zero software fees (free platform)
        inputs_b = ValueCalculatorInputs(
            base_platform_fee=0.0,
            per_store_annual_fee=0.0,
        )
        output_b = calculate_value_realization(inputs_b)
        self.assertEqual(output_b.annual_software_investment, 0.0)
        self.assertEqual(output_b.net_annual_benefit, output_b.total_annual_value)
        # ROI and Payback when investment is 0.0 should be guarded to 0.0
        self.assertEqual(output_b.net_3year_roi_pct, 0.0)
        self.assertEqual(output_b.payback_period_months, 0.0)

        # Scenario C: Cost-to-retail ratio extremes (1.0 = no markup, 0.0 = pure margin)
        inputs_c1 = ValueCalculatorInputs(cost_to_retail_ratio=1.0)
        output_c1 = calculate_value_realization(inputs_c1)
        self.assertEqual(output_c1.gross_margin_recovery, 0.0)

        inputs_c2 = ValueCalculatorInputs(cost_to_retail_ratio=0.0)
        output_c2 = calculate_value_realization(inputs_c2)
        self.assertEqual(output_c2.gross_margin_recovery, output_c2.annual_markdown_savings)

    def test_financial_integrity_invariants(self):
        """Validates fundamental financial accounting identities across diverse test inputs."""
        scenarios = [
            ValueCalculatorInputs(store_count=10, avg_store_sales=1_000_000.0),
            ValueCalculatorInputs(store_count=50, avg_store_sales=2_500_000.0),
            ValueCalculatorInputs(store_count=500, avg_store_sales=5_000_000.0),
            ValueCalculatorInputs(store_count=2000, avg_store_sales=10_000_000.0),
        ]

        for inp in scenarios:
            out = calculate_value_realization(inp)

            # Invariant 1: Total Annual Value is sum of 4 components
            calc_total = (
                out.annual_markdown_savings
                + out.annual_shrinkage_recovery
                + out.annual_labor_savings
                + out.annual_stockout_recapture
            )
            self.assertAlmostEqual(out.total_annual_value, calc_total, places=2)

            # Invariant 2: Net Annual Benefit = Total Annual Value - Annual Investment
            self.assertAlmostEqual(
                out.net_annual_benefit,
                out.total_annual_value - out.annual_software_investment,
                places=2,
            )

            # Invariant 3: 3-Year Cumulative ROI formula
            three_yr_inv = 3.0 * out.annual_software_investment
            three_yr_val = 3.0 * out.total_annual_value
            expected_roi = ((three_yr_val - three_yr_inv) / three_yr_inv) * 100.0
            self.assertAlmostEqual(out.net_3year_roi_pct, expected_roi, places=2)

            # Invariant 4: Payback Period Months formula
            expected_payback = (out.annual_software_investment / out.total_annual_value) * 12.0
            self.assertAlmostEqual(out.payback_period_months, expected_payback, places=2)


if __name__ == "__main__":
    unittest.main()
