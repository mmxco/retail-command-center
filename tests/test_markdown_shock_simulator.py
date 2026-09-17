"""
Unit tests for the Markdown Shock Simulator & Valuation Forensics Engine.
"""

import unittest
from pathlib import Path

import pandas as pd

from markdown_shock_simulator import (
    generate_ai_cfo_brief,
    generate_mock_apparel_data,
    simulate_multi_period_rollforward,
    simulate_period_shock,
)


class TestMarkdownShockSimulator(unittest.TestCase):
    """Test suite for RIM markdown shock mathematics, roll-forward, and fallbacks."""

    def test_simulate_period_shock_mechanics(self):
        """Validates exact formula recalculations under markdown shock."""
        beg_cost = 100_000.0
        beg_retail = 250_000.0
        purch_cost = 40_000.0
        purch_retail = 100_000.0
        base_sales = 80_000.0
        base_mds = 15_000.0
        shrink = 1_500.0
        shock_pct = 25.0
        elasticity = 0.35

        res = simulate_period_shock(
            beg_cost=beg_cost,
            beg_retail=beg_retail,
            purchases_cost=purch_cost,
            purchases_retail=purch_retail,
            baseline_sales=base_sales,
            baseline_markdowns=base_mds,
            shrinkage=shrink,
            shock_pct=shock_pct,
            elasticity=elasticity,
        )

        # 1. Total Goods Available
        self.assertEqual(res["tgas_cost"], 140_000.0)
        self.assertEqual(res["tgas_retail"], 350_000.0)
        expected_cr = 140_000.0 / 350_000.0  # 0.40
        self.assertAlmostEqual(res["cr_ratio"], expected_cr, places=4)

        # 2. Simulated Reductions
        expected_sim_mds = base_mds * (1.0 + 0.25)  # 18,750
        self.assertAlmostEqual(res["sim_markdowns"], expected_sim_mds, places=2)

        expected_sim_sales = base_sales * (1.0 + (0.25 * 0.35))  # 80,000 * 1.0875 = 87,000
        self.assertAlmostEqual(res["sim_sales"], expected_sim_sales, places=2)

        # 3. Valuation & Deflation
        self.assertGreater(res["asset_deflation"], 0.0)
        self.assertLess(res["margin_decay"], 0.0)  # Margin must decrease under markdown shock

    def test_multi_period_rollforward_compounding(self):
        """Validates that multi-period roll-forward compounds inventory asset deflation."""
        mock_df = generate_mock_apparel_data(n_stores=1, n_months=12)

        sim_df, agg_metrics = simulate_multi_period_rollforward(
            store_df=mock_df,
            shock_pct=30.0,
            elasticity=0.35,
            trailing_months=12,
        )

        self.assertEqual(len(sim_df), 12)
        self.assertIn("total_asset_deflation", agg_metrics)
        self.assertIn("total_gm_decay", agg_metrics)
        self.assertGreater(agg_metrics["total_asset_deflation"], 0.0)
        self.assertLess(agg_metrics["total_gm_decay"], 0.0)

    def test_mock_data_fallback_generator(self):
        """Verifies that the mock data fallback generates a valid, non-empty dataset."""
        df = generate_mock_apparel_data(n_stores=5, n_months=6)
        self.assertEqual(len(df), 30)
        required_cols = [
            "store_id", "period_date", "beginning_inv_cost", "beginning_inv_retail",
            "purchases_cost", "purchases_retail", "net_sales", "promo_markdowns",
            "shrinkage", "ending_inv_cost", "ending_inv_retail", "gross_margin"
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)

    def test_ai_cfo_brief_generation(self):
        """Tests that the CFO brief generator returns structured executive text."""
        agg_metrics = {
            "total_base_sales": 1_000_000.0,
            "total_shock_sales": 1_080_000.0,
            "total_base_gm": 500_000.0,
            "total_shock_gm": 420_000.0,
            "base_gm_pct": 50.0,
            "shock_gm_pct": 38.89,
            "final_base_end_cost": 300_000.0,
            "final_shock_end_cost": 240_000.0,
            "total_asset_deflation": 60_000.0,
            "avg_cr_ratio": 0.4150,
            "total_gm_decay": -11.11,
        }

        brief = generate_ai_cfo_brief(
            store_id="STORE_104",
            tier="Flagship Tier 1",
            region="West",
            shock_pct=25,
            agg_metrics=agg_metrics,
        )

        self.assertIsInstance(brief, str)
        self.assertIn("Balance Sheet", brief)
        self.assertIn("Guardrails", brief)


if __name__ == "__main__":
    unittest.main()
