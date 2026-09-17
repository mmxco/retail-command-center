"""
Unit tests for the Forward 12-Period Valuation Divergence Charts Module.
"""

import unittest
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from valuation_charts import (
    create_synchronized_valuation_figure,
    generate_mock_store_history,
    project_forward_12_periods,
)


class TestValuationChartsModule(unittest.TestCase):
    """Test suite for forward 12-period projection engine and Plotly subplots figure."""

    def setUp(self):
        self.mock_history = generate_mock_store_history(store_id="STORE_104", n_months=12)

    def test_mock_store_history_generation(self):
        """Validates that mock historical baseline generates 12 complete rows."""
        self.assertEqual(len(self.mock_history), 12)
        required_cols = [
            "period_date", "store_id", "beginning_inv_cost", "beginning_inv_retail",
            "purchases_cost", "purchases_retail", "net_sales", "promo_markdowns",
            "ending_inv_cost", "ending_inv_retail"
        ]
        for col in required_cols:
            self.assertIn(col, self.mock_history.columns)

    def test_project_forward_12_periods_structure(self):
        """Validates that projection engine returns exactly 12 future periods with expected columns."""
        df_proj = project_forward_12_periods(
            baseline_store_df=self.mock_history,
            markdown_uplift_pct=0.25,
            sales_lift_elasticity=0.35,
            shrink_rate=0.02,
        )

        self.assertEqual(len(df_proj), 12)
        expected_cols = [
            "period_index", "period_label", "period_date",
            "planned_ending_retail", "planned_ending_cost",
            "shocked_ending_retail", "shocked_ending_cost",
            "planned_markdowns", "shocked_markdowns",
            "asset_deflation", "planned_cr_ratio", "shocked_cr_ratio"
        ]
        for col in expected_cols:
            self.assertIn(col, df_proj.columns)

        # Period index from 1 to 12
        self.assertEqual(list(df_proj["period_index"]), list(range(1, 13)))

    def test_projection_mathematical_deflation(self):
        """Validates that markdown uplift causes higher reductions and lower ending cost."""
        df_proj = project_forward_12_periods(
            baseline_store_df=self.mock_history,
            markdown_uplift_pct=0.30,
            sales_lift_elasticity=0.35,
            shrink_rate=0.02,
        )

        # Shocked markdowns must be 30% higher than planned markdowns (within penny rounding)
        for _, row in df_proj.iterrows():
            self.assertAlmostEqual(
                row["shocked_markdowns"], row["planned_markdowns"] * 1.30, delta=0.05
            )
            # Asset deflation must be positive (planned cost > shocked cost)
            self.assertGreater(row["asset_deflation"], 0.0)
            self.assertGreater(row["planned_ending_cost"], row["shocked_ending_cost"])

    def test_create_synchronized_valuation_figure(self):
        """Validates that create_synchronized_valuation_figure generates a valid 2-panel Plotly figure."""
        df_proj = project_forward_12_periods(self.mock_history)
        fig = create_synchronized_valuation_figure(df_proj)

        self.assertIsInstance(fig, go.Figure)
        # Verify 6 traces total (3 top panel + 3 bottom panel)
        self.assertEqual(len(fig.data), 6)

        # Check trace names
        trace_names = [t.name for t in fig.data]
        self.assertIn("Retail Book Value (Planned Retail)", trace_names)
        self.assertIn("Planned Cost Asset (Baseline)", trace_names)
        self.assertIn("Shocked Cost Asset (Impaired)", trace_names)
        self.assertIn("Planned Markdowns ($)", trace_names)
        self.assertIn("Shocked Markdowns ($)", trace_names)
        self.assertIn("Cost-to-Retail Complement (%)", trace_names)

        # Verify layout configurations
        self.assertEqual(fig.layout.hovermode, "x unified")
        self.assertEqual(fig.layout.barmode, "group")


if __name__ == "__main__":
    unittest.main()
