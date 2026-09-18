"""
Unit tests for value_calculator.py: Dynamic Value-Realization Calculator & ROI Simulator.
Validates retail financial mechanics, Plotly waterfall and 3-year horizon charts,
and AI business case memo generation.
"""

import unittest
from value_calculator import (
    ValueCalculatorInputs,
    ValueRealizationOutput,
    calculate_value_realization,
    create_value_waterfall_chart,
    create_3year_horizon_chart,
    generate_heuristic_business_case,
    generate_executive_business_case,
)


class TestValueCalculator(unittest.TestCase):
    """Test suite verifying mathematical precision and component generation."""

    def setUp(self):
        self.inputs = ValueCalculatorInputs(
            store_count=120,
            avg_store_sales=3_200_000.0,
            baseline_markdown_pct=28.0,
            target_markdown_opt_pct=3.5,
            baseline_shrinkage_pct=2.1,
            target_shrink_red_pct=12.0,
            cost_to_retail_ratio=0.4820,
            base_platform_fee=50_000.0,
            per_store_annual_fee=1_200.0,
            weekly_stockout_hours_per_store=4.0,
            store_mgr_hourly_rate=38.00,
            annual_stockout_incidents=45,
            avg_basket_value=165.00,
            stockout_recapture_rate_pct=25.0,
        )
        self.output = calculate_value_realization(self.inputs)

    def test_fleet_baseline_scale(self):
        """Verify fleet scale and gross volume calculations."""
        # Total sales = 120 * 3,200,000 = $384,000,000
        self.assertAlmostEqual(self.output.total_fleet_sales, 384_000_000.0, places=2)
        # Markdown volume = $384M * 28% = $107,520,000
        self.assertAlmostEqual(self.output.annual_fleet_markdown_volume, 107_520_000.0, places=2)
        # Shrinkage loss = $384M * 2.1% = $8,064,000
        self.assertAlmostEqual(self.output.annual_fleet_shrinkage_loss, 8_064_000.0, places=2)

    def test_pillar1_markdown_optimization(self):
        """Verify Pillar 1: Markdown savings & gross margin recovery."""
        # Markdown savings = $107,520,000 * 3.5% = $3,763,200
        self.assertAlmostEqual(self.output.annual_markdown_savings, 3_763_200.0, places=2)
        # Gross margin recovery = $3,763,200 * (1 - 0.4820) = $1,949,337.60
        expected_gm = 3_763_200.0 * (1.0 - 0.4820)
        self.assertAlmostEqual(self.output.gross_margin_recovery, expected_gm, places=2)

    def test_pillar2_shrinkage_recovery(self):
        """Verify Pillar 2: Shrinkage reduction recaptured."""
        # Shrink recovery = $8,064,000 * 12% = $967,680
        self.assertAlmostEqual(self.output.annual_shrinkage_recovery, 967_680.0, places=2)

    def test_pillar3_labor_and_stockouts(self):
        """Verify Pillar 3: Labor capacity and stockout walkaway recapture."""
        # Hours saved = 120 * 4 * 52 = 24,960 hours
        self.assertAlmostEqual(self.output.hours_saved_annually, 24_960.0, places=2)
        # Labor savings = 24,960 * $38 = $948,480
        self.assertAlmostEqual(self.output.annual_labor_savings, 948_480.0, places=2)
        # Stockout recapture = 120 * 45 * $165 * 25% = $222,750
        self.assertAlmostEqual(self.output.annual_stockout_recapture, 222_750.0, places=2)

    def test_net_economic_value_and_roi(self):
        """Verify total annual value, software cost, payback period, and 3-year ROI."""
        expected_total = 3_763_200.0 + 967_680.0 + 948_480.0 + 222_750.0
        self.assertAlmostEqual(self.output.total_annual_value, expected_total, places=2)

        # Software cost = $50,000 + (120 * $1,200) = $194,000
        self.assertEqual(self.output.annual_software_investment, 194_000.0)

        # Net annual benefit = Total - Cost
        self.assertAlmostEqual(
            self.output.net_annual_benefit,
            expected_total - 194_000.0,
            places=2,
        )

        # Payback period (Months) = (194,000 / Total) * 12
        expected_payback = (194_000.0 / expected_total) * 12.0
        self.assertAlmostEqual(self.output.payback_period_months, expected_payback, places=2)
        self.assertLess(self.output.payback_period_months, 1.0)  # Payback is ~0.39 months

        # 3-Year ROI > 2000%
        self.assertGreater(self.output.net_3year_roi_pct, 2000.0)

    def test_waterfall_chart(self):
        """Verify Waterfall Plotly figure creation and trace data."""
        fig = create_value_waterfall_chart(self.output)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.data), 1)
        waterfall_trace = fig.data[0]
        self.assertEqual(len(waterfall_trace.x), 6)
        self.assertEqual(waterfall_trace.measure[-1], "total")

    def test_horizon_chart(self):
        """Verify 3-Year Cumulative Value Horizon Plotly figure."""
        fig = create_3year_horizon_chart(self.output)
        self.assertIsNotNone(fig)
        # Should have 4 traces: Gross value, Software cost, Net benefit area, Payback marker
        self.assertEqual(len(fig.data), 4)

    def test_heuristic_business_case(self):
        """Verify deterministic fallback business case formatting."""
        brief = generate_heuristic_business_case(self.inputs, self.output)
        self.assertIsInstance(brief, str)
        self.assertIn("120 stores", brief)
        self.assertIn("Payback Horizon", brief)
        self.assertIn("Cost of Inaction", brief)
        self.assertIn("Steering Committee", brief)
        # Verify no unescaped raw LaTeX math delimiters
        self.assertNotIn("$EI", brief)

    def test_generate_executive_business_case_execution(self):
        """Verify executive business case memo generator execution."""
        result = generate_executive_business_case(self.inputs, self.output)
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        self.assertIn("brief_text", result)
        self.assertIn("model_used", result)
        self.assertGreater(len(result["brief_text"]), 200)


if __name__ == "__main__":
    unittest.main()
