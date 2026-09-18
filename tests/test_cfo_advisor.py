"""
Unit tests for cfo_advisor.py: Google Gemini 2.5 Pro reasoning layer
and RIMDashboardState financial diagnostics for Retail Inventory Method forensics.
"""

import unittest
from cfo_advisor import (
    RIMDashboardState,
    build_cfo_reasoning_prompt,
    generate_heuristic_cfo_brief,
    generate_cfo_risk_brief,
    normalize_cfo_markdown,
)


class TestCFOAdvisor(unittest.TestCase):
    """Test suite verifying executive CFO advisory functionality."""

    def setUp(self):
        self.sample_state = RIMDashboardState(
            store_id="STORE_104",
            store_tier="Flagship Tier 1",
            region="West",
            markdown_shock_pct=25.0,
            cost_to_retail_ratio=0.4820,
            baseline_ending_retail=1_250_000.0,
            shocked_ending_retail=1_120_000.0,
            baseline_ending_cost=602_500.0,
            shocked_ending_cost=539_840.0,
            balance_sheet_asset_deflation=62_660.0,
            gross_margin_decay_pct=-7.0,
            projected_gmroi=10.2,
            nearest_peers=["STORE_124", "STORE_110", "STORE_144"],
            archetype_label="Balanced Regional Performers",
            sales_lift_pct=8.75,
            baseline_gm_pct=58.4,
            shocked_gm_pct=51.4,
        )

    def test_state_dataclass_properties(self):
        """Verify dataclass fields and types."""
        state = self.sample_state
        self.assertEqual(state.store_id, "STORE_104")
        self.assertEqual(state.markdown_shock_pct, 25.0)
        self.assertEqual(len(state.nearest_peers), 3)
        self.assertEqual(state.balance_sheet_asset_deflation, 62_660.0)

    def test_prompt_construction(self):
        """Verify prompt contains required retail financial forensics context."""
        prompt = build_cfo_reasoning_prompt(self.sample_state)
        self.assertIn("STORE_104", prompt)
        self.assertIn("+25%", prompt)
        self.assertIn("0.4820", prompt)
        self.assertIn("$62,660.00", prompt)
        self.assertIn("STORE_124", prompt)
        self.assertIn("Working Capital & Balance Sheet Asset Deflation", prompt)
        self.assertIn("Top-Line Volume Illusion vs. Margin Destruction", prompt)
        self.assertIn("Actionable Merchandising Guardrails", prompt)

    def test_heuristic_fallback_generation(self):
        """Verify deterministic rule-based brief produces structured C-suite output."""
        brief = generate_heuristic_cfo_brief(self.sample_state)
        self.assertGreater(len(brief), 250)
        self.assertIn("$62,660", brief)
        self.assertIn("0.4820", brief)
        self.assertIn("STORE_124", brief)
        self.assertIn("Clearance Throttle", brief)

    def test_normalize_cfo_markdown(self):
        """Verify LaTeX math equations are stripped and currency dollar signs escaped."""
        raw_text = (
            "Under the RIM framework ($EI_{\\text{Cost}} = EI_{\\text{Retail}} \\times 0.4102$). "
            "The unbudgeted +25% clearance markdown shock forces an immediate retail value destruction of "
            "$990,295.29 at STORE_104 (collapsing Ending Inventory at Retail from $2,185,801.33 to "
            "$1,195,506.04). Applying the 41.02% cost complement yields **$406,394.88** (reducing inventory "
            "cost basis from $907,945.68 to $501,550.80). Asset collapse shrinks borrowing base by $406.4k."
        )
        normalized = normalize_cfo_markdown(raw_text)
        # Verify LaTeX commands were converted
        self.assertNotIn("\\text", normalized)
        self.assertNotIn("\\times", normalized)
        self.assertIn("EI_Cost = EI_Retail * 0.4102", normalized)
        # Verify dollar amounts are properly escaped to prevent KaTeX math mode corruption
        self.assertIn(r"\$990,295.29", normalized)
        self.assertIn(r"\$2,185,801.33", normalized)
        self.assertIn(r"\$1,195,506.04", normalized)
        self.assertIn(r"**\$406,394.88**", normalized)
        self.assertIn(r"\$907,945.68", normalized)
        self.assertIn(r"\$501,550.80", normalized)
        self.assertIn(r"\$406.4k", normalized)
        # Verify store name is untouched
        self.assertIn("STORE_104", normalized)

    def test_generate_cfo_risk_brief_execution(self):
        """Verify top-level generator returns valid dictionary with text and metadata."""
        result = generate_cfo_risk_brief(self.sample_state)
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        self.assertIn("brief_text", result)
        self.assertIn("model_used", result)
        self.assertGreater(len(result["brief_text"]), 100)


if __name__ == "__main__":
    unittest.main()
