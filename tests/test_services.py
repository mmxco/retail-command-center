"""
tests/test_services.py
Unit tests for shared formatting, KaTeX sanitization, and LLM advisory services (Milestone 5 / R4 / R5).
Verifies:
- Dollar sign sanitization across currency formats, decimals, and markdown bold/italics
- KaTeX mathematical operator conversion and formula math-mode stripping
- Sanitizer idempotency (multiple passes do not double-escape)
- Underscore variable identifier protection (STORE_104 untouched)
- Windows SSL client configuration (http_options client_args verify=False)
- Offline heuristic fallbacks for CFO briefs, business case memos, and brief text
- Candidate model fallback cascade execution under simulated API failures
"""

import unittest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from services.formatters import (
    sanitize_markdown_dollars,
    escape_dollars,
    clean_latex_operators,
    normalize_cfo_markdown,
)
from services.llm_advisor import (
    get_gemini_client,
    call_gemini_cascade,
    generate_cfo_risk_brief,
    generate_heuristic_cfo_brief,
    generate_executive_business_case,
    generate_heuristic_business_case,
    generate_ai_cfo_brief_text,
    DEFAULT_CANDIDATE_MODELS,
)


@dataclass
class DummyDashboardState:
    store_id: str = "STORE_104"
    store_tier: str = "Flagship Tier 1"
    region: str = "West"
    markdown_shock_pct: float = 25.0
    cost_to_retail_ratio: float = 0.4820
    baseline_ending_retail: float = 1_250_000.0
    shocked_ending_retail: float = 1_120_000.0
    baseline_ending_cost: float = 602_500.0
    shocked_ending_cost: float = 539_840.0
    balance_sheet_asset_deflation: float = 62_660.0
    gross_margin_decay_pct: float = -7.0
    projected_gmroi: float = 10.2
    nearest_peers: list = None
    archetype_label: str = "Balanced Regional Performers"
    sales_lift_pct: float = 8.75
    baseline_gm_pct: float = 58.4
    shocked_gm_pct: float = 51.4

    def __post_init__(self):
        if self.nearest_peers is None:
            self.nearest_peers = ["STORE_124", "STORE_110", "STORE_144"]


@dataclass
class DummyInputs:
    store_count: int = 120
    base_platform_fee: float = 50000.0
    per_store_annual_fee: float = 1200.0


@dataclass
class DummyOutput:
    total_annual_value: float = 5902110.0
    annual_software_investment: float = 194000.0
    net_annual_benefit: float = 5708110.0
    payback_period_months: float = 0.39
    net_3year_roi_pct: float = 2842.0


class TestSharedServices(unittest.TestCase):
    """Test suite for formatters, KaTeX math stripping, and LLM advisor cascade."""

    # --------------------------------------------------------------------------
    # 1. DOLLAR SIGN SANITIZATION & IDEMPOTENCY
    # --------------------------------------------------------------------------
    def test_sanitize_markdown_dollars_standard(self):
        """Verify standard currency formats are properly escaped."""
        self.assertEqual(sanitize_markdown_dollars("$100"), r"\$100")
        self.assertEqual(sanitize_markdown_dollars("$1,250,000.00"), r"\$1,250,000.00")
        self.assertEqual(sanitize_markdown_dollars("$406.4k"), r"\$406.4k")
        self.assertEqual(
            sanitize_markdown_dollars("Sales of $500k with $50k markdown"),
            r"Sales of \$500k with \$50k markdown",
        )

    def test_sanitize_markdown_dollars_preserves_already_escaped(self):
        """Verify already-escaped dollar signs are not double-escaped."""
        self.assertEqual(sanitize_markdown_dollars(r"\$100"), r"\$100")
        self.assertEqual(sanitize_markdown_dollars(r"Price is \$1,250.00"), r"Price is \$1,250.00")
        self.assertEqual(
            sanitize_markdown_dollars(r"Mixed: $50 and \$100"),
            r"Mixed: \$50 and \$100",
        )

    def test_sanitize_markdown_dollars_markdown_styling(self):
        """Verify bold, italic, and strikethrough markdown syntax is preserved."""
        self.assertEqual(sanitize_markdown_dollars("**$406,394.88**"), r"**\$406,394.88**")
        self.assertEqual(sanitize_markdown_dollars("*$50*"), r"*\$50*")
        self.assertEqual(sanitize_markdown_dollars("***$25,000***"), r"***\$25,000***")
        self.assertEqual(sanitize_markdown_dollars("~~$12.50~~"), r"~~\$12.50~~")

    def test_sanitize_markdown_dollars_idempotency(self):
        """Verify multiple passes are completely idempotent: f(f(x)) == f(x)."""
        samples = [
            "$100",
            r"\$100",
            "**$406,394.88**",
            "Cost basis of $907,945.68 shrinks by $406.4k.",
            "",
            "No dollars here",
        ]
        for s in samples:
            first_pass = sanitize_markdown_dollars(s)
            second_pass = sanitize_markdown_dollars(first_pass)
            self.assertEqual(second_pass, first_pass, f"Failed idempotency on: {s}")

    def test_sanitize_markdown_dollars_edge_cases(self):
        """Verify edge cases: None, empty string, and whitespace."""
        self.assertEqual(sanitize_markdown_dollars(None), "")
        self.assertEqual(sanitize_markdown_dollars(""), "")
        self.assertEqual(escape_dollars("$100"), r"\$100")  # Backward-compatible alias

    # --------------------------------------------------------------------------
    # 2. LATEX OPERATOR & FORMULA NORMALIZATION
    # --------------------------------------------------------------------------
    def test_clean_latex_operators(self):
        """Verify LaTeX math operators are converted to clean plain-text operators."""
        self.assertIn(" * ", clean_latex_operators(r"A \times B"))
        self.assertIn(" * ", clean_latex_operators(r"A \cdot B"))
        self.assertIn(" ≈ ", clean_latex_operators(r"A \approx B"))
        self.assertIn(" <= ", clean_latex_operators(r"A \le B"))
        self.assertIn(" <= ", clean_latex_operators(r"A \leq B"))
        self.assertIn(" >= ", clean_latex_operators(r"A \ge B"))
        self.assertIn(" >= ", clean_latex_operators(r"A \geq B"))
        self.assertEqual(clean_latex_operators(r"\text{Ending Inventory}"), "Ending Inventory")

    def test_latex_subscript_and_math_mode_stripping(self):
        """Verify LaTeX subscripts and math mode delimiters are stripped."""
        self.assertEqual(clean_latex_operators(r"EI_{Cost}"), "EI_Cost")
        self.assertEqual(clean_latex_operators(r"$EI_{Cost} = EI_{Retail} \times 0.40$"), "EI_Cost = EI_Retail * 0.40")
        self.assertEqual(clean_latex_operators(r"$EI$"), "EI")

    def test_underscore_variable_protection(self):
        """Verify store identifiers with underscores are not converted to LaTeX subscripts."""
        text = "Store STORE_104 in REGION_WEST"
        self.assertEqual(clean_latex_operators(text), text)

    def test_composite_normalize_cfo_markdown(self):
        """Verify composite normalize_cfo_markdown passes 100% of legacy assertions."""
        raw_text = (
            r"Under the RIM framework ($EI_{\text{Cost}} = EI_{\text{Retail}} \times 0.4102$). "
            "The unbudgeted +25% clearance markdown shock forces an immediate retail value destruction of "
            "$990,295.29 at STORE_104 (collapsing Ending Inventory at Retail from $2,185,801.33 to "
            "$1,195,506.04). Applying the 41.02% cost complement yields **$406,394.88** (reducing inventory "
            "cost basis from $907,945.68 to $501,550.80). Asset collapse shrinks borrowing base by $406.4k."
        )
        normalized = normalize_cfo_markdown(raw_text)
        self.assertNotIn(r"\text", normalized)
        self.assertNotIn(r"\times", normalized)
        self.assertIn("EI_Cost = EI_Retail * 0.4102", normalized)
        self.assertIn(r"\$990,295.29", normalized)
        self.assertIn(r"\$2,185,801.33", normalized)
        self.assertIn(r"\$1,195,506.04", normalized)
        self.assertIn(r"**\$406,394.88**", normalized)
        self.assertIn(r"\$907,945.68", normalized)
        self.assertIn(r"\$501,550.80", normalized)
        self.assertIn(r"\$406.4k", normalized)
        self.assertIn("STORE_104", normalized)

    # --------------------------------------------------------------------------
    # 3. WINDOWS SSL CONFIGURATION & CLIENT CREATION
    # --------------------------------------------------------------------------
    @patch("services.llm_advisor.genai")
    @patch("services.llm_advisor.types")
    def test_windows_ssl_configuration(self, mock_types, mock_genai):
        """Verify Gemini client specifies client_args={'verify': False} for Windows SSL cert compatibility."""
        mock_types.HttpOptions.return_value = MagicMock()
        client = get_gemini_client(api_key="test_api_key")
        
        # Verify HttpOptions called with client_args={"verify": False}
        mock_types.HttpOptions.assert_called_with(client_args={"verify": False})
        mock_genai.Client.assert_called_once()

    # --------------------------------------------------------------------------
    # 4. OFFLINE HEURISTIC FALLBACKS
    # --------------------------------------------------------------------------
    def test_offline_heuristic_cfo_brief(self):
        """Verify CFO brief generation in offline mode produces structured executive memo."""
        state = DummyDashboardState()
        result = generate_cfo_risk_brief(state, api_key=None)
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        self.assertIn("brief_text", result)
        self.assertGreater(len(result["brief_text"]), 200)
        self.assertIn(r"\$62,660", result["brief_text"])
        self.assertIn("STORE_124", result["brief_text"])
        self.assertIn("Clearance Throttle", result["brief_text"])

    def test_offline_heuristic_business_case(self):
        """Verify business case memo generation in offline mode produces complete 1-pager."""
        inputs = DummyInputs()
        output = DummyOutput()
        result = generate_executive_business_case(inputs, output, api_key=None)
        self.assertIsInstance(result, dict)
        self.assertTrue(result["success"])
        self.assertIn("brief_text", result)
        self.assertGreater(len(result["brief_text"]), 200)
        self.assertIn("120 stores", result["brief_text"])
        self.assertIn("Payback Horizon", result["brief_text"])

    def test_offline_heuristic_tab3_brief_text(self):
        """Verify Tab 3 backward-compatibility brief text generator."""
        brief = generate_ai_cfo_brief_text(
            store_id="STORE_104",
            archetype="Balanced Regional Performers",
            tier="Flagship Tier 1",
            region="West",
            shock_pct=25,
            metrics={
                "sales_lift_pct": 8.75,
                "total_gm_decay": -7.0,
                "total_asset_deflation": 45000.0,
                "avg_cr_ratio": 0.4125,
                "base_gm_pct": 58.4,
                "shock_gm_pct": 51.4,
                "top_peer": "STORE_124",
            },
        )
        self.assertIsInstance(brief, str)
        self.assertGreater(len(brief), 100)
        self.assertIn("Balance Sheet Asset Deflation", brief)
        self.assertIn("Guardrails", brief)

    # --------------------------------------------------------------------------
    # 5. MODEL FALLBACK CASCADE
    # --------------------------------------------------------------------------
    @patch("services.llm_advisor.get_gemini_client")
    def test_model_fallback_cascade_recovers_on_secondary_model(self, mock_get_client):
        """Verify that when primary model raises an exception, cascade succeeds on secondary model."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Simulate Model 1 (e.g. gemini-2.5-pro) raising an exception (e.g. 404 or 429 quota error)
        # and Model 2 (gemini-3.6-flash) returning a valid text response
        mock_chat_fail = MagicMock()
        mock_chat_fail.send_message.side_effect = Exception("Model unavailable / QuotaExceeded")

        mock_chat_success = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Boardroom CFO brief text successfully generated by fallback model."
        mock_chat_success.send_message.return_value = mock_response

        # chats.create returns fail first, then success
        mock_client.chats.create.side_effect = [mock_chat_fail, mock_chat_success]

        text, model_used = call_gemini_cascade(
            prompt="Generate forensic financial diagnostic",
            candidate_models=["gemini-2.5-pro", "gemini-3.6-flash"],
            api_key="mock_key",
        )

        self.assertIsNotNone(text)
        self.assertEqual(model_used, "gemini-3.6-flash")
        self.assertIn("Boardroom CFO brief", text)

    @patch("services.llm_advisor.get_gemini_client")
    def test_model_fallback_cascade_total_exhaustion_returns_none(self, mock_get_client):
        """Verify that when all models in cascade fail, returns (None, None) cleanly without crash."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.chats.create.side_effect = Exception("All models exhausted")

        text, model_used = call_gemini_cascade(
            prompt="Generate analysis",
            candidate_models=["gemini-2.5-pro", "gemini-3.6-flash"],
            api_key="mock_key",
        )
        self.assertIsNone(text)
        self.assertIsNone(model_used)


if __name__ == "__main__":
    unittest.main()
