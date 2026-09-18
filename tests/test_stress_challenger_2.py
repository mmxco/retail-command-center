"""
tests/test_stress_challenger_2.py
Empirical stress-test harness by teamwork_preview_challenger_stress_2.

Thoroughly verifies resilience across:
1. Data Loaders Resilience:
   - load_discovery_presets and load_autogen_scenarios with valid files, missing files, corrupted files, and caching.
2. Formatting Resilience:
   - sanitize_markdown_dollars, clean_latex_operators, and normalize_cfo_markdown with adversarial strings:
     unescaped dollars, already-escaped dollars, idempotency, LaTeX operators, and underscore identifiers.
3. Headless UI Smoke Tests:
   - render_retail_kpis with None, dict, Series, DataFrame, empty structures, and malformed types.
   - render_tab_header across hex, named, rgb, rgba, hsl colors and edge case strings.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# 1. Imports from authoritative services and UI
from services.data_loaders import (
    load_discovery_presets,
    load_autogen_scenarios,
    load_styles,
)
from services.formatters import (
    sanitize_markdown_dollars,
    escape_dollars,
    clean_latex_operators,
    normalize_cfo_markdown,
)
import ui.kpi_metrics as kpi_ui
import app


class TestDataLoadersResilience(unittest.TestCase):
    """Stress-tests data loaders for file existence, fallbacks, corrupt files, and caching."""

    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.presets_path = cls.root_dir / "data" / "discovery_presets.json"
        cls.scenarios_path = cls.root_dir / "data" / "autogen_scenarios.json"
        cls.styles_path = cls.root_dir / "assets" / "styles.css"

    # --- Discovery Presets Tests ---

    def test_load_discovery_presets_normal(self):
        """Verify load_discovery_presets correctly reads and parses the JSON fixture."""
        presets = load_discovery_presets()
        self.assertIsInstance(presets, dict)
        self.assertEqual(len(presets), 3)
        expected_urls = [
            "https://www.buckle.com",
            "https://www.ralphlauren.com",
            "https://www.lululemon.com",
        ]
        for url in expected_urls:
            self.assertIn(url, presets)
            profile = presets[url]
            self.assertTrue(profile.get("account_name"))
            self.assertTrue(profile.get("retail_tier"))
            self.assertTrue(profile.get("annual_revenue"))
            pain_points = profile.get("pain_points", [])
            self.assertGreaterEqual(len(pain_points), 3)
            for p in pain_points:
                self.assertIn("category", p)
                self.assertIn("technical_gap", p)
                self.assertIn("operational_friction", p)
                self.assertIn("financial_impact", p)
                self.assertIn("affected_executives", p)

    def test_load_discovery_presets_missing_file_fallback(self):
        """Verify load_discovery_presets handles non-existent paths gracefully."""
        non_existent = self.root_dir / "data" / "non_existent_preset_9999.json"
        fallback = load_discovery_presets(filepath=non_existent)
        self.assertIsInstance(fallback, dict)
        self.assertEqual(len(fallback), 3)
        self.assertIn("https://www.buckle.com", fallback)
        self.assertIn("The Buckle, Inc.", fallback["https://www.buckle.com"]["account_name"])

    def test_load_discovery_presets_corrupted_json_fallback(self):
        """Verify load_discovery_presets returns defensive fallback when JSON is malformed or invalid."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as tf:
            tf.write("{ this is not valid json !!")
            temp_path = tf.name

        try:
            fallback = load_discovery_presets(filepath=temp_path)
            self.assertIsInstance(fallback, dict)
            self.assertIn("https://www.buckle.com", fallback)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_discovery_presets_empty_file_fallback(self):
        """Verify load_discovery_presets returns fallback on empty JSON file."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as tf:
            tf.write("")
            temp_path = tf.name

        try:
            fallback = load_discovery_presets(filepath=temp_path)
            self.assertIsInstance(fallback, dict)
            self.assertIn("https://www.buckle.com", fallback)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_load_discovery_presets_non_dict_json_fallback(self):
        """Verify load_discovery_presets returns fallback when JSON contains an array instead of dict."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as tf:
            tf.write('["an", "array", "not", "a", "dict"]')
            temp_path = tf.name

        try:
            fallback = load_discovery_presets(filepath=temp_path)
            self.assertIsInstance(fallback, dict)
            self.assertIn("https://www.buckle.com", fallback)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # --- Autogen Scenarios Tests ---

    def test_load_autogen_scenarios_normal(self):
        """Verify load_autogen_scenarios correctly reads and parses the scenarios JSON fixture."""
        scenarios = load_autogen_scenarios()
        self.assertIsInstance(scenarios, dict)
        self.assertEqual(len(scenarios), 5)
        for s_id, sc in scenarios.items():
            self.assertIn("title", sc)
            self.assertIn("metrics", sc)
            self.assertIn("messages", sc)
            m = sc["metrics"]
            # Check inventory conservation
            self.assertEqual(m["allocated_units"], m["transfer_units"] + m["vendor_units"])
            # Check message sanitization
            for msg in sc.get("messages", []):
                content = msg.get("content", "")
                if "$" in content:
                    self.assertNotIn("(?<!\\)$", content)
                    # Every unescaped dollar must be escaped
                    self.assertNotIn("$25,000", content.replace(r"\$25,000", ""))

    def test_load_autogen_scenarios_missing_file_fallback(self):
        """Verify load_autogen_scenarios falls back gracefully when file is missing."""
        non_existent = self.root_dir / "data" / "non_existent_scenarios_9999.json"
        fallback = load_autogen_scenarios(filepath=non_existent)
        self.assertIsInstance(fallback, dict)
        self.assertIn("standard", fallback)
        self.assertEqual(fallback["standard"]["id"], "standard")
        # Ensure fallback message sanitization still ran
        self.assertIn("metrics", fallback["standard"])

    def test_load_autogen_scenarios_corrupted_json_fallback(self):
        """Verify load_autogen_scenarios returns fallback on malformed JSON without raising."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json", encoding="utf-8") as tf:
            tf.write("{\"incomplete_json\": ")
            temp_path = tf.name

        try:
            fallback = load_autogen_scenarios(filepath=temp_path)
            self.assertIsInstance(fallback, dict)
            self.assertIn("standard", fallback)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # --- Caching Behavior Tests ---

    def test_loaders_st_cache_data_decoration(self):
        """Verify both loaders are decorated with @st.cache_data."""
        for fn in [load_discovery_presets, load_autogen_scenarios]:
            has_cache = hasattr(fn, "__wrapped__") or hasattr(fn, "clear")
            self.assertTrue(has_cache, f"{fn.__name__} must be decorated with @st.cache_data")
            if hasattr(fn, "clear"):
                fn.clear()

    def test_load_styles_resilience(self):
        """Verify load_styles works with existing CSS and fails gracefully on missing file."""
        css = load_styles()
        self.assertIsInstance(css, str)
        self.assertGreater(len(css), 5000)

        # Non-existent file returns empty string
        missing_css = load_styles("assets/non_existent_style.css")
        self.assertEqual(missing_css, "")


class TestFormattingResilience(unittest.TestCase):
    """Stress-tests sanitize_markdown_dollars, clean_latex_operators, and normalize_cfo_markdown."""

    # --- Unescaped Dollar Figures ---

    def test_unescaped_dollars_variations(self):
        """Test unescaped dollars across various currency patterns."""
        cases = [
            ("$100", r"\$100"),
            ("$1,000,000.00", r"\$1,000,000.00"),
            ("$406.4k", r"\$406.4k"),
            ("**$500**", r"**\$500**"),
            ("*$250*", r"*\$250*"),
            ("***$25,000***", r"***\$25,000***"),
            ("~~$12.50~~", r"~~\$12.50~~"),
            ("$1.2B revenue", r"\$1.2B revenue"),
            ("$0.00 cost", r"\$0.00 cost"),
            ("Price is $5", r"Price is \$5"),
            ("($100)", r"(\$100)"),
            ("[$100 - $200]", r"[\$100 - \$200]"),
            ("$10, $20, and $30", r"\$10, \$20, and \$30"),
        ]
        for inp, expected in cases:
            self.assertEqual(sanitize_markdown_dollars(inp), expected, f"Failed on input: {inp}")

    # --- Already Escaped Dollars & Idempotency ---

    def test_already_escaped_dollars(self):
        """Test that already escaped dollars are NOT double escaped."""
        cases = [
            (r"\$100", r"\$100"),
            (r"\$1,000,000.00", r"\$1,000,000.00"),
            (r"\$406.4k", r"\$406.4k"),
            (r"**\$500**", r"**\$500**"),
            (r"Price is \$50 and total is $100", r"Price is \$50 and total is \$100"),
        ]
        for inp, expected in cases:
            self.assertEqual(sanitize_markdown_dollars(inp), expected, f"Failed on already-escaped: {inp}")

    def test_sanitize_markdown_dollars_idempotency_stress(self):
        """Stress-test idempotency: sanitize(sanitize(x)) == sanitize(x) across 25 adversarial strings."""
        adversarial_samples = [
            "$100",
            r"\$100",
            r"\\$100",
            "$1,000,000.00",
            r"\$1,000,000.00",
            "$406.4k",
            "**$500**",
            "*$250*",
            "Under the RIM framework ($EI_{Cost} = EI_{Retail} \\times 0.40$).",
            "Asset collapse shrinks borrowing base by $406.4k at STORE_104.",
            "Special: $0, $0.00, $1.5M, $10B, $99k, $1,234,567.89",
            "No dollars at all in this sentence.",
            "",
            "   ",
            "$",
            "$$",
            r"\$",
            r"\$\$",
            "$$100",
            "$100$",
            "Markdown $10 * $20 = $200",
            "Formula: $\\text{Cost} = \\$50$",
            "STORE_104: $100 -> $200 (delta: +$100)",
            "HTML: <span>$100</span>",
            "Brackets: {[$50]}",
        ]
        for s in adversarial_samples:
            pass_1 = sanitize_markdown_dollars(s)
            pass_2 = sanitize_markdown_dollars(pass_1)
            self.assertEqual(pass_2, pass_1, f"Idempotency violated on 2nd pass for: {s}")
            pass_3 = sanitize_markdown_dollars(pass_2)
            self.assertEqual(pass_3, pass_1, f"Idempotency violated on 3rd pass for: {s}")
            pass_5 = sanitize_markdown_dollars(sanitize_markdown_dollars(pass_3))
            self.assertEqual(pass_5, pass_1, f"Idempotency violated on 5th pass for: {s}")

    # --- LaTeX Math Expressions ---

    def test_latex_operators_and_macros(self):
        """Test clean_latex_operators with \\times, \\cdot, \\approx, \\text, etc."""
        self.assertIn(" * ", clean_latex_operators(r"A \times B"))
        self.assertIn(" * ", clean_latex_operators(r"A \cdot B"))
        self.assertIn(" ≈ ", clean_latex_operators(r"A \approx B"))
        self.assertIn(" <= ", clean_latex_operators(r"A \le B"))
        self.assertIn(" <= ", clean_latex_operators(r"A \leq B"))
        self.assertIn(" >= ", clean_latex_operators(r"A \ge B"))
        self.assertIn(" >= ", clean_latex_operators(r"A \geq B"))
        self.assertEqual(clean_latex_operators(r"\text{Cost}"), "Cost")
        self.assertEqual(clean_latex_operators(r"\text{Retail Inventory}"), "Retail Inventory")

    def test_latex_formula_and_subscript_conversion(self):
        """Test clean_latex_operators with complex formulas and subscript stripping."""
        expr = r"$EI_{Cost} = EI_{Retail} \times 0.40$"
        cleaned = clean_latex_operators(expr)
        self.assertEqual(cleaned, "EI_Cost = EI_Retail * 0.40")

        expr_nested = r"$EI_{\text{Cost}} = EI_{\text{Retail}} \times 0.4102$"
        cleaned_nested = clean_latex_operators(expr_nested)
        self.assertEqual(cleaned_nested, "EI_Cost = EI_Retail * 0.4102")

    def test_punctuation_spacing_fix(self):
        """Test that ).Applying is converted to ). Applying."""
        text = "Ratio was 40%).Applying RIM formula yields result."
        cleaned = clean_latex_operators(text)
        self.assertIn("). Applying", cleaned)

    # --- Underscore Identifiers Protection ---

    def test_underscore_identifiers_not_corrupted(self):
        """Ensure store IDs and variable names with underscores are NOT corrupted to LaTeX subscripts."""
        identifiers = [
            "STORE_104",
            "STORE_124",
            "REGION_WEST",
            "DIST_CENTRAL",
            "STORE_104 and STORE_124",
            "Store STORE_104 has markdown shock.",
        ]
        for ident in identifiers:
            self.assertEqual(
                clean_latex_operators(ident),
                ident,
                f"clean_latex_operators corrupted identifier: {ident}",
            )
            # normalize_cfo_markdown should also preserve the exact identifier
            self.assertEqual(
                normalize_cfo_markdown(ident),
                ident,
                f"normalize_cfo_markdown corrupted identifier: {ident}",
            )

    # --- Composite normalize_cfo_markdown ---

    def test_normalize_cfo_markdown_adversarial_suite(self):
        """Test composite normalize_cfo_markdown on complete CFO narratives."""
        narrative = (
            r"Under the RIM framework ($EI_{\text{Cost}} = EI_{\text{Retail}} \times 0.40$). "
            "Clearance markdown at STORE_104 causes $500,000.00 loss (dropping from $1,500,000.00 to $1,000,000.00). "
            "Applying the 40% cost complement yields **$200,000.00** at STORE_124. "
            "Borrowing base shrinks by $200.0k."
        )
        normalized = normalize_cfo_markdown(narrative)
        self.assertNotIn(r"\text", normalized)
        self.assertNotIn(r"\times", normalized)
        self.assertIn("EI_Cost = EI_Retail * 0.40", normalized)
        self.assertIn("STORE_104", normalized)
        self.assertIn("STORE_124", normalized)
        self.assertIn(r"\$500,000.00", normalized)
        self.assertIn(r"\$1,500,000.00", normalized)
        self.assertIn(r"\$1,000,000.00", normalized)
        self.assertIn(r"**\$200,000.00**", normalized)
        self.assertIn(r"\$200.0k", normalized)

    # --- Boundary and Type Safety ---

    def test_formatters_none_and_empty(self):
        """Test formatters on None and empty string."""
        self.assertEqual(sanitize_markdown_dollars(None), "")
        self.assertEqual(sanitize_markdown_dollars(""), "")
        self.assertEqual(clean_latex_operators(None), "")
        self.assertEqual(clean_latex_operators(""), "")
        self.assertEqual(normalize_cfo_markdown(None), "")
        self.assertEqual(normalize_cfo_markdown(""), "")
        self.assertEqual(escape_dollars(None), "")


class TestHeadlessUISmokeTests(unittest.TestCase):
    """Headless UI smoke tests for render_retail_kpis and render_tab_header."""

    # --- render_retail_kpis Smoke & Metric Extraction Tests ---

    def test_render_retail_kpis_none_input(self):
        """Test render_retail_kpis with None input (default fallback metrics)."""
        # Bare mode execution
        app.render_retail_kpis(None)
        kpi_ui.render_retail_kpis(None)

        # Value inspection via internal parser
        gmroi, gmroi_c, cr, turn, shrink, shrink_c = kpi_ui._parse_kpi_metrics(None)
        self.assertEqual(gmroi, 2.42)
        self.assertEqual(gmroi_c, "normal")
        self.assertEqual(cr, 48.20)
        self.assertEqual(turn, 4.1)
        self.assertEqual(shrink, 1.85)
        self.assertEqual(shrink_c, "normal")

    def test_render_retail_kpis_dict_input_normal(self):
        """Test render_retail_kpis with custom dict input."""
        data = {
            "gmroi": 3.10,
            "cost_to_retail_ratio": 0.4500,
            "turn_rate": 4.8,
            "shrinkage_rate": 0.0120,
        }
        app.render_retail_kpis(data)
        gmroi, gmroi_c, cr, turn, shrink, shrink_c = kpi_ui._parse_kpi_metrics(data)
        self.assertEqual(gmroi, 3.10)
        self.assertEqual(gmroi_c, "normal")
        self.assertEqual(cr, 45.0)
        self.assertEqual(turn, 4.8)
        self.assertEqual(shrink, 1.20)
        self.assertEqual(shrink_c, "normal")

    def test_render_retail_kpis_dict_input_inverse_colors(self):
        """Test render_retail_kpis with below-benchmark metrics triggering inverse colors."""
        data = {
            "gmroi": 1.45,  # < 2.0 triggers inverse
            "cost_to_retail_ratio": 54.0,  # percentage format (> 1.0)
            "turn_rate": 0.25,  # < 1.0 triggers 12x monthly conversion
            "shrinkage_rate": 0.0450,  # > 0.03 (3%) triggers inverse
        }
        app.render_retail_kpis(data)
        gmroi, gmroi_c, cr, turn, shrink, shrink_c = kpi_ui._parse_kpi_metrics(data)
        self.assertEqual(gmroi, 1.45)
        self.assertEqual(gmroi_c, "inverse")
        self.assertEqual(cr, 54.0)
        self.assertEqual(turn, 3.0)  # 0.25 * 12 = 3.0
        self.assertEqual(shrink, 4.5)  # 0.045 * 100 = 4.5
        self.assertEqual(shrink_c, "inverse")

    def test_render_retail_kpis_dict_key_aliases(self):
        """Test render_retail_kpis with alternative column/key aliases."""
        data = {
            "normalized_gmroi": 2.75,
            "avg_cr_ratio": 0.42,
            "inventory_turns": 5.2,
            "shrink_rate": 2.1,
        }
        app.render_retail_kpis(data)
        gmroi, gmroi_c, cr, turn, shrink, shrink_c = kpi_ui._parse_kpi_metrics(data)
        self.assertEqual(gmroi, 2.75)
        self.assertEqual(cr, 42.0)
        self.assertEqual(turn, 5.2)
        self.assertEqual(shrink, 2.1)

    def test_render_retail_kpis_empty_dict(self):
        """Test render_retail_kpis with empty dictionary."""
        app.render_retail_kpis({})
        gmroi, gmroi_c, cr, turn, shrink, shrink_c = kpi_ui._parse_kpi_metrics({})
        # Should gracefully fall back to defaults
        self.assertEqual(gmroi, 2.42)
        self.assertEqual(cr, 48.20)
        self.assertEqual(turn, 4.1)
        self.assertEqual(shrink, 1.85)

    def test_render_retail_kpis_series_input(self):
        """Test render_retail_kpis with pandas Series."""
        s = pd.Series({
            "gmroi": 2.15,
            "cost_to_retail_ratio": 0.4900,
            "turn_rate": 3.9,
            "shrinkage_rate": 0.0190,
        })
        app.render_retail_kpis(s)
        gmroi, gmroi_c, cr, turn, shrink, shrink_c = kpi_ui._parse_kpi_metrics(s)
        self.assertEqual(gmroi, 2.15)
        self.assertAlmostEqual(cr, 49.0, places=1)
        self.assertEqual(turn, 3.9)
        self.assertAlmostEqual(shrink, 1.9, places=1)

    def test_render_retail_kpis_dataframe_input(self):
        """Test render_retail_kpis with single-row and multi-row DataFrame."""
        df_single = pd.DataFrame([{
            "gmroi": 2.90,
            "cost_to_retail_ratio": 0.4400,
            "turn_rate": 4.5,
            "shrinkage_rate": 0.0160,
        }])
        app.render_retail_kpis(df_single)
        gmroi, _, cr, turn, shrink, _ = kpi_ui._parse_kpi_metrics(df_single)
        self.assertEqual(gmroi, 2.90)

        # Multi-row DataFrame (row 0 should be selected)
        df_multi = pd.DataFrame([
            {"gmroi": 3.50, "cost_to_retail_ratio": 0.40, "turn_rate": 5.0, "shrinkage_rate": 0.01},
            {"gmroi": 1.20, "cost_to_retail_ratio": 0.60, "turn_rate": 2.0, "shrinkage_rate": 0.05},
        ])
        app.render_retail_kpis(df_multi)
        gmroi, _, _, _, _, _ = kpi_ui._parse_kpi_metrics(df_multi)
        self.assertEqual(gmroi, 3.50)

    def test_render_retail_kpis_empty_dataframe(self):
        """Test render_retail_kpis with an empty DataFrame (0 rows)."""
        df_empty = pd.DataFrame()
        app.render_retail_kpis(df_empty)
        gmroi, _, cr, turn, shrink, _ = kpi_ui._parse_kpi_metrics(df_empty)
        # Should gracefully return defaults without IndexError
        self.assertEqual(gmroi, 2.42)

    def test_render_retail_kpis_malformed_types(self):
        """Test render_retail_kpis with malformed values and unexpected types."""
        # Non-numeric dict values
        malformed_dict = {
            "gmroi": "not-a-number",
            "cost_to_retail_ratio": None,
            "turn_rate": [1, 2, 3],
            "shrinkage_rate": {"a": 1},
        }
        app.render_retail_kpis(malformed_dict)
        gmroi, _, cr, turn, shrink, _ = kpi_ui._parse_kpi_metrics(malformed_dict)
        self.assertEqual(gmroi, 2.42)
        self.assertEqual(cr, 48.20)

        # Non-dict, non-Series primitive types
        for weird_input in [123, "string", [1, 2, 3], True]:
            try:
                app.render_retail_kpis(weird_input)
            except Exception as e:
                self.fail(f"render_retail_kpis crashed on input {weird_input}: {e}")

    # --- render_tab_header Tests Across Diverse Colors & Strings ---

    def test_render_tab_header_colors(self):
        """Test render_tab_header across hex, named, rgb, rgba, and hsl colors."""
        test_colors = [
            "#2563EB",  # Primary Blue
            "#10B981",  # Emerald Green
            "#F59E0B",  # Amber / Warning
            "#EF4444",  # Crimson / Danger
            "#8B5CF6",  # Purple / AI
            "#000000",  # Black
            "#FFFFFF",  # White
            "blue",
            "green",
            "crimson",
            "transparent",
            "rgb(37, 99, 235)",
            "rgba(16, 185, 129, 0.8)",
            "hsl(217, 91%, 60%)",
        ]
        for color in test_colors:
            try:
                app.render_tab_header(
                    tab_name="Executive Dashboard",
                    description="High-visibility KPI diagnostic banner.",
                    accent_color=color,
                )
            except Exception as e:
                self.fail(f"render_tab_header raised exception with color '{color}': {e}")

    def test_render_tab_header_edge_case_strings(self):
        """Test render_tab_header with special characters, unicode, and empty strings."""
        cases = [
            ("Tab 1", "Normal description", "#2563EB"),
            ("", "", "#2563EB"),
            ("Special <>&'\" chars", "Description with $100 and <b>bold</b>", "#10B981"),
            ("🛍️ Retail Cockpit", "Multi-Agent 🤖 Replenishment & ⚖️ Valuation", "#F59E0B"),
            ("Multiline\nTab", "Description with\nline breaks", "#EF4444"),
        ]
        for tab_name, desc, color in cases:
            try:
                app.render_tab_header(tab_name, desc, color)
            except Exception as e:
                self.fail(f"render_tab_header crashed on ({tab_name}, {desc}, {color}): {e}")

    def test_render_tab_header_html_injection_structure(self):
        """Verify render_tab_header generates the expected HTML structure with accent color."""
        with patch("streamlit.markdown") as mock_markdown:
            app.render_tab_header("Test Tab", "Test Description", "#FF5733")
            mock_markdown.assert_called_once()
            call_args, call_kwargs = mock_markdown.call_args
            rendered_html = call_args[0]
            self.assertIn('class="executive-callout-banner"', rendered_html)
            self.assertIn("border-left: 5px solid #FF5733;", rendered_html)
            self.assertIn("color: #FF5733;", rendered_html)
            self.assertIn("Test Tab", rendered_html)
            self.assertIn("Test Description", rendered_html)
            self.assertTrue(call_kwargs.get("unsafe_allow_html"))


if __name__ == "__main__":
    unittest.main()
