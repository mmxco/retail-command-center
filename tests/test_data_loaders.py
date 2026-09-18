"""
tests/test_data_loaders.py
Unit tests for data loaders, externalized JSON fixtures, and stylesheet assets (Milestone 5 / R3 / R5).
Verifies:
- data/discovery_presets.json validity, schema, and Value Triangle consistency
- data/autogen_scenarios.json validity, schema, HITL flags, and metrics conservation
- @st.cache_data caching behavior for load_discovery_presets and load_autogen_scenarios
- Missing-file fallback resilience without uncaught exceptions
- assets/styles.css existence, non-empty size (>5KB), and critical CSS class presence
"""

import json
import unittest
from pathlib import Path
from unittest.mock import patch

# Test loader imports (supporting both direct services/ui imports and root re-exports)
try:
    from services.data_loaders import (
        load_discovery_presets,
        load_autogen_scenarios,
        load_styles,
    )
except ImportError:
    from tab_discovery import load_discovery_presets  # type: ignore
    from tab_autogen import load_autogen_scenarios      # type: ignore
    try:
        from app import load_styles                     # type: ignore
    except ImportError:
        def load_styles(path="assets/styles.css"):
            p = Path(path)
            if p.exists():
                return p.read_text(encoding="utf-8")
            return ""


class TestDataLoadersAndAssets(unittest.TestCase):
    """Test suite for externalized JSON datasets, styles, and cached loaders."""

    @classmethod
    def setUpClass(cls):
        cls.root_dir = Path(__file__).resolve().parent.parent
        cls.presets_json_path = cls.root_dir / "data" / "discovery_presets.json"
        cls.scenarios_json_path = cls.root_dir / "data" / "autogen_scenarios.json"
        cls.styles_css_path = cls.root_dir / "assets" / "styles.css"

    # --------------------------------------------------------------------------
    # 1. DISCOVERY PRESETS JSON VALIDITY & SCHEMA
    # --------------------------------------------------------------------------
    def test_discovery_presets_json_validity(self):
        """Verify data/discovery_presets.json exists, is valid JSON, and matches schema."""
        self.assertTrue(self.presets_json_path.exists(), "data/discovery_presets.json must exist")
        
        with open(self.presets_json_path, "r", encoding="utf-8") as f:
            presets = json.load(f)

        self.assertIsInstance(presets, dict)
        required_urls = [
            "https://www.buckle.com",
            "https://www.ralphlauren.com",
            "https://www.lululemon.com",
        ]
        for url in required_urls:
            self.assertIn(url, presets, f"Missing required prospect URL: {url}")
            profile = presets[url]
            self.assertTrue(profile.get("account_name"), "account_name must be non-empty")
            self.assertTrue(profile.get("retail_tier"), "retail_tier must be non-empty")
            self.assertIn("tech_stack", profile)
            self.assertIn("discovery_questions", profile)
            self.assertIn("gdocs_url", profile)
            
            # Verify Value Triangle in pain points
            pain_points = profile.get("pain_points", [])
            self.assertGreaterEqual(len(pain_points), 3, "Each preset must have at least 3 pain points")
            for pain in pain_points:
                self.assertIn("category", pain)
                self.assertIn("technical_gap", pain)
                self.assertIn("operational_friction", pain)
                self.assertIn("financial_impact", pain)

    # --------------------------------------------------------------------------
    # 2. AUTOGEN SCENARIOS JSON VALIDITY & SCHEMA
    # --------------------------------------------------------------------------
    def test_autogen_scenarios_json_validity(self):
        """Verify data/autogen_scenarios.json exists, is valid JSON, and preserves metrics."""
        self.assertTrue(self.scenarios_json_path.exists(), "data/autogen_scenarios.json must exist")

        with open(self.scenarios_json_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

        self.assertIsInstance(scenarios, dict)
        required_scenarios = ["standard", "port_strike", "viral_spike", "air_charter", "ice_storm"]
        for s_id in required_scenarios:
            self.assertIn(s_id, scenarios, f"Missing scenario: {s_id}")
            sc = scenarios[s_id]
            self.assertTrue(sc.get("title"), f"Scenario {s_id} must have a title")
            self.assertTrue(sc.get("description"), f"Scenario {s_id} must have a description")
            self.assertIn("metrics", sc, f"Scenario {s_id} must have metrics")
            self.assertIn("messages", sc, f"Scenario {s_id} must have messages")

            # Validate inventory conservation equation
            m = sc["metrics"]
            self.assertGreater(m["allocated_units"], 0)
            self.assertEqual(
                m["allocated_units"],
                m["transfer_units"] + m["vendor_units"],
                f"Scenario {s_id} violates inventory conservation equation"
            )

        # Validate HITL scenarios
        for hitl_id in ["viral_spike", "air_charter"]:
            sc = scenarios[hitl_id]
            self.assertTrue(sc.get("requires_hitl", False), f"{hitl_id} must flag requires_hitl=True")
            self.assertIn("pending_metrics", sc)
            self.assertIn("rejected_metrics", sc)
            self.assertIn("rejected_messages", sc)
            self.assertGreater(sc["metrics"]["total_commitment"], 25000.0)
            self.assertLessEqual(sc["rejected_metrics"]["total_commitment"], 25000.0)

    # --------------------------------------------------------------------------
    # 3. CACHED LOADER & RESILIENCE TESTS
    # --------------------------------------------------------------------------
    def test_discovery_presets_loader_caching(self):
        """Verify load_discovery_presets loads data and is decorated with @st.cache_data."""
        data_1 = load_discovery_presets()
        data_2 = load_discovery_presets()
        self.assertIsInstance(data_1, dict)
        self.assertEqual(len(data_1), 3)
        self.assertEqual(data_1, data_2)
        # Check Streamlit cache wrapper attribute
        is_cached = hasattr(load_discovery_presets, "__wrapped__") or hasattr(load_discovery_presets, "clear")
        self.assertTrue(is_cached, "load_discovery_presets must be decorated with @st.cache_data")

    def test_discovery_presets_loader_missing_file_fallback(self):
        """Verify load_discovery_presets returns fallback dictionary without crashing on missing file."""
        non_existent = self.root_dir / "data" / "non_existent_presets.json"
        try:
            result = load_discovery_presets(filepath=non_existent)
            self.assertIsInstance(result, dict)
        except TypeError:
            # Fallback if function doesn't accept filepath arg: mock open
            with patch("builtins.open", side_effect=FileNotFoundError):
                # Clear cache if wrapped
                if hasattr(load_discovery_presets, "clear"):
                    load_discovery_presets.clear()
                result = load_discovery_presets()
                self.assertIsInstance(result, dict)

    def test_autogen_scenarios_loader_caching(self):
        """Verify load_autogen_scenarios loads scenarios and is cached."""
        data_1 = load_autogen_scenarios()
        data_2 = load_autogen_scenarios()
        self.assertIsInstance(data_1, dict)
        self.assertEqual(len(data_1), 5)
        self.assertEqual(data_1, data_2)
        is_cached = hasattr(load_autogen_scenarios, "__wrapped__") or hasattr(load_autogen_scenarios, "clear")
        self.assertTrue(is_cached, "load_autogen_scenarios must be decorated with @st.cache_data")

    def test_autogen_scenarios_loader_missing_file_fallback(self):
        """Verify load_autogen_scenarios returns fallback dictionary without crashing on missing file."""
        non_existent = self.root_dir / "data" / "non_existent_scenarios.json"
        try:
            result = load_autogen_scenarios(filepath=non_existent)
            self.assertIsInstance(result, dict)
        except TypeError:
            with patch("builtins.open", side_effect=FileNotFoundError):
                if hasattr(load_autogen_scenarios, "clear"):
                    load_autogen_scenarios.clear()
                result = load_autogen_scenarios()
                self.assertIsInstance(result, dict)

    def test_autogen_scenarios_dollar_sanitization(self):
        """Verify that load_autogen_scenarios returns messages with sanitized currency figures."""
        scenarios = load_autogen_scenarios()
        for s_id, sc in scenarios.items():
            for m in sc.get("messages", []):
                content = m.get("content", "")
                if "$25,000" in content:
                    self.assertIn(r"\$25,000", content, f"Unescaped currency dollar in scenario {s_id}")

    # --------------------------------------------------------------------------
    # 4. STYLESHEET INTEGRITY & SIZE TESTS
    # --------------------------------------------------------------------------
    def test_styles_css_existence_and_size(self):
        """Verify assets/styles.css exists, is >5KB, and contains required CSS classes."""
        self.assertTrue(self.styles_css_path.exists(), "assets/styles.css must exist")
        css_content = self.styles_css_path.read_text(encoding="utf-8")
        
        # Original embedded CSS was ~7KB (218 LOC)
        self.assertGreater(len(css_content), 5000, "assets/styles.css must be at least 5KB")
        
        # Verify critical styling selectors
        required_classes = [
            ".stMetric",
            ".retail-kpi-card",
            ".command-header-container",
            ".katex",
            ".block-container",
        ]
        for cls in required_classes:
            self.assertIn(cls, css_content, f"Missing critical CSS selector: {cls}")

    def test_styles_css_loader_resilience(self):
        """Verify load_styles executes without crashing even if CSS file is missing."""
        try:
            # Test normal execution
            load_styles(str(self.styles_css_path))
            # Test missing file resilience
            load_styles(str(self.root_dir / "assets" / "missing.css"))
        except Exception as e:
            self.fail(f"load_styles raised unexpected exception on missing file: {e}")


if __name__ == "__main__":
    unittest.main()
