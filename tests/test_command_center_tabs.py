"""
Unit and Integration Tests for Retail AI Pre-Sales Command Center Tabs:
- Tab 1: Strategic Account Discovery (tab_discovery.py)
- Tab 2: Autonomous Multi-Agent Replenishment Ops (tab_autogen.py)
- Tab 3: RIM Valuation Forensics & Predictive Store Clustering (tab_rim_knn.py)
- Main App Cockpit (app.py)
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

import tab_discovery
import tab_autogen
import tab_rim_knn
import app


class TestCommandCenterTabs(unittest.TestCase):
    """Test suite covering the unified 3-tab pre-sales command center."""

    # --------------------------------------------------------------------------
    # TAB 1: STRATEGIC ACCOUNT DISCOVERY TESTS
    # --------------------------------------------------------------------------
    def test_tab_discovery_presets(self):
        """Verify pre-configured enterprise prospect profiles."""
        presets = tab_discovery.PRESET_PROFILES
        self.assertIn("https://www.buckle.com", presets)
        self.assertIn("https://www.ralphlauren.com", presets)
        self.assertIn("https://www.lululemon.com", presets)

        for url, profile in presets.items():
            self.assertTrue(profile["account_name"])
            self.assertTrue(profile["retail_tier"])
            self.assertGreaterEqual(len(profile["pain_points"]), 3)
            self.assertIn("tech_stack", profile)
            self.assertIn("discovery_questions", profile)
            self.assertIn("gdocs_url", profile)

            # Check Value Triangle fields in pain points
            for pain in profile["pain_points"]:
                self.assertIn("category", pain)
                self.assertIn("technical_gap", pain)
                self.assertIn("operational_friction", pain)
                self.assertIn("financial_impact", pain)

    def test_tab_discovery_generic_synthesis(self):
        """Verify dynamic profile generation for arbitrary retail domains."""
        custom_url = "https://www.nordstrom.com"
        profile = tab_discovery.build_generic_prospect_profile(
            domain=custom_url, retail_tier="Regional Department Store Tier 2"
        )
        self.assertEqual(profile["domain"], custom_url)
        self.assertEqual(profile["retail_tier"], "Regional Department Store Tier 2")
        self.assertGreaterEqual(len(profile["pain_points"]), 3)
        self.assertGreaterEqual(len(profile["discovery_questions"]), 3)

    # --------------------------------------------------------------------------
    # TAB 2: AUTONOMOUS MULTI-AGENT OPS TESTS
    # --------------------------------------------------------------------------
    def test_tab_autogen_scenarios(self):
        """Verify disruption scenario profiles and persona configurations."""
        scenarios = tab_autogen.SCENARIOS
        required_scenarios = ["standard", "port_strike", "viral_spike", "ice_storm"]
        for s_id in required_scenarios:
            self.assertIn(s_id, scenarios)
            sc = scenarios[s_id]
            self.assertTrue(sc["title"])
            self.assertTrue(sc["description"])
            self.assertIn("metrics", sc)
            self.assertIn("messages", sc)

            # Validate consensus resolution in messages
            has_consensus = any(tab_autogen.CONSENSUS_TAG in m.get("content", "") for m in sc["messages"])
            self.assertTrue(has_consensus, f"Scenario {s_id} must have a resolution consensus block")

            # Validate metrics structure
            m = sc["metrics"]
            self.assertGreater(m["allocated_units"], 0)
            self.assertGreaterEqual(m["transfer_units"], 0)
            self.assertGreaterEqual(m["vendor_units"], 0)
            self.assertEqual(m["allocated_units"], m["transfer_units"] + m["vendor_units"])

    def test_tab_autogen_tool_traces(self):
        """Verify that agent messages contain ERP tool execution traces."""
        standard_sc = tab_autogen.SCENARIOS["standard"]
        tool_messages = [m for m in standard_sc["messages"] if "tool_call" in m]
        self.assertGreaterEqual(len(tool_messages), 2)
        tools_called = [m["tool_call"]["tool"] for m in tool_messages]
        self.assertIn("validate_stock_levels", tools_called)
        self.assertIn("generate_purchase_order", tools_called)

    def test_tab_autogen_hitl_approval_scenarios(self):
        """Verify that HITL stress scenarios have required gate flags, pending/rejected states, and approval branches."""
        scenarios = tab_autogen.SCENARIOS
        hitl_scenarios = ["viral_spike", "air_charter"]
        for s_id in hitl_scenarios:
            self.assertIn(s_id, scenarios)
            sc = scenarios[s_id]
            self.assertTrue(sc.get("requires_hitl", False), f"{s_id} must flag requires_hitl=True")
            self.assertIn("pending_metrics", sc)
            self.assertIn("rejected_metrics", sc)
            self.assertIn("rejected_messages", sc)

            # Metrics verification
            self.assertGreater(sc["metrics"]["total_commitment"], 25000.0)
            self.assertLessEqual(sc["rejected_metrics"]["total_commitment"], 25000.0)

            # Consensus verification in rejected branch
            has_rejected_consensus = any(tab_autogen.CONSENSUS_TAG in m.get("content", "") for m in sc["rejected_messages"])
            self.assertTrue(has_rejected_consensus, f"Scenario {s_id} must have resolution consensus in rejected branch")

    def test_tab_autogen_capped_resolution(self):
        """Verify dynamic capped resolution generates adjusted messages, vendor PO, and metrics."""
        for s_id in ["viral_spike", "air_charter"]:
            sc = tab_autogen.SCENARIOS[s_id]
            cap_val = 28000
            msgs, metrics = tab_autogen.generate_capped_resolution(
                scenario_data=sc,
                cap_val=cap_val,
                target_store="Store #104 (Denver Downtown)",
                sku="SKU-4092 (Vintage Stretch Denim, 32x32)",
            )
            # Verify message count and structure
            self.assertEqual(len(msgs), 6)  # 3 pre-gate messages + 3 post-resolution messages
            self.assertIn("Consultant_Approver", [m["sender"] for m in msgs])
            self.assertIn("Admin_Executor", [m["sender"] for m in msgs])
            self.assertIn("Inventory_Merchandising_Lead", [m["sender"] for m in msgs])

            # Verify capped metrics
            self.assertLessEqual(metrics["total_commitment"], cap_val)
            self.assertIn("HITL CAPPED", metrics["governance_badge"])
            self.assertGreater(metrics["vendor_units"], 0)
            self.assertEqual(metrics["allocated_units"], metrics["transfer_units"] + metrics["vendor_units"])

            # Verify EDI purchase order trace
            po_msg = [m for m in msgs if m.get("sender") == "Admin_Executor" and "tool_call" in m][0]
            self.assertEqual(po_msg["tool_call"]["tool"], "generate_purchase_order")
            self.assertTrue(po_msg["tool_call"]["args"]["consultant_approved"])
            self.assertEqual(po_msg["tool_call"]["args"]["authorized_cap"], cap_val)


    # --------------------------------------------------------------------------
    # TAB 3: RIM VALUATION FORENSICS & CLUSTERING TESTS
    # --------------------------------------------------------------------------
    def test_tab_rim_knn_pipeline(self):
        """Verify clustering pipeline loads, finds peer groups, and returns 3 neighbors."""
        store_summary_df, scaler, kmeans, knn_engine = tab_rim_knn.load_or_train_clustering_pipeline()
        self.assertGreaterEqual(len(store_summary_df), 10)
        self.assertIn("STORE_104", store_summary_df["store_id"].values)

        peer_data = knn_engine.find_peer_group("STORE_104")
        self.assertEqual(peer_data["target_store_id"], "STORE_104")
        self.assertEqual(len(peer_data["nearest_peers"]), 3)

        # Check distances are sorted ascending
        distances = [p["euclidean_distance"] for p in peer_data["nearest_peers"]]
        self.assertEqual(distances, sorted(distances))

    def test_tab_rim_markdown_shock_simulation(self):
        """Verify 12-period forward projection under markdown shock."""
        df_raw = tab_rim_knn.load_apparel_rim_dataset()
        store_records = df_raw[df_raw["store_id"] == "STORE_104"].copy()

        df_proj = tab_rim_knn.project_forward_12_periods(
            baseline_store_df=store_records,
            markdown_uplift_pct=0.25,
            sales_lift_elasticity=0.35,
        )
        self.assertEqual(len(df_proj), 12)
        self.assertIn("planned_ending_cost", df_proj.columns)
        self.assertIn("shocked_ending_cost", df_proj.columns)
        self.assertIn("asset_deflation", df_proj.columns)

        # Ensure cumulative asset deflation occurs
        final_plan_cost = df_proj["planned_ending_cost"].iloc[-1]
        final_shock_cost = df_proj["shocked_ending_cost"].iloc[-1]
        self.assertGreater(final_plan_cost, final_shock_cost)

    def test_tab_rim_ai_cfo_brief_fallback(self):
        """Verify AI CFO risk brief outputs 3 structured recommendations without crashing."""
        brief = tab_rim_knn.generate_ai_cfo_brief_text(
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
        self.assertTrue(len(brief) > 100)
        self.assertIn("Balance Sheet Asset Deflation", brief)
        self.assertIn("Guardrails", brief)

    # --------------------------------------------------------------------------
    # THEMED EXECUTIVE BANNERS & RETAIL KPI CARDS TESTS
    # --------------------------------------------------------------------------
    def test_render_tab_header_execution(self):
        """Verify render_tab_header executes cleanly across all accent themes."""
        app.render_tab_header("Test Tab 1", "🎯 Executive Objective: Test 1", "#2563EB")
        app.render_tab_header("Test Tab 2", "🤖 Executive Objective: Test 2", "#10B981")
        app.render_tab_header("Test Tab 3", "⚖️ Executive Objective: Test 3", "#F59E0B")

    def test_render_retail_kpis_variations(self):
        """Verify render_retail_kpis handles default None, dict, Series, and DataFrame inputs."""
        # 1. Default None baseline
        app.render_retail_kpis()

        # 2. Custom dictionary input
        app.render_retail_kpis({
            "gmroi": 2.85,
            "cost_to_retail_ratio": 0.4650,
            "turn_rate": 4.3,
            "shrinkage_rate": 0.0150,
        })

        # 3. Pandas Series input
        series_input = pd.Series({
            "gmroi": 1.75,  # below benchmark to test inverse delta color
            "cost_to_retail_ratio": 0.5200,
            "sell_through_velocity": 0.35,
            "shrinkage_rate": 0.0380,  # above threshold to test inverse delta color
        })
        app.render_retail_kpis(series_input)

        # 4. Pandas DataFrame input
        df_input = pd.DataFrame([series_input.to_dict()])
        app.render_retail_kpis(df_input)


if __name__ == "__main__":
    unittest.main()
