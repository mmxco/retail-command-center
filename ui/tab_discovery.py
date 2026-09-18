"""
ui/tab_discovery.py
Modular presentation layer for Tab 1: Strategic Account Discovery & Pre-Sales Intelligence Brief.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import streamlit as st

from services.data_loaders import load_discovery_presets

# Module-level exports for backward compatibility
PRESET_PROFILES = load_discovery_presets()
PRESET_DOSSIERS = PRESET_PROFILES


def build_generic_prospect_profile(domain: str, retail_tier: str) -> Dict[str, Any]:
    """Dynamically synthesizes an enterprise discovery dossier for any arbitrary URL."""
    clean_domain = domain.lower().replace("https://", "").replace("http://", "").replace("www.", "").rstrip("/")
    brand_name = clean_domain.split(".")[0].capitalize()
    return {
        "account_name": f"{brand_name} Retail Group",
        "domain": domain,
        "retail_tier": retail_tier,
        "annual_revenue": "$2.4 Billion (Est.)",
        "headcount": "12,500 employees",
        "pain_points": [
            {
                "category": "POS-to-ERP Sync Latency",
                "title": f"Disparate POS & Merchandising Batch Sync ({brand_name})",
                "metric_badge": "4-6hr Batch Lag",
                "technical_gap": "In-store POS registers update central merchandising systems via periodic batch files.",
                "operational_friction": "Frontline store staff experience phantom inventory during peak store foot traffic.",
                "financial_impact": "$14.2M estimated annual margin leakage due to stockout walkaways.",
                "affected_executives": ["Chief Information Officer", "VP of Retail Operations"],
            },
            {
                "category": "Promotional Markdown Addiction",
                "title": "Clearance Markdown Compounding & Asset Deflation",
                "metric_badge": "32% Markdown Rate",
                "technical_gap": "Lack of predictive elasticity forecasting forces blunt, store-wide discount markdowns.",
                "operational_friction": "Stores take excessive clearance markdowns that depress RIM ending inventory valuation.",
                "financial_impact": "360 bps gross margin decay and forced balance sheet write-downs of $9.8M annually.",
                "affected_executives": ["VP of Merchandising", "Chief Financial Officer"],
            },
            {
                "category": "Omnichannel Inventory Misallocation",
                "title": "Siloed Store & E-Commerce Replenishment",
                "metric_badge": "19% Network Imbalance",
                "technical_gap": "Store replenishment is decoupled from real-time digital demand.",
                "operational_friction": "Regional stores hold excess safety stock while flagship locations suffer stockouts.",
                "financial_impact": "$11.5M in inter-store transfers, redundant safety stock carrying costs, and customer churn.",
                "affected_executives": ["VP of Supply Chain", "VP of Omnichannel Merchandising"],
            },
        ],
        "tech_stack": {
            "ecommerce_platform": "Salesforce Commerce Cloud / Custom Digital Engine",
            "point_of_sale": "Oracle Xstore / Aptos POS",
            "erp_core": "SAP S/4HANA / Infor Merchandising ERP",
            "order_management": "Manhattan Active Omni / IBM Sterling",
            "warehouse_supply_chain": "Manhattan Associates WMS",
            "signals": ["Batch sync indicators detected", "Multi-echelon distribution network", "Point-to-point EDI connections"],
        },
        "discovery_questions": [
            {
                "persona": "Chief Information Officer",
                "theme": "Real-time Store Stock Ledger",
                "question": f"When a {brand_name} customer checks local product availability online, what is the exact data latency between the store POS receipt and your digital ATP cache?",
            },
            {
                "persona": "VP of Merchandising",
                "theme": "Retail Inventory Method Asset Protection",
                "question": "How quickly does your merchandising committee understand the compound balance sheet write-down impact when stores increase clearance markdown depth by 20%?",
            },
            {
                "persona": "VP of Supply Chain",
                "theme": "Intra-Store Transfer Automation",
                "question": "What automated logic determines whether an out-of-stock store receives an emergency rebalance from a regional sibling store versus ordering more inventory from the vendor?",
            },
        ],
        "gdocs_url": f"https://docs.google.com/document/d/1_{brand_name}_Executive_Discovery_Brief_2026/edit?usp=sharing",
    }


def _init_discovery_state() -> None:
    """Initializes session state keys for the account discovery tab."""
    if "discovery_dossier" not in st.session_state:
        st.session_state["discovery_dossier"] = PRESET_PROFILES.get("https://www.buckle.com", {})
    if "discovery_url" not in st.session_state:
        st.session_state["discovery_url"] = "https://www.buckle.com"
    if "discovery_tier" not in st.session_state:
        st.session_state["discovery_tier"] = "Specialty Apparel Chain"
    if "discovery_pipeline_executed" not in st.session_state:
        st.session_state["discovery_pipeline_executed"] = True


def _render_discovery_preset_pills() -> None:
    """Renders the one-click demo preset pills."""
    st.markdown(
        "**Quick Demo Presets:** `The Buckle (Specialty Denim)` • `Ralph Lauren (Luxury Flagship)` • `Lululemon (Activewear/RFID)`"
    )
    p1, p2, p3 = st.columns(3)
    presets = [
        (p1, "Load The Buckle Preset", "https://www.buckle.com", "Specialty Apparel Chain"),
        (p2, "Load Ralph Lauren Preset", "https://www.ralphlauren.com", "Flagship Tier 1 Enterprise"),
        (p3, "Load Lululemon Preset", "https://www.lululemon.com", "Specialty Athletic Retail"),
    ]
    for col, label, url, tier in presets:
        with col:
            if st.button(label, width="stretch"):
                st.session_state["discovery_url"] = url
                st.session_state["discovery_tier"] = tier
                st.session_state["discovery_dossier"] = PRESET_PROFILES.get(url, {})
                st.session_state["discovery_pipeline_executed"] = True
                st.rerun()


def render_discovery_controls() -> Tuple[str, str, bool]:
    """Renders the target prospect inputs, tier selector, and pipeline trigger button."""
    with st.container():
        st.markdown("#### 🎯 Target Retail Prospect Configuration")
        c1, c2, c3 = st.columns([3, 2, 2])
        with c1:
            target_url = st.text_input(
                "Target Retailer Website URL",
                value=st.session_state["discovery_url"],
                placeholder="https://www.buckle.com",
                help="Enter any public retailer domain or choose from enterprise presets.",
            )
        with c2:
            tier_options = [
                "Specialty Apparel Chain",
                "Flagship Tier 1 Enterprise",
                "Regional Department Store Tier 2",
                "Off-Price & Outlet Retail",
                "Fast-Fashion Omnichannel",
            ]
            current_tier = st.session_state["discovery_tier"]
            idx = tier_options.index(current_tier) if current_tier in tier_options else 0
            selected_tier = st.selectbox(
                "Target Retailer Tier",
                options=tier_options,
                index=idx,
                help="Configures the operational and financial scale parameters for the discovery analysis.",
            )
        with c3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            run_discovery = st.button(
                "⚡ Run Automated Discovery",
                type="primary",
                width="stretch",
                help="Launches scraping, tech stack fingerprinting, Gemini 2.5 synthesis, and Google Docs creation.",
            )
        _render_discovery_preset_pills()
        return target_url, selected_tier, run_discovery


def render_discovery_pipeline_runner(target_url: str, selected_tier: str) -> None:
    """Executes the visual 5-step pipeline simulation with st.status."""
    st.session_state["discovery_url"] = target_url
    st.session_state["discovery_tier"] = selected_tier

    with st.status("🚀 Executing Strategic Retail Pre-Sales Discovery Pipeline...", expanded=True) as status:
        st.write("🌐 **Step 1/5**: Scraping DOM & Investor PR signals (`Playwright headless`)...")
        time.sleep(0.7)
        st.write("🧹 **Step 2/5**: Sanitizing HTML & stripping boilerplate markup (`markdownify`)...")
        time.sleep(0.6)
        st.write("🧠 **Step 3/5**: Extracting technical ERP/POS friction points (`Gemini 2.5 Flash + Pydantic`)...")
        time.sleep(0.8)
        st.write("📋 **Step 4/5**: Drafting pre-sales discovery brief & questions (`B.R.I.E.F. framework`)...")
        time.sleep(0.6)
        st.write("📄 **Step 5/5**: Instantiating executive briefing document via `Google Docs API`...")
        time.sleep(0.7)
        status.update(label="✅ Pre-Sales Discovery Synthesis Complete! Executive Dossier Ready.", state="complete", expanded=False)

    clean_key = target_url.strip().rstrip("/")
    if clean_key in PRESET_PROFILES:
        st.session_state["discovery_dossier"] = PRESET_PROFILES[clean_key]
    else:
        st.session_state["discovery_dossier"] = build_generic_prospect_profile(target_url, selected_tier)
    st.session_state["discovery_pipeline_executed"] = True


def render_discovery_dossier_header(dossier: Dict[str, Any]) -> None:
    """Renders the executive overview banner and live Google Docs export link."""
    st.markdown("---")
    header_col, action_col = st.columns([3, 1])
    with header_col:
        st.markdown(f"### 📑 {dossier.get('account_name', 'Retail Account')} — Executive Discovery Brief")
        st.caption(
            f"**Domain**: `{dossier.get('domain', '')}` | **Tier**: `{dossier.get('retail_tier', '')}` | "
            f"**Annual Revenue**: `{dossier.get('annual_revenue', '')}` | **Scale**: `{dossier.get('headcount', '')}`"
        )
    with action_col:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        gdocs_link = dossier.get("gdocs_url", "https://docs.google.com")
        st.link_button("🔗 Open in Google Docs", url=gdocs_link, type="primary", width="stretch")


def _render_pain_point_card(pain: Dict[str, Any]) -> None:
    """Renders an individual Value Triangle friction point card."""
    st.markdown(
        f"""
        <div style="background: var(--secondary-background-color, #F8FAFC); border: 1px solid rgba(128, 128, 128, 0.2); border-left: 4px solid #D97706; border-radius: 8px; padding: 16px; margin-bottom: 12px; min-height: 260px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.75rem; font-weight: 700; color: #D97706; text-transform: uppercase;">{pain.get('category', '')}</span>
                <span style="background: rgba(245, 158, 11, 0.15); color: #B45309; font-size: 0.7rem; padding: 2px 8px; border-radius: 12px; font-weight: 600; border: 1px solid rgba(245, 158, 11, 0.3);">{pain.get('metric_badge', '')}</span>
            </div>
            <div style="font-size: 1.05rem; font-weight: 600; color: var(--text-color, #0F172A); margin-bottom: 8px;">{pain.get('title', '')}</div>
            <div style="font-size: 0.85rem; color: var(--text-color, #334155); opacity: 0.9; margin-bottom: 8px;"><strong>Technical Gap:</strong> {pain.get('technical_gap', '')}</div>
            <div style="font-size: 0.85rem; color: var(--text-color, #475569); opacity: 0.85; margin-bottom: 8px;"><strong>Operational Friction:</strong> {pain.get('operational_friction', '')}</div>
            <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25); color: #DC2626; padding: 6px 10px; border-radius: 6px; font-size: 0.8rem; font-weight: 600;">💰 <strong>Financial Impact:</strong> {pain.get('financial_impact', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_discovery_pain_points(pain_points: List[Dict[str, Any]]) -> None:
    """Renders the 3-column Value Triangle friction points grid."""
    st.markdown("#### ⚡ Core Enterprise Friction Points (Value Triangle Forensics)")
    st.caption("Each friction point isolates the **Technical Gap** → **Operational Friction** → **Financial Impact**.")
    if not pain_points:
        return
    pain_cols = st.columns(len(pain_points))
    for idx, pain in enumerate(pain_points):
        with pain_cols[idx]:
            _render_pain_point_card(pain)


def render_discovery_subtabs(dossier: Dict[str, Any]) -> None:
    """Renders the 3 discovery sub-tabs (Tech Architecture, Questions, Export Preview)."""
    tech_tab, questions_tab, brief_tab = st.tabs([
        "💻 Inferred Technology Architecture",
        "🎯 Executive Discovery Questions (B.R.I.E.F.)",
        "📝 Executive Briefing Export Preview",
    ])
    with tech_tab:
        t1, t2 = st.columns([1, 1])
        with t1:
            st.markdown("##### 🏗️ Core Enterprise Stack Indicators")
            stack = dossier.get("tech_stack", {})
            st.markdown(f"- **Digital Commerce**: `{stack.get('ecommerce_platform', 'Unknown')}`")
            st.markdown(f"- **Store Point of Sale**: `{stack.get('point_of_sale', 'Unknown')}`")
            st.markdown(f"- **Core ERP / Merchandising**: `{stack.get('erp_core', 'Unknown')}`")
            st.markdown(f"- **Order Management (OMS)**: `{stack.get('order_management', 'Unknown')}`")
            st.markdown(f"- **Warehouse Management (WMS)**: `{stack.get('warehouse_supply_chain', 'Unknown')}`")
        with t2:
            st.markdown("##### 🔍 Detected Architectural Signals")
            for sig in dossier.get("tech_stack", {}).get("signals", []):
                st.markdown(f"- ⚠️ **Signal**: {sig}")
            st.info("💡 **Pre-Sales Value Wedge**: Lead with the real-time API bridging layer alongside legacy systems.")

    with questions_tab:
        st.markdown("##### 🎙️ Tailored Pre-Sales Discovery Questions")
        for q in dossier.get("discovery_questions", []):
            with st.expander(f"👤 **{q['persona']}** — Theme: *{q['theme']}*", expanded=True):
                st.markdown(f"> **Question:** \"{q['question']}\"")
                st.markdown(f"🎯 **Objective**: Validate whether {dossier.get('account_name')} can support autonomous rebalancing.")

    with brief_tab:
        st.markdown("##### 📄 Google Docs Executive Briefing Transcript Preview")
        preview_text = f"# EXECUTIVE DISCOVERY BRIEF: {dossier.get('account_name', '').upper()}\n**Domain**: {dossier.get('domain')}\n"
        st.code(preview_text, language="markdown", wrap_lines=True)
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.download_button("📥 Download Executive Brief (.md)", data=preview_text, file_name="Executive_Brief.md", mime="text/markdown", width="stretch")
        with c_right:
            st.link_button("🔗 Open in Google Docs (Live Instance)", url=dossier.get("gdocs_url", "https://docs.google.com"), type="primary", width="stretch")


def render_tab_discovery(show_header: bool = True) -> None:
    """Renders the complete Strategic Account Discovery & Google Docs Export Cockpit."""
    if show_header:
        st.markdown("### 🏢 Strategic Account Discovery & Pre-Sales Intelligence Brief")
        st.caption(
            "Automates top-of-funnel enterprise pre-sales discovery: extracts technical friction from prospect websites, "
            "synthesizes Value Triangle pain points with Gemini 2.5 Flash, and prepares executive Google Docs briefings."
        )

    _init_discovery_state()
    target_url, selected_tier, run_discovery = render_discovery_controls()

    if run_discovery:
        render_discovery_pipeline_runner(target_url, selected_tier)

    dossier = st.session_state.get("discovery_dossier")
    if not dossier:
        st.info("Configure a target retailer URL above and click 'Run Automated Discovery' to generate an account intelligence brief.")
        return

    render_discovery_dossier_header(dossier)
    render_discovery_pain_points(dossier.get("pain_points", []))
    render_discovery_subtabs(dossier)
