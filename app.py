"""Retail AI Pre-Sales Command Center: Autonomous Merchandising & Valuation Forensics (<100 LOC)."""
from __future__ import annotations
from pathlib import Path
import sys
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from services.data_loaders import load_styles
from ui.tab_discovery import render_tab_discovery
from ui.tab_autogen import render_tab_autogen
from ui.tab_rim_knn import render_tab_rim_knn
from ui.kpi_metrics import render_retail_kpis as _render_kpis

st.set_page_config(
    page_title="Retail AI Pre-Sales Command Center",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_styles = load_styles()
if _styles:
    st.markdown(f"<style>{_styles}</style>", unsafe_allow_html=True)


def render_retail_kpis(df_store_metrics: Any = None, *args: Any, **kwargs: Any) -> None:
    """Renders top-level KPI metrics cards."""
    _render_kpis(df_store_metrics)


def render_global_header() -> None:
    """Renders the executive command center header banner with live status chips."""
    st.markdown(
        '<div class="command-header-container">'
        '<div class="command-header-title">🛍️ Retail AI Pre-Sales Command Center: Autonomous Merchandising & Valuation Forensics</div>'
        '<div class="command-header-subtitle">Unified Enterprise Cockpit Integrating Account Discovery, Multi-Agent Replenishment Negotiation, and Balance Sheet Valuation Forensics.</div>'
        '<div class="status-chip-container">'
        '<span class="status-chip chip-green">🟢 Gemini 2.5 Flash / Pro: Active</span>'
        '<span class="status-chip chip-blue">🤖 AutoGen v0.2: Connected</span>'
        '<span class="status-chip chip-purple">⚙️ ERP Adapter: Mock EDI 850</span>'
        '<span class="status-chip chip-amber">📊 RIM Engine: 50-Store / 24-Month Roll-Forward</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )


def render_tab_header(tab_name: str, description: str, accent_color: str = "#2563eb") -> None:
    """Renders a visually distinct executive callout banner for a dashboard tab."""
    st.markdown(
        f'<div class="executive-callout-banner" style="border-left: 5px solid {accent_color};">'
        f'<div class="banner-title" style="color: {accent_color};">{tab_name}</div>'
        f'<div class="banner-description">{description}</div></div>',
        unsafe_allow_html=True,
    )


def main() -> None:
    """Coordinates top-level tab views across all three pre-sales phases."""
    render_global_header()
    tab1, tab2, tab3 = st.tabs([
        "🏢 Tab 1: Strategic Account Discovery & Google Docs Export",
        "🤖 Tab 2: Autonomous Multi-Agent Replenishment Ops",
        "📊 Tab 3: RIM Valuation Forensics & Predictive Store Clustering",
    ])
    with tab1:
        render_tab_header(
            "🏢 Strategic Account Discovery & Pre-Sales Intelligence Brief",
            "🎯 <strong>Executive Objective</strong>: Programmatically crawl target apparel domains, isolate ERP/POS architectural friction, and auto-generate an executive pre-sales brief directly into Google Docs in under 60 seconds.",
            "#2563EB",
        )
        render_tab_discovery(show_header=False)
    with tab2:
        render_tab_header(
            "🤖 Autonomous Multi-Agent Replenishment Ops (AutoGen)",
            "🤖 <strong>Executive Objective</strong>: Orchestrate multi-agent autonomous inventory negotiations using Microsoft AutoGen, resolving promotional denim stockouts with audited ERP purchase order tool execution.",
            "#10B981",
        )
        render_tab_autogen(show_header=False)
    with tab3:
        render_tab_header(
            "📊 RIM Valuation Forensics & Predictive Store Clustering",
            "⚖️ <strong>Executive Objective</strong>: Expose the structural flaw of the Retail Inventory Method (RIM) where clearance markdowns deflate balance sheet inventory cost valuations, paired with Scikit-learn KNN store clustering and Gemini Pro CFO risk briefings.",
            "#F59E0B",
        )
        render_tab_rim_knn(show_header=False, kpi_renderer=render_retail_kpis)


if __name__ == "__main__":
    main()
