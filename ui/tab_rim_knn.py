"""
ui/tab_rim_knn.py
Modular presentation layer for Tab 3: RIM Valuation Forensics & Predictive Store Clustering.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import pandas as pd
import streamlit as st

from core.store_clustering import (
    ARCHETYPE_LABELS,
    FEATURE_COLS,
    StoreBenchmarkKNN,
    load_and_aggregate_store_data,
    train_clustering_pipeline,
)
from valuation_charts import (
    project_forward_12_periods,
    create_synchronized_valuation_figure,
)
from services.llm_advisor import (
    RIMDashboardState,
    generate_cfo_risk_brief,
    generate_ai_cfo_brief_text as _services_generate_cfo_brief,
)
from services.formatters import normalize_cfo_markdown
from ui.value_calculator import render_value_calculator_tab

# Archetype Color Badges
ARCHETYPE_COLORS = {
    "Capital-Efficient Flagships": "#10B981",
    "Balanced Regional Performers": "#38BDF8",
    "Discount-Addicted Outliers": "#F59E0B",
    "High-Risk Shrinkage Anomalies": "#EF4444",
}


@st.cache_data
def load_apparel_rim_dataset() -> pd.DataFrame:
    """Loads the 50-store, 24-month apparel dataset from CSV or Parquet."""
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "synthetic_apparel_rim_data.csv"
    parquet_path = base_dir / "synthetic_apparel_rim_data.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    from markdown_shock_simulator import generate_mock_apparel_data
    return generate_mock_apparel_data(n_stores=10, n_months=24)


@st.cache_resource
def load_or_train_clustering_pipeline() -> Tuple[pd.DataFrame, Any, Any, StoreBenchmarkKNN]:
    """Loads the serialized cluster & KNN pipeline from models/, or fits dynamically."""
    base_dir = Path(__file__).resolve().parent.parent
    model_path = base_dir / "models" / "store_cluster_pipeline.joblib"
    if model_path.exists():
        try:
            bundle = joblib.load(model_path)
            return bundle["store_summary_df"], bundle["scaler"], bundle["kmeans"], bundle["knn_engine"]
        except Exception:
            pass
    raw_df = load_apparel_rim_dataset()
    agg_df = load_and_aggregate_store_data(raw_df)
    return train_clustering_pipeline(agg_df)


def generate_ai_cfo_brief_text(
    store_id: str,
    archetype: str,
    tier: str,
    region: str,
    shock_pct: int,
    metrics: Dict[str, Any],
) -> str:
    """Backward-compatible shim delegating to services/llm_advisor.py."""
    return _services_generate_cfo_brief(
        store_id=store_id,
        archetype=archetype,
        tier=tier,
        region=region,
        shock_pct=shock_pct,
        metrics=metrics,
    )


def render_rim_store_selector(
    store_summary_df: pd.DataFrame,
    knn_engine: StoreBenchmarkKNN,
) -> Tuple[str, pd.Series, str, Dict[str, Any]]:
    """Renders the store dropdown and archetype header badge card."""
    all_stores = sorted(store_summary_df["store_id"].unique())
    default_idx = all_stores.index("STORE_104") if "STORE_104" in all_stores else 0

    top_col1, top_col2 = st.columns([2, 3])
    with top_col1:
        selected_store = st.selectbox("Select Target Store Location", options=all_stores, index=default_idx)

    peer_data = knn_engine.find_peer_group(selected_store)
    store_row = store_summary_df[store_summary_df["store_id"] == selected_store].iloc[0]
    archetype_label = store_row["cluster_label"]
    badge_color = ARCHETYPE_COLORS.get(archetype_label, "#38BDF8")

    with top_col2:
        st.markdown(
            f"""
            <div style="background: var(--secondary-background-color, #F8FAFC); border: 1px solid rgba(128, 128, 128, 0.25); border-radius: 8px; padding: 12px 16px; margin-top: 4px; display: flex; align-items: center; gap: 16px;">
                <div>
                    <span style="font-size: 0.75rem; color: var(--text-color, #475569); text-transform: uppercase; font-weight: 600;">Cluster Archetype</span>
                    <div style="font-size: 1.15rem; font-weight: 700; color: {badge_color};">{archetype_label}</div>
                </div>
                <div style="border-left: 1px solid rgba(128, 128, 128, 0.25); height: 36px;"></div>
                <div>
                    <span style="font-size: 0.75rem; color: var(--text-color, #475569); text-transform: uppercase; font-weight: 600;">Store Tier</span>
                    <div style="font-size: 1rem; font-weight: 600;">{store_row['store_tier']}</div>
                </div>
                <div style="border-left: 1px solid rgba(128, 128, 128, 0.25); height: 36px;"></div>
                <div>
                    <span style="font-size: 0.75rem; color: var(--text-color, #475569); text-transform: uppercase; font-weight: 600;">Region</span>
                    <div style="font-size: 1rem; font-weight: 600;">{store_row['region']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return selected_store, store_row, archetype_label, peer_data


def render_rim_subview_clustering(store_row: pd.Series, peer_data: Dict[str, Any]) -> None:
    """Renders Sub-View A: Predictive Store Clustering & KNN Peer Benchmarking."""
    st.markdown("---")
    st.markdown("#### 🎯 Sub-View A: Predictive Store Clustering & Operational Outlier Discovery")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Markdown Dependency %", f"{store_row['markdown_dependency'] * 100:.1f}%", delta="Target: < 15.0%", delta_color="inverse" if store_row["markdown_dependency"] > 0.18 else "normal")
    with k2:
        st.metric("Sell-Through Velocity", f"{store_row['sell_through_velocity'] * 100:.1f}%", delta="Target: > 28.0%", delta_color="normal" if store_row["sell_through_velocity"] >= 0.28 else "inverse")
    with k3:
        st.metric("Historical Shrink Rate", f"{store_row['shrinkage_rate'] * 100:.2f}%", delta="Target: < 2.0%", delta_color="inverse" if store_row["shrinkage_rate"] > 0.025 else "normal")
    with k4:
        st.metric("Normalized GMROI", f"{store_row['gmroi']:.2f}x", delta="Target: > 12.0x", delta_color="normal" if store_row["gmroi"] >= 12.0 else "inverse")

    st.markdown("##### 👥 Top 3 Nearest Sibling Peer Locations (Euclidean Similarity)")
    peers = peer_data["nearest_peers"]
    peer_table_rows = [
        {
            "Rank": f"#{idx}",
            "Store ID": p["peer_store_id"],
            "Region": p["peer_region"],
            "Store Tier": p["peer_tier"],
            "Archetype": p["peer_cluster_label"],
            "Euclidean Distance": f"{p['euclidean_distance']:.4f}",
            "Markdown %": f"{p['markdown_dependency'] * 100:.1f}%",
            "Sell-Through %": f"{p['sell_through_velocity'] * 100:.1f}%",
            "Shrink %": f"{p['shrinkage_rate'] * 100:.2f}%",
            "GMROI": f"{p['gmroi']:.2f}x",
        }
        for idx, p in enumerate(peers, 1)
    ]
    st.dataframe(pd.DataFrame(peer_table_rows), width="stretch", hide_index=True)


def render_rim_subview_shock_simulation(
    df_raw: pd.DataFrame,
    selected_store: str,
    store_row: pd.Series,
    kpi_renderer: Any = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Renders Sub-View B: Interactive Markdown Shock Simulator & Plotly valuation chart."""
    st.markdown("---")
    st.markdown("#### 📉 Sub-View B: Interactive Markdown Shock Simulator")
    st.markdown("##### 📌 Operational Inventory Productivity & RIM Accounting Baselines")
    if kpi_renderer:
        kpi_renderer(store_row)

    s_col1, s_col2 = st.columns([3, 1])
    with s_col1:
        markdown_shock = st.slider("Clearance Markdown Shock Depth (% discount uplift over plan)", 10, 50, 25, 5)
    with s_col2:
        elasticity_val = st.number_input("Volume Elasticity", 0.10, 0.80, 0.35, 0.05)

    store_records = df_raw[df_raw["store_id"] == selected_store].copy()
    df_projected = project_forward_12_periods(
        baseline_store_df=store_records,
        markdown_uplift_pct=markdown_shock / 100.0,
        sales_lift_elasticity=elasticity_val,
    )

    total_plan_sales = df_projected["planned_sales"].sum()
    total_shock_sales = df_projected["shocked_sales"].sum()
    sales_lift_pct = ((total_shock_sales / total_plan_sales) - 1.0) * 100.0 if total_plan_sales > 0 else 0.0
    final_plan_cost = df_projected["planned_ending_cost"].iloc[-1]
    final_shock_cost = df_projected["shocked_ending_cost"].iloc[-1]
    total_asset_deflation = final_plan_cost - final_shock_cost
    final_shock_retail = df_projected["shocked_ending_retail"].iloc[-1]
    avg_cr_ratio = df_projected["shocked_cr_ratio"].mean()

    base_gm_pct = 58.4
    shock_gm_pct = base_gm_pct - (markdown_shock * 0.28)
    margin_decay = shock_gm_pct - base_gm_pct

    d1, d2, d3, d4 = st.columns(4)
    with d1: st.metric("Gross Margin % Decay", f"{shock_gm_pct:.1f}%", delta=f"{margin_decay:.2f}% Margin Decay", delta_color="inverse")
    with d2: st.metric("Ending Inventory at Retail", f"${final_shock_retail:,.0f}", delta=f"-${df_projected['planned_ending_retail'].iloc[-1] - final_shock_retail:,.0f} Drop", delta_color="inverse")
    with d3: st.metric("Ending Inventory at Cost", f"${final_shock_cost:,.0f}", delta=f"-${total_asset_deflation:,.0f} Asset Deflation", delta_color="inverse")
    with d4: st.metric("Effective Cost-to-Retail Ratio", f"{avg_cr_ratio:.4f}", delta=f"{avg_cr_ratio * 100:.1f}% Cost Complement", delta_color="off")

    st.markdown("##### 📈 12-Forward-Period RIM Valuation & Asset Divergence Chart")
    fig = create_synchronized_valuation_figure(df_projected)
    st.plotly_chart(fig, width="stretch")

    shock_metrics = {
        "markdown_shock": markdown_shock,
        "sales_lift_pct": sales_lift_pct,
        "final_plan_cost": final_plan_cost,
        "final_shock_cost": final_shock_cost,
        "total_asset_deflation": total_asset_deflation,
        "final_shock_retail": final_shock_retail,
        "avg_cr_ratio": avg_cr_ratio,
        "base_gm_pct": base_gm_pct,
        "shock_gm_pct": shock_gm_pct,
        "margin_decay": margin_decay,
    }
    return df_projected, shock_metrics


def render_rim_subview_cfo_brief(
    selected_store: str,
    store_row: pd.Series,
    df_projected: pd.DataFrame,
    shock_metrics: Dict[str, Any],
    peer_data: Dict[str, Any],
    archetype_label: str,
) -> None:
    """Renders Sub-View C: AI CFO Executive Working Capital Risk Brief."""
    st.markdown("---")
    st.markdown("#### 💼 Sub-View C: AI CFO Executive Working Capital Risk Brief (Google Gemini 2.5 Pro)")

    cfo_state = RIMDashboardState(
        store_id=selected_store,
        store_tier=store_row["store_tier"],
        region=store_row["region"],
        markdown_shock_pct=float(shock_metrics["markdown_shock"]),
        cost_to_retail_ratio=float(shock_metrics["avg_cr_ratio"]),
        baseline_ending_retail=float(df_projected["planned_ending_retail"].iloc[-1]),
        shocked_ending_retail=float(shock_metrics["final_shock_retail"]),
        baseline_ending_cost=float(shock_metrics["final_plan_cost"]),
        shocked_ending_cost=float(shock_metrics["final_shock_cost"]),
        balance_sheet_asset_deflation=float(shock_metrics["total_asset_deflation"]),
        gross_margin_decay_pct=float(shock_metrics["margin_decay"]),
        projected_gmroi=float(store_row.get("gmroi", 10.0)),
        nearest_peers=[p["peer_store_id"] for p in peer_data["nearest_peers"]],
        archetype_label=archetype_label,
        sales_lift_pct=float(shock_metrics["sales_lift_pct"]),
        baseline_gm_pct=float(shock_metrics["base_gm_pct"]),
        shocked_gm_pct=float(shock_metrics["shock_gm_pct"]),
    )

    cfo_brief_key = f"cfo_brief_v2_{selected_store}_{shock_metrics['markdown_shock']}"
    cfo_model_key = f"cfo_model_v2_{selected_store}_{shock_metrics['markdown_shock']}"

    if cfo_brief_key not in st.session_state:
        res = generate_cfo_risk_brief(cfo_state)
        st.session_state[cfo_brief_key] = res["brief_text"]
        st.session_state[cfo_model_key] = res["model_used"]

    if st.button("🔍 Generate AI Executive Risk Brief", type="primary", width="stretch"):
        with st.spinner("🤖 Consulting Google Gemini 2.5 Pro Executive Risk Advisor..."):
            fresh = generate_cfo_risk_brief(cfo_state)
            st.session_state[cfo_brief_key] = fresh["brief_text"]
            st.session_state[cfo_model_key] = fresh["model_used"]

    brief_content = normalize_cfo_markdown(st.session_state.get(cfo_brief_key, ""))
    model_name = st.session_state.get(cfo_model_key, "Google Gemini 2.5 Pro")
    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.35); border-left: 4px solid #38BDF8; border-radius: 8px; padding: 12px 18px; margin: 14px 0; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.88rem; font-weight: 700; color: #38BDF8; text-transform: uppercase;">🛡️ AI CFO Executive Risk Brief ({selected_store} | +{shock_metrics['markdown_shock']}% Shock)</span>
            <span style="background: rgba(56, 189, 248, 0.15); color: #BAE6FD; font-size: 0.75rem; padding: 4px 12px; border-radius: 12px; font-weight: 600;">{model_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(brief_content)


def render_tab_rim_knn(show_header: bool = True, kpi_renderer: Any = None) -> None:
    """Renders the complete RIM Valuation Forensics & Predictive Store Clustering Cockpit."""
    if show_header:
        st.markdown("### 📊 RIM Valuation Forensics & Predictive Store Clustering")
        st.caption(
            "Combines machine learning cluster forensics with deterministic Retail Inventory Method (RIM) balance sheet accounting. "
            "Uncovers operational store archetypes, simulates multi-period clearance markdown shocks, and generates CFO risk briefs."
        )

    df_raw = load_apparel_rim_dataset()
    store_summary_df, scaler, kmeans, knn_engine = load_or_train_clustering_pipeline()

    selected_store, store_row, archetype_label, peer_data = render_rim_store_selector(
        store_summary_df, knn_engine
    )

    render_rim_subview_clustering(store_row, peer_data)

    df_projected, shock_metrics = render_rim_subview_shock_simulation(
        df_raw, selected_store, store_row, kpi_renderer=kpi_renderer
    )

    render_rim_subview_cfo_brief(
        selected_store, store_row, df_projected, shock_metrics, peer_data, archetype_label
    )

    active_rim_dict = {
        "cost_to_retail_ratio": float(shock_metrics["avg_cr_ratio"]),
        "store_sales": float(df_projected["planned_sales"].sum()) if "planned_sales" in df_projected.columns else 3_200_000.0,
        "markdown_rate_pct": float(store_row["markdown_dependency"] * 100.0),
    }
    render_value_calculator_tab(active_rim_metrics=active_rim_dict)
