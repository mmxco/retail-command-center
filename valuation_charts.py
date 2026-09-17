"""
Forward 12-Period Valuation & Asset Divergence Visualization Module.

Demonstrates the Retail Inventory Method (RIM) structural divergence between
Retail Book Value (Ending Inventory at Retail) and Balance Sheet Asset Valuation
(Ending Inventory at Cost) across a 12-month forward simulation horizon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ==============================================================================
# DATA LOADER & ROBUST FALLBACK GENERATOR
# ==============================================================================

def generate_mock_store_history(store_id: str = "STORE_104", n_months: int = 12) -> pd.DataFrame:
    """Generates a realistic 12-month baseline history for a store if data files are missing."""
    dates = pd.date_range("2024-11-01", periods=n_months, freq="MS")
    rng = np.random.default_rng(42)

    base_sales = 450_000.0
    cur_cost = base_sales * 1.2
    cur_retail = cur_cost / 0.42

    records = []
    for p_date in dates:
        sales = float(base_sales * rng.uniform(0.85, 1.30))
        mds = float(sales * rng.uniform(0.12, 0.18))
        shrink = float(sales * 0.02)
        p_retail = float(sales * 1.12)
        p_cost = float(p_retail * 0.41)

        tgas_retail = cur_retail + p_retail
        tgas_cost = cur_cost + p_cost
        cr_ratio = tgas_cost / tgas_retail if tgas_retail > 0 else 0.41
        end_retail = tgas_retail - (sales + mds + shrink)
        end_cost = end_retail * cr_ratio

        records.append({
            "period_date": p_date.strftime("%Y-%m-%d"),
            "store_id": store_id,
            "region": "West",
            "store_tier": "Flagship Tier 1",
            "beginning_inv_cost": cur_cost,
            "beginning_inv_retail": cur_retail,
            "purchases_cost": p_cost,
            "purchases_retail": p_retail,
            "net_sales": sales,
            "promo_markdowns": mds,
            "shrinkage": shrink,
            "ending_inv_retail": end_retail,
            "cost_to_retail_ratio": cr_ratio,
            "ending_inv_cost": end_cost,
            "gross_margin": sales - (tgas_cost - end_cost),
        })
        cur_cost = end_cost
        cur_retail = end_retail

    return pd.DataFrame(records)


@st.cache_data
def load_store_dataset() -> pd.DataFrame:
    """Loads dataset from parquet or csv, with mock fallback if absent."""
    root_dir = Path(__file__).resolve().parent
    parquet_path = root_dir / "synthetic_apparel_rim_data.parquet"
    csv_path = root_dir / "synthetic_apparel_rim_data.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        return generate_mock_store_history(store_id="STORE_104", n_months=24)


@st.cache_data
def load_store_clusters() -> Dict[str, str]:
    """Loads store_id -> cluster_label mapping as assessed by the KNN/KMeans clustering model."""
    root_dir = Path(__file__).resolve().parent
    summary_path = root_dir / "store_clusters_summary.csv"
    if summary_path.exists():
        try:
            cdf = pd.read_csv(summary_path)
            if "store_id" in cdf.columns and "cluster_label" in cdf.columns:
                return dict(zip(cdf["store_id"], cdf["cluster_label"]))
        except Exception:
            pass

    pipeline_path = root_dir / "models" / "store_cluster_pipeline.joblib"
    if pipeline_path.exists():
        try:
            import joblib
            bundle = joblib.load(pipeline_path)
            sdf = bundle.get("store_summary_df")
            if sdf is not None and "cluster_label" in sdf.columns:
                return dict(zip(sdf["store_id"], sdf["cluster_label"]))
        except Exception:
            pass

    return {}


# ==============================================================================
# FORWARD 12-PERIOD SIMULATION ENGINE
# ==============================================================================

def project_forward_12_periods(
    baseline_store_df: pd.DataFrame,
    markdown_uplift_pct: float = 0.25,
    sales_lift_elasticity: float = 0.35,
    shrink_rate: float = 0.02,
) -> pd.DataFrame:
    """
    Simulates 12 forward monthly periods comparing a planned baseline against
    a discount-shocked scenario under Retail Inventory Method (RIM) roll-forward rules.

    Args:
        baseline_store_df: Historical records for the store (trailing 12 months used as run-rate).
        markdown_uplift_pct: Clearance markdown depth shock (0.10 to 0.50).
        sales_lift_elasticity: Volume elasticity coefficient on price markdowns (default 0.35).
        shrink_rate: Retail shrinkage rate as percentage of net sales (default 0.02).

    Returns:
        DataFrame containing 12 forward monthly periods with planned and shocked metrics.
    """
    if baseline_store_df.empty:
        baseline_store_df = generate_mock_store_history()

    sorted_base = baseline_store_df.sort_values("period_date").reset_index(drop=True)
    # Use the last 12 historical periods as seasonal pattern template
    ref_window = sorted_base.tail(12).reset_index(drop=True)
    if len(ref_window) < 12:
        # Replicate to ensure 12 periods
        ref_window = pd.concat([ref_window] * (12 // len(ref_window) + 1)).iloc[:12].reset_index(drop=True)

    # Starting balance for Month +1 is the final ending balance of historical window
    last_row = sorted_base.iloc[-1]
    last_date_str = str(last_row.get("period_date", "2026-10-01"))
    try:
        start_date = pd.to_datetime(last_date_str) + pd.DateOffset(months=1)
    except Exception:
        start_date = pd.to_datetime("2026-11-01")

    forward_dates = pd.date_range(start_date, periods=12, freq="MS")

    # Planned roll-forward initial balances
    plan_beg_cost = float(last_row["ending_inv_cost"])
    plan_beg_retail = float(last_row["ending_inv_retail"])

    # Shocked roll-forward initial balances (starts equal, then diverges)
    shock_beg_cost = plan_beg_cost
    shock_beg_retail = plan_beg_retail

    projected_rows: List[Dict[str, Any]] = []

    for idx, f_date in enumerate(forward_dates):
        ref_period = ref_window.iloc[idx]
        base_sales = float(ref_period["net_sales"])
        base_mds = float(ref_period["promo_markdowns"])
        purch_cost = float(ref_period["purchases_cost"])
        purch_retail = float(ref_period["purchases_retail"])

        # ----------------------------------------------------------------------
        # 1. Planned Scenario
        # ----------------------------------------------------------------------
        plan_tgas_ret = plan_beg_retail + purch_retail
        plan_tgas_cost = plan_beg_cost + purch_cost
        plan_cr_ratio = plan_tgas_cost / plan_tgas_ret if plan_tgas_ret > 0 else 0.40

        plan_shrink = base_sales * shrink_rate
        plan_reductions = base_sales + base_mds + plan_shrink
        plan_end_retail = max(0.0, plan_tgas_ret - plan_reductions)
        plan_end_cost = max(0.0, plan_end_retail * plan_cr_ratio)

        # ----------------------------------------------------------------------
        # 2. Shocked Scenario
        # ----------------------------------------------------------------------
        shock_tgas_ret = shock_beg_retail + purch_retail
        shock_tgas_cost = shock_beg_cost + purch_cost
        shock_cr_ratio = shock_tgas_cost / shock_tgas_ret if shock_tgas_ret > 0 else 0.40

        # Promotional shock applied
        shock_mds = base_mds * (1.0 + markdown_uplift_pct)
        shock_sales = base_sales * (1.0 + (markdown_uplift_pct * sales_lift_elasticity))
        shock_shrink = shock_sales * shrink_rate
        shock_reductions = shock_sales + shock_mds + shock_shrink

        shock_end_retail = max(0.0, shock_tgas_ret - shock_reductions)
        shock_end_cost = max(0.0, shock_end_retail * shock_cr_ratio)

        # Divergence metrics
        asset_deflation = plan_end_cost - shock_end_cost
        retail_divergence_gap = plan_end_retail - plan_end_cost
        shock_divergence_gap = shock_end_retail - shock_end_cost

        projected_rows.append({
            "period_index": idx + 1,
            "period_label": f_date.strftime("%b %Y"),
            "period_date": f_date.strftime("%Y-%m-%d"),
            # Planned figures
            "planned_ending_retail": round(plan_end_retail, 2),
            "planned_ending_cost": round(plan_end_cost, 2),
            "planned_sales": round(base_sales, 2),
            "planned_markdowns": round(base_mds, 2),
            "planned_cr_ratio": round(plan_cr_ratio, 4),
            # Shocked figures
            "shocked_ending_retail": round(shock_end_retail, 2),
            "shocked_ending_cost": round(shock_end_cost, 2),
            "shocked_sales": round(shock_sales, 2),
            "shocked_markdowns": round(shock_mds, 2),
            "shocked_cr_ratio": round(shock_cr_ratio, 4),
            # Financial Forensics Deltas
            "asset_deflation": round(asset_deflation, 2),
            "cumulative_asset_deflation": round(asset_deflation, 2),
            "planned_divergence_gap": round(retail_divergence_gap, 2),
            "shocked_divergence_gap": round(shock_divergence_gap, 2),
        })

        # Roll forward balances: Month T ending becomes Month T+1 beginning
        plan_beg_cost = plan_end_cost
        plan_beg_retail = plan_end_retail
        shock_beg_cost = shock_end_cost
        shock_beg_retail = shock_end_retail

    df_proj = pd.DataFrame(projected_rows)
    return df_proj


# ==============================================================================
# SYNCHRONIZED PLOTLY VISUALIZATION
# ==============================================================================

def create_synchronized_valuation_figure(df_projected: pd.DataFrame) -> go.Figure:
    """
    Constructs a 2-row synchronized Plotly subplots figure:
    Top Panel: Asset Valuation Divergence (Retail Book Value vs. Planned Cost vs. Shocked Cost).
    Bottom Panel: Monthly Promotional Markdowns ($) vs. Cost-to-Retail Complement Ratio (%).
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    x_labels = df_projected["period_label"]

    # --------------------------------------------------------------------------
    # SUBPLOT 1: ASSET VALUATION & BOOK VALUE DIVERGENCE (TOP PANEL)
    # --------------------------------------------------------------------------
    # Trace 1: Retail Book Value (Planned Ending Retail)
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=df_projected["planned_ending_retail"],
            mode="lines+markers",
            name="Retail Book Value (Planned Retail)",
            line=dict(color="#38BDF8", width=2.5, dash="dash"),
            marker=dict(size=6),
            hovertemplate="Retail Book Value: %{y:$,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Trace 2: Planned Ending Inventory at Cost (Baseline Balance Sheet)
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=df_projected["planned_ending_cost"],
            mode="lines+markers",
            name="Planned Cost Asset (Baseline)",
            line=dict(color="#10B981", width=3),
            marker=dict(size=6),
            hovertemplate="Planned Cost Asset: %{y:$,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Trace 3: Shocked Ending Inventory at Cost (Impaired Balance Sheet)
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=df_projected["shocked_ending_cost"],
            mode="lines+markers",
            name="Shocked Cost Asset (Impaired)",
            line=dict(color="#EF4444", width=3),
            marker=dict(size=6),
            fill="tonexty",
            fillcolor="rgba(239, 68, 68, 0.18)",
            hovertemplate="Shocked Cost Asset: %{y:$,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # --------------------------------------------------------------------------
    # SUBPLOT 2: MARKDOWN VOLUME & MARKON COMPLEMENT (BOTTOM PANEL)
    # --------------------------------------------------------------------------
    # Bar 1: Planned Markdowns ($)
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=df_projected["planned_markdowns"],
            name="Planned Markdowns ($)",
            marker_color="#6366F1",
            opacity=0.6,
            hovertemplate="Planned Markdowns: %{y:$,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    # Bar 2: Shocked Markdowns ($)
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=df_projected["shocked_markdowns"],
            name="Shocked Markdowns ($)",
            marker_color="#F59E0B",
            opacity=0.85,
            hovertemplate="Shocked Markdowns: %{y:$,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
        secondary_y=False,
    )

    # Line on Secondary Y: Shocked Cost-to-Retail Ratio (%)
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=df_projected["shocked_cr_ratio"],
            mode="lines+markers",
            name="Cost-to-Retail Complement (%)",
            line=dict(color="#C084FC", width=2.5, dash="dot"),
            marker=dict(size=5),
            hovertemplate="Cost-to-Retail Ratio: %{y:.2%}<extra></extra>",
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # --------------------------------------------------------------------------
    # PEAK WRITE-DOWN INFLECTION ANNOTATION
    # --------------------------------------------------------------------------
    max_deflation_idx = int(df_projected["asset_deflation"].idxmax())
    max_deflation_val = df_projected.loc[max_deflation_idx, "asset_deflation"]
    max_deflation_period = df_projected.loc[max_deflation_idx, "period_label"]
    max_deflation_shock_cost = df_projected.loc[max_deflation_idx, "shocked_ending_cost"]

    fig.add_annotation(
        x=max_deflation_period,
        y=max_deflation_shock_cost,
        xref="x1",
        yref="y1",
        text=f"Max Impairment Gap<br>-${max_deflation_val:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=2,
        arrowcolor="#EF4444",
        ax=0,
        ay=45,
        bgcolor="rgba(15, 23, 42, 0.9)",
        bordercolor="#EF4444",
        borderwidth=1,
        borderpad=4,
        font=dict(size=11, color="#FFFFFF"),
        row=1,
        col=1,
    )

    # --------------------------------------------------------------------------
    # LAYOUT & INTERACTIVE CONTROLS
    # --------------------------------------------------------------------------
    fig.update_layout(
        template="plotly_dark",
        height=680,
        barmode="group",
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.0,
        ),
        hovermode="x unified",
    )

    # Top Panel Axes
    fig.update_yaxes(
        title_text="Valuation Balance ($)",
        tickformat="$,.0f",
        row=1,
        col=1,
    )

    # Bottom Panel Axes
    fig.update_yaxes(
        title_text="Markdowns ($)",
        tickformat="$,.0f",
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Cost Complement (%)",
        tickformat=".1%",
        showgrid=False,
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.update_xaxes(title_text="Forward Simulation Horizon", row=2, col=1)

    return fig


# ==============================================================================
# STREAMLIT COMPONENT PREVIEW & RUNNER
# ==============================================================================

def render_valuation_forensics_dashboard() -> None:
    """Renders the complete interactive Streamlit valuation divergence dashboard."""
    st.set_page_config(
        page_title="Apparel Command Center | 12-Period Valuation Divergence",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📈 Forward 12-Period RIM Valuation Divergence Engine")
    st.caption("Retail Book Value vs. Balance Sheet Asset Valuation (Cost) Under Progressive Markdown Shocks")

    df_full = load_store_dataset()
    cluster_map = load_store_clusters()
    store_list = sorted(df_full["store_id"].unique())
    default_idx = store_list.index("STORE_104") if "STORE_104" in store_list else 0

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Simulation Settings")
        selected_store = st.selectbox(
            "Select Store Location",
            options=store_list,
            index=default_idx,
            format_func=lambda s: f"{s} — [{cluster_map.get(s, 'Balanced Regional Performers')}]",
            help="Choose an apparel store to project its forward 12-month balance sheet.",
        )

        store_meta = df_full[df_full["store_id"] == selected_store].iloc[0]
        assigned_cluster = cluster_map.get(
            selected_store, store_meta.get("anomaly_profile", "Balanced Regional Performers")
        )
        st.info(
            f"**KNN Cluster:** {assigned_cluster}\n\n"
            f"**Tier:** {store_meta.get('store_tier', 'Standard')}\n\n"
            f"**Region:** {store_meta.get('region', 'National')}"
        )

        st.markdown("---")
        markdown_shock = st.slider(
            "⚠️ Promotional Markdown Uplift (% over plan)",
            min_value=10,
            max_value=50,
            value=25,
            step=5,
            help="Simulates heavy promotional clearance discounting.",
        )

        with st.expander("Advanced Mechanics"):
            elasticity = st.slider(
                "Promotional Elasticity Factor",
                min_value=0.10,
                max_value=0.80,
                value=0.35,
                step=0.05,
                help="Proportion of discount converted into volume revenue lift.",
            )
            shrinkage = st.slider(
                "Historical Shrink Rate",
                min_value=0.01,
                max_value=0.06,
                value=0.02,
                step=0.005,
                format="%.3f",
            )

    # Filter store history and run 12-period projection
    store_history = df_full[df_full["store_id"] == selected_store].copy()
    df_proj = project_forward_12_periods(
        baseline_store_df=store_history,
        markdown_uplift_pct=markdown_shock / 100.0,
        sales_lift_elasticity=elasticity,
        shrink_rate=shrinkage,
    )

    # Key Aggregates
    final_planned_retail = df_proj["planned_ending_retail"].iloc[-1]
    final_planned_cost = df_proj["planned_ending_cost"].iloc[-1]
    final_shocked_cost = df_proj["shocked_ending_cost"].iloc[-1]
    total_asset_deflation = final_planned_cost - final_shocked_cost
    deflation_pct = (total_asset_deflation / final_planned_cost * 100.0) if final_planned_cost > 0 else 0.0

    # Executive Metric Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="12-Month Ending Retail Book Value",
            value=f"${final_planned_retail:,.0f}",
            delta=f"${df_proj['shocked_ending_retail'].iloc[-1] - final_planned_retail:,.0f} Under Shock",
            delta_color="off",
            help="Terminal floor inventory at original ticket retail.",
        )

    with col2:
        st.metric(
            label="Balance Sheet Ending Cost Asset",
            value=f"${final_shocked_cost:,.0f}",
            delta=f"-${total_asset_deflation:,.0f} (-{deflation_pct:.1f}%)",
            delta_color="inverse",
            help="Realized balance sheet inventory valuation at cost under RIM accounting.",
        )

    with col3:
        st.metric(
            label="Total Net Asset Impairment",
            value=f"${total_asset_deflation:,.0f}",
            delta="Forced Balance Sheet Write-Down",
            delta_color="inverse",
            help="Cumulative inventory asset valuation write-off driven by markdown dilution.",
        )

    st.markdown("---")

    # Plotly Synchronized Visualization
    fig = create_synchronized_valuation_figure(df_proj)
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )

    # Domain Brief
    with st.expander("📘 Executive RIM Accounting Takeaway"):
        st.markdown(f"""
        **The Retail Inventory Method Structural Trap**:
        * In Month 1, clearance discounts clear physical units off the floor, reducing Ending Inventory at Retail.
        * However, because purchase margins are fixed in previous cycles, the **Cost-to-Retail Ratio** ({df_proj['planned_cr_ratio'].mean():.2%}) forces an immediate downward adjustment to ending inventory at cost.
        * Across the 12 forward periods, **${total_asset_deflation:,.2f}** in working capital is eliminated from the balance sheet, illustrating why high clearance velocity without margin protection leads to corporate asset impairment.
        """)


if __name__ == "__main__":
    render_valuation_forensics_dashboard()
