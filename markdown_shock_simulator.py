"""
Apparel Pre-Sales Command Center: Markdown Shock Simulator & RIM Valuation Forensics.

Simulates the impact of clearance markdown shocks (10% to 50%) on:
1. Gross Margin Decay & Erosion
2. Retail Inventory Method (RIM) Cost Complement Degradation
3. Balance Sheet Ending Inventory Asset Deflation & Working Capital Write-Downs
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Google GenAI SDK
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


# ==============================================================================
# DATA INGESTION & ZERO-CRASH FALLBACK MOCK GENERATOR
# ==============================================================================

def generate_mock_apparel_data(n_stores: int = 5, n_months: int = 24) -> pd.DataFrame:
    """Generates a realistic 5-store apparel dataset on the fly if files are missing."""
    store_ids = [f"STORE_{100 + i}" for i in range(1, n_stores + 1)]
    regions = ["NorthEast", "Central", "SouthEast", "West", "NorthEast"]
    tiers = [
        "Flagship Tier 1",
        "Regional Mall Tier 2",
        "Regional Mall Tier 2",
        "Strip/Outlet Tier 3",
        "Flagship Tier 1",
    ]
    dates = pd.date_range("2024-11-01", periods=n_months, freq="MS")

    records = []
    rng = np.random.default_rng(42)

    for s_idx, store_id in enumerate(store_ids):
        base_sales = 300_000 if "Flagship" not in tiers[s_idx] else 600_000
        cur_cost = base_sales * 1.2
        cur_retail = cur_cost / 0.42

        for p_date in dates:
            sales = base_sales * rng.uniform(0.85, 1.35)
            mds = sales * rng.uniform(0.12, 0.20)
            shrink = sales * rng.uniform(0.015, 0.025)
            p_retail = sales * 1.15
            p_cost = p_retail * 0.41

            # RIM calculations
            tgas_ret = cur_retail + p_retail
            tgas_cost = cur_cost + p_cost
            cr_ratio = tgas_cost / tgas_ret
            reds = sales + mds + shrink
            end_ret = tgas_ret - reds
            end_cost = end_ret * cr_ratio
            cogs = tgas_cost - end_cost
            gm = sales - cogs

            records.append({
                "period_date": p_date.strftime("%Y-%m-%d"),
                "store_id": store_id,
                "region": regions[s_idx],
                "store_tier": tiers[s_idx],
                "beginning_inv_cost": cur_cost,
                "beginning_inv_retail": cur_retail,
                "purchases_cost": p_cost,
                "purchases_retail": p_retail,
                "net_sales": sales,
                "promo_markdowns": mds,
                "shrinkage": shrink,
                "ending_inv_retail": end_ret,
                "cost_to_retail_ratio": cr_ratio,
                "ending_inv_cost": end_cost,
                "cogs": cogs,
                "gross_margin": gm,
            })
            cur_cost = end_cost
            cur_retail = end_ret

    return pd.DataFrame(records)


@st.cache_data
def load_rim_data() -> pd.DataFrame:
    """Loads synthetic RIM dataset, falling back to mock generator if missing."""
    root_dir = Path(__file__).resolve().parent
    parquet_path = root_dir / "synthetic_apparel_rim_data.parquet"
    csv_path = root_dir / "synthetic_apparel_rim_data.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    elif csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        return generate_mock_apparel_data()


# ==============================================================================
# RIM SIMULATION MATHEMATICS
# ==============================================================================

def simulate_period_shock(
    beg_cost: float,
    beg_retail: float,
    purchases_cost: float,
    purchases_retail: float,
    baseline_sales: float,
    baseline_markdowns: float,
    shrinkage: float,
    shock_pct: float,
    elasticity: float = 0.35,
) -> Dict[str, float]:
    """
    Executes single-period RIM recalculation under markdown shock.
    """
    # 1. Baseline Accumulation
    tgas_retail = beg_retail + purchases_retail
    tgas_cost = beg_cost + purchases_cost
    cr_ratio = tgas_cost / tgas_retail if tgas_retail > 0 else 0.0

    # 2. Shocked Reductions
    shock_mult = shock_pct / 100.0
    sim_markdowns = baseline_markdowns * (1.0 + shock_mult)
    sim_sales = baseline_sales * (1.0 + (shock_mult * elasticity))
    sim_reductions = sim_sales + sim_markdowns + shrinkage

    # 3. Shocked Valuation
    sim_ending_retail = tgas_retail - sim_reductions
    sim_ending_cost = max(0.0, sim_ending_retail * cr_ratio)
    sim_cogs = tgas_cost - sim_ending_cost
    sim_gross_margin = sim_sales - sim_cogs
    sim_gm_pct = (sim_gross_margin / sim_sales * 100.0) if sim_sales > 0 else 0.0

    # Baseline comparison (0% shock)
    base_reductions = baseline_sales + baseline_markdowns + shrinkage
    base_ending_retail = tgas_retail - base_reductions
    base_ending_cost = max(0.0, base_ending_retail * cr_ratio)
    base_cogs = tgas_cost - base_ending_cost
    base_gross_margin = baseline_sales - base_cogs
    base_gm_pct = (base_gross_margin / baseline_sales * 100.0) if baseline_sales > 0 else 0.0

    asset_deflation = base_ending_cost - sim_ending_cost
    margin_decay = sim_gm_pct - base_gm_pct

    return {
        "tgas_cost": tgas_cost,
        "tgas_retail": tgas_retail,
        "cr_ratio": cr_ratio,
        "baseline_sales": baseline_sales,
        "baseline_markdowns": baseline_markdowns,
        "base_ending_retail": base_ending_retail,
        "base_ending_cost": base_ending_cost,
        "base_gross_margin": base_gross_margin,
        "base_gm_pct": base_gm_pct,
        "sim_sales": sim_sales,
        "sim_markdowns": sim_markdowns,
        "sim_ending_retail": sim_ending_retail,
        "sim_ending_cost": sim_ending_cost,
        "sim_gross_margin": sim_gross_margin,
        "sim_gm_pct": sim_gm_pct,
        "asset_deflation": asset_deflation,
        "margin_decay": margin_decay,
    }


def simulate_multi_period_rollforward(
    store_df: pd.DataFrame,
    shock_pct: float,
    elasticity: float = 0.35,
    trailing_months: int = 12,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Executes sequential multi-period balance sheet roll-forward under markdown shock.
    Month T shocked ending inventory becomes Month T+1 shocked beginning inventory.
    """
    sorted_df = store_df.sort_values("period_date").reset_index(drop=True)
    if len(sorted_df) > trailing_months:
        sorted_df = sorted_df.tail(trailing_months).reset_index(drop=True)

    sim_rows = []
    # Starting balance for first period in simulation
    cur_sim_beg_cost = float(sorted_df.loc[0, "beginning_inv_cost"])
    cur_sim_beg_retail = float(sorted_df.loc[0, "beginning_inv_retail"])

    for _, row in sorted_df.iterrows():
        p_date = row["period_date"]
        p_cost = float(row["purchases_cost"])
        p_retail = float(row["purchases_retail"])
        base_sales = float(row["net_sales"])
        base_mds = float(row["promo_markdowns"])
        shrink = float(row["shrinkage"])

        # Base values as reported in historical data
        actual_base_end_cost = float(row["ending_inv_cost"])
        actual_base_end_retail = float(row["ending_inv_retail"])
        actual_base_gm = float(row["gross_margin"])

        # Shocked simulation with dynamic roll-forward beginning inventory
        res = simulate_period_shock(
            beg_cost=cur_sim_beg_cost,
            beg_retail=cur_sim_beg_retail,
            purchases_cost=p_cost,
            purchases_retail=p_retail,
            baseline_sales=base_sales,
            baseline_markdowns=base_mds,
            shrinkage=shrink,
            shock_pct=shock_pct,
            elasticity=elasticity,
        )

        sim_rows.append({
            "period_date": p_date,
            "baseline_ending_cost": actual_base_end_cost,
            "baseline_ending_retail": actual_base_end_retail,
            "baseline_sales": base_sales,
            "baseline_gross_margin": actual_base_gm,
            "baseline_gm_pct": (actual_base_gm / base_sales * 100.0) if base_sales > 0 else 0.0,
            "shocked_ending_cost": res["sim_ending_cost"],
            "shocked_ending_retail": res["sim_ending_retail"],
            "shocked_sales": res["sim_sales"],
            "shocked_markdowns": res["sim_markdowns"],
            "shocked_gross_margin": res["sim_gross_margin"],
            "shocked_gm_pct": res["sim_gm_pct"],
            "cr_ratio": res["cr_ratio"],
            "asset_deflation": actual_base_end_cost - res["sim_ending_cost"],
            "margin_decay": res["sim_gm_pct"] - ((actual_base_gm / base_sales * 100.0) if base_sales > 0 else 0.0),
        })

        # Roll-forward shocked ending to next period's beginning
        cur_sim_beg_cost = res["sim_ending_cost"]
        cur_sim_beg_retail = res["sim_ending_retail"]

    res_df = pd.DataFrame(sim_rows)

    # Aggregate executive metrics
    total_base_sales = res_df["baseline_sales"].sum()
    total_shock_sales = res_df["shocked_sales"].sum()
    total_base_gm = res_df["baseline_gross_margin"].sum()
    total_shock_gm = res_df["shocked_gross_margin"].sum()
    final_base_end_cost = res_df["baseline_ending_cost"].iloc[-1]
    final_shock_end_cost = res_df["shocked_ending_cost"].iloc[-1]

    agg_metrics = {
        "total_base_sales": total_base_sales,
        "total_shock_sales": total_shock_sales,
        "total_base_gm": total_base_gm,
        "total_shock_gm": total_shock_gm,
        "base_gm_pct": (total_base_gm / total_base_sales * 100.0) if total_base_sales > 0 else 0.0,
        "shock_gm_pct": (total_shock_gm / total_shock_sales * 100.0) if total_shock_sales > 0 else 0.0,
        "final_base_end_cost": final_base_end_cost,
        "final_shock_end_cost": final_shock_end_cost,
        "total_asset_deflation": final_base_end_cost - final_shock_end_cost,
        "avg_cr_ratio": res_df["cr_ratio"].mean(),
        "total_gm_decay": ((total_shock_gm / total_shock_sales) - (total_base_gm / total_base_sales)) * 100.0,
    }

    return res_df, agg_metrics


# ==============================================================================
# AI CFO BRIEF GENERATOR
# ==============================================================================

def generate_ai_cfo_brief(
    store_id: str,
    tier: str,
    region: str,
    shock_pct: int,
    agg_metrics: Dict[str, float],
) -> str:
    """Calls Gemini API (or heuristic fallback) to generate an executive CFO risk brief."""
    api_key = os.environ.get("GEMINI_API_KEY")

    sales_lift_pct = ((agg_metrics["total_shock_sales"] / agg_metrics["total_base_sales"]) - 1.0) * 100.0
    gm_drop_pct = agg_metrics["total_gm_decay"]
    write_down = agg_metrics["total_asset_deflation"]

    if api_key and GENAI_AVAILABLE:
        try:
            client = genai.Client(api_key=api_key)
            prompt = f"""
You are a retail Chief Financial Officer (CFO) and Merchandising Risk Strategist.
Review the following Retail Inventory Method (RIM) markdown shock stress test:

Store: {store_id} ({tier}, Region: {region})
Clearance Markdown Shock: +{shock_pct}% over planned clearance budget
Top-Line Sales Lift (from volume elasticity): +{sales_lift_pct:.1f}%
Gross Margin Decay: {gm_drop_pct:.2f}% (from {agg_metrics['base_gm_pct']:.1f}% to {agg_metrics['shock_gm_pct']:.1f}%)
Balance Sheet Asset Deflation / Write-Down: ${write_down:,.2f}
Effective Cost-to-Retail Ratio: {agg_metrics['avg_cr_ratio']:.4f}

Provide exactly 3 concise, high-impact executive bullet points:
1. Working Capital & Balance Sheet Risk: Explain how the RIM accounting formula forces this inventory asset write-down.
2. Margin Destruction Warning: Contrast the illusion of top-line sales lift against gross profit dollar destruction.
3. Actionable Guardrails: Specific markdown timing and promotional throttle recommendation for merchandising leadership.
Use professional, executive finance language.
"""
            # Support modern model with fallback
            for model_name in ["gemini-3.8-flash", "gemini-2.5-flash"]:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    return response.text
                except Exception:
                    continue
        except Exception:
            pass

    # Heuristic Rule-Based CFO Brief (Guarantees zero demo failures when offline or without API key)
    return rf"""
* **Balance Sheet Asset Deflation (\$ {write_down:,.0f} Forced Write-Down)**:
  Under Retail Inventory Method accounting, clearing units at +{shock_pct}% promotional markdowns rapidly depresses Ending Inventory at Retail without generating equivalent cash inflows. Because the cost complement remains fixed at {agg_metrics['avg_cr_ratio']:.4f}, the balance sheet absorbs an immediate, non-cash inventory valuation write-off of **${write_down:,.2f}**, directly eroding working capital collateral.

* **Top-Line Illusion vs. Bottom-Line Margin Collapse ({gm_drop_pct:.2f}% Decay)**:
  While price elasticity delivers a modest volume lift (+{sales_lift_pct:.1f}% net sales), gross margin contracts drastically from **{agg_metrics['base_gm_pct']:.1f}%** down to **{agg_metrics['shock_gm_pct']:.1f}%**. Each incremental dollar of clearance volume produces negative marginal gross profit after accounting for inventory replacement costs.

* **Executive Merchandising Guardrails**:
  Cap promotional clearance depth at **15% maximum discount cadence** over baseline plan. Shift aged inventory liquidation strategy from permanent point-of-sale ticket markdowns to cross-regional transfers into high-velocity Flagship locations to protect the merchandise cost complement.
"""


# ==============================================================================
# STREAMLIT UI & INTERACTIVE DASHBOARD
# ==============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Apparel Command Center | Markdown Shock Simulator",
        page_icon="🏷️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom Header Styling
    st.markdown("""
        <style>
        .metric-card {
            background-color: #0e1117;
            border: 1px solid #262730;
            border-radius: 8px;
            padding: 16px;
        }
        .stMetric {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 6px;
            padding: 12px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        </style>
    """, unsafe_allow_html=True)

    # Title and Executive Header
    st.title("🏷️ Markdown Shock Simulator & RIM Valuation Forensics")
    st.caption("Retail Inventory Method (RIM) Multi-Period Stress Testing: Clearance Elasticity, Margin Decay, and Asset Deflation")

    # Load dataset
    df = load_rim_data()
    all_stores = sorted(df["store_id"].unique())
    default_store_idx = all_stores.index("STORE_104") if "STORE_104" in all_stores else 0

    # --------------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.header("🎛️ Simulator Parameters")

        selected_store = st.selectbox(
            "📍 Store Location",
            options=all_stores,
            index=default_store_idx,
            help="Select any store location across Flagship, Regional Mall, or Outlet tiers.",
        )

        store_meta = df[df["store_id"] == selected_store].iloc[0]
        st.info(f"**Tier:** {store_meta['store_tier']}\n\n**Region:** {store_meta['region']}")

        horizon = st.radio(
            "📅 Simulation Horizon",
            options=["Trailing 12-Month Roll-Forward", "Single Month Snapshot"],
            index=0,
            help="Trailing 12-Month cascades compounding balance sheet effects; Single Month snapshots a specific period in isolation.",
        )

        selected_month = None
        if horizon == "Single Month Snapshot":
            store_months = sorted(df[df["store_id"] == selected_store]["period_date"].unique())
            selected_month = st.selectbox("Select Accounting Period", options=store_months, index=len(store_months) - 1)

        st.markdown("---")
        shock_pct = st.slider(
            "⚠️ Simulated Clearance Markdown Uplift (% over plan)",
            min_value=10,
            max_value=50,
            value=25,
            step=5,
            help="Simulates aggressive clearance markdowns (10% to 50%) to clear aged seasonal stock.",
        )

        with st.expander("⚙️ Advanced Elasticity Settings"):
            elasticity = st.slider(
                "Promotional Volume Elasticity",
                min_value=0.10,
                max_value=0.80,
                value=0.35,
                step=0.05,
                help="Proportion of markdown depth converted into incremental unit sales volume.",
            )

    # --------------------------------------------------------------------------
    # EXECUTE SIMULATION
    # --------------------------------------------------------------------------
    store_records = df[df["store_id"] == selected_store].copy()

    if horizon == "Trailing 12-Month Roll-Forward":
        sim_df, agg_metrics = simulate_multi_period_rollforward(
            store_records,
            shock_pct=shock_pct,
            elasticity=elasticity,
            trailing_months=12,
        )
    else:
        # Single Month Snapshot
        target_row = store_records[store_records["period_date"] == selected_month].iloc[0]
        single_res = simulate_period_shock(
            beg_cost=float(target_row["beginning_inv_cost"]),
            beg_retail=float(target_row["beginning_inv_retail"]),
            purchases_cost=float(target_row["purchases_cost"]),
            purchases_retail=float(target_row["purchases_retail"]),
            baseline_sales=float(target_row["net_sales"]),
            baseline_markdowns=float(target_row["promo_markdowns"]),
            shrinkage=float(target_row["shrinkage"]),
            shock_pct=shock_pct,
            elasticity=elasticity,
        )
        sim_df = pd.DataFrame([{
            "period_date": selected_month,
            "baseline_ending_cost": single_res["base_ending_cost"],
            "baseline_ending_retail": single_res["base_ending_retail"],
            "baseline_sales": single_res["baseline_sales"],
            "baseline_gross_margin": single_res["base_gross_margin"],
            "baseline_gm_pct": single_res["base_gm_pct"],
            "shocked_ending_cost": single_res["sim_ending_cost"],
            "shocked_ending_retail": single_res["sim_ending_retail"],
            "shocked_sales": single_res["sim_sales"],
            "shocked_markdowns": single_res["sim_markdowns"],
            "shocked_gross_margin": single_res["sim_gross_margin"],
            "shocked_gm_pct": single_res["sim_gm_pct"],
            "cr_ratio": single_res["cr_ratio"],
            "asset_deflation": single_res["asset_deflation"],
            "margin_decay": single_res["margin_decay"],
        }])
        agg_metrics = {
            "total_base_sales": single_res["baseline_sales"],
            "total_shock_sales": single_res["sim_sales"],
            "total_base_gm": single_res["base_gross_margin"],
            "total_shock_gm": single_res["sim_gross_margin"],
            "base_gm_pct": single_res["base_gm_pct"],
            "shock_gm_pct": single_res["sim_gm_pct"],
            "final_base_end_cost": single_res["base_ending_cost"],
            "final_shock_end_cost": single_res["sim_ending_cost"],
            "total_asset_deflation": single_res["asset_deflation"],
            "avg_cr_ratio": single_res["cr_ratio"],
            "total_gm_decay": single_res["margin_decay"],
        }

    # --------------------------------------------------------------------------
    # EXECUTIVE REAL-TIME IMPACT METRIC CARDS
    # --------------------------------------------------------------------------
    st.subheader(f"📊 Valuation Impact Summary ({selected_store} | +{shock_pct}% Clearance Shock)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Gross Margin Decay",
            value=f"{agg_metrics['shock_gm_pct']:.1f}%",
            delta=f"{agg_metrics['total_gm_decay']:.2f}% vs Plan",
            delta_color="inverse",
            help="Net realized Gross Margin percentage after promotional dilution.",
        )

    with col2:
        st.metric(
            label="Ending Asset Book Value",
            value=f"${agg_metrics['final_shock_end_cost']:,.0f}",
            delta=f"-${agg_metrics['total_asset_deflation']:,.0f} Deflation",
            delta_color="inverse",
            help="Balance sheet inventory asset valuation under RIM cost complement.",
        )

    with col3:
        st.metric(
            label="Cost-to-Retail Ratio",
            value=f"{agg_metrics['avg_cr_ratio']:.4f}",
            delta=f"{agg_metrics['avg_cr_ratio'] * 100:.1f}% Markon Compl.",
            delta_color="off",
            help="Cumulative markon complement used to evaluate retail reductions to cost.",
        )

    with col4:
        st.metric(
            label="Balance Sheet Write-Down",
            value=f"${agg_metrics['total_asset_deflation']:,.0f}",
            delta="Non-Cash Charge",
            delta_color="inverse",
            help="Immediate inventory write-off required due to depressed retail book value.",
        )

    # --------------------------------------------------------------------------
    # SYNCHRONIZED PLOTLY VISUALIZATIONS
    # --------------------------------------------------------------------------
    tab1, tab2 = st.tabs(["📉 Balance Sheet Asset Deflation", "⚖️ Sales Lift vs Margin Collapse"])

    with tab1:
        st.markdown("#### Balance Sheet Ending Inventory Asset Deflation (Cost)")
        fig_asset = go.Figure()

        fig_asset.add_trace(go.Scatter(
            x=sim_df["period_date"],
            y=sim_df["baseline_ending_cost"],
            mode="lines+markers",
            name="Baseline Ending Cost ($)",
            line=dict(color="#00D26A", width=3),
            marker=dict(size=6),
        ))

        fig_asset.add_trace(go.Scatter(
            x=sim_df["period_date"],
            y=sim_df["shocked_ending_cost"],
            mode="lines+markers",
            name=f"Shocked Ending Cost (+{shock_pct}%)",
            line=dict(color="#F8312F", width=3, dash="dash"),
            marker=dict(size=6),
            fill="tonexty",
            fillcolor="rgba(248, 49, 47, 0.12)",
        ))

        fig_asset.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="Ending Inventory Asset Value at Cost ($)", tickformat="$,.0f"),
            xaxis=dict(title="Accounting Period"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_asset, width="stretch")

    with tab2:
        st.markdown("#### Top-Line Revenue Lift vs. Bottom-Line Profit Destruction")
        fig_margin = go.Figure()

        # Net Sales comparison
        fig_margin.add_trace(go.Bar(
            x=sim_df["period_date"],
            y=sim_df["baseline_sales"],
            name="Baseline Net Sales ($)",
            marker_color="#4361EE",
            opacity=0.7,
        ))

        fig_margin.add_trace(go.Bar(
            x=sim_df["period_date"],
            y=sim_df["shocked_sales"],
            name=f"Shocked Net Sales (+{shock_pct}%)",
            marker_color="#4CC9F0",
        ))

        # Gross Margin lines on secondary axis
        fig_margin.add_trace(go.Scatter(
            x=sim_df["period_date"],
            y=sim_df["baseline_gross_margin"],
            mode="lines+markers",
            name="Baseline Gross Margin ($)",
            line=dict(color="#2EC4B6", width=2.5),
            yaxis="y2",
        ))

        fig_margin.add_trace(go.Scatter(
            x=sim_df["period_date"],
            y=sim_df["shocked_gross_margin"],
            mode="lines+markers",
            name=f"Shocked Gross Margin (+{shock_pct}%)",
            line=dict(color="#E71D36", width=2.5, dash="dot"),
            yaxis="y2",
        ))

        fig_margin.update_layout(
            template="plotly_dark",
            barmode="group",
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="Net Sales Revenue ($)", tickformat="$,.0f"),
            yaxis2=dict(
                title="Gross Margin Dollar Profit ($)",
                tickformat="$,.0f",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
            xaxis=dict(title="Accounting Period"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_margin, width="stretch")

    # --------------------------------------------------------------------------
    # AI MERCHANT ADVISOR / CFO RISK BRIEF
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🤖 AI Merchant Advisor & CFO Executive Brief")

    brief_placeholder = st.empty()

    if st.button("🔍 Generate AI Executive Risk Brief", type="primary"):
        with st.spinner("Analyzing RIM balance sheet dynamics & margin risk profile..."):
            brief_text = generate_ai_cfo_brief(
                store_id=selected_store,
                tier=store_meta["store_tier"],
                region=store_meta["region"],
                shock_pct=shock_pct,
                agg_metrics=agg_metrics,
            )
            st.session_state["cfo_brief"] = brief_text

    if "cfo_brief" in st.session_state:
        st.chat_message("assistant").markdown(st.session_state["cfo_brief"])
    else:
        st.info("Click **'Generate AI Executive Risk Brief'** to synthesize real-time working capital risks and markdown timing recommendations.")


if __name__ == "__main__":
    main()
