"""
ui/value_calculator.py
Presentation layer for Sub-View D: Enterprise Value Realization Calculator.
Pure domain logic resides in core.value_engine.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import plotly.graph_objects as go
import streamlit as st

from core.value_engine import (
    ValueCalculatorInputs,
    ValueRealizationOutput,
    calculate_value_realization,
)
from services.llm_advisor import (
    generate_executive_business_case,
    generate_heuristic_business_case,
)
from services.formatters import normalize_cfo_markdown


def create_value_waterfall_chart(output: ValueRealizationOutput) -> go.Figure:
    """Creates a step-by-step Plotly waterfall chart showing value accumulation."""
    x = [
        "Markdown Savings",
        "Shrinkage Reduction",
        "Labor Capacity",
        "Stockout Recapture",
        "Software Investment",
        "Net Annual Value",
    ]
    y = [
        output.annual_markdown_savings,
        output.annual_shrinkage_recovery,
        output.annual_labor_savings,
        output.annual_stockout_recapture,
        -output.annual_software_investment,
        output.net_annual_benefit,
    ]
    measure = ["relative", "relative", "relative", "relative", "relative", "total"]
    text = [f"${abs(v):,.0f}" for v in y]

    fig = go.Figure(
        go.Waterfall(
            name="Value Realization",
            orientation="v",
            measure=measure,
            x=x,
            textposition="outside",
            text=text,
            y=y,
            connector={"line": {"color": "rgba(128, 128, 128, 0.4)"}},
            decreasing={"marker": {"color": "#EF4444"}},
            increasing={"marker": {"color": "#10B981"}},
            totals={"marker": {"color": "#2563EB"}},
        )
    )
    fig.update_layout(
        title="Annual Value Realization Bridge ($)",
        waterfallgap=0.3,
        showlegend=False,
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def create_3year_horizon_chart(output: ValueRealizationOutput) -> go.Figure:
    """Creates a 36-month timeline showing cumulative gross value vs software investment."""
    months = list(range(1, 37))
    monthly_gross = output.total_annual_value / 12.0
    monthly_cost = output.annual_software_investment / 12.0
    cum_gross = [monthly_gross * m for m in months]
    cum_cost = [monthly_cost * m for m in months]
    cum_net = [g - c for g, c in zip(cum_gross, cum_cost)]

    payback_m = output.payback_period_months
    payback_val = monthly_gross * payback_m

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=cum_gross, mode="lines", name="Cumulative Gross Value", line=dict(color="#10B981", width=3)))
    fig.add_trace(go.Scatter(x=months, y=cum_cost, mode="lines", name="Software Investment", line=dict(color="#EF4444", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=months, y=cum_net, mode="lines", fill="tozeroy", name="Net Economic Benefit", line=dict(color="#38BDF8", width=2)))
    fig.add_trace(go.Scatter(x=[payback_m], y=[payback_val], mode="markers+text", name="Payback Milestone", text=[f"Payback ({payback_m:.1f} Mo)"], textposition="top left", marker=dict(size=12, color="#F59E0B", symbol="star")))

    fig.update_layout(
        title="3-Year Cumulative Value Horizon ($)",
        xaxis_title="Timeline (Months)",
        yaxis_title="Cumulative Dollars ($)",
        hovermode="x unified",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def _render_fleet_input_controls(active_rim_metrics: Optional[Dict[str, Any]] = None) -> ValueCalculatorInputs:
    """Renders the 3-column input configuration expander and builds ValueCalculatorInputs."""
    default_cr = 0.4820
    default_sales = 3_200_000.0
    default_md_pct = 28.0
    if active_rim_metrics:
        default_cr = float(active_rim_metrics.get("cost_to_retail_ratio", default_cr))
        default_sales = float(active_rim_metrics.get("store_sales", default_sales))
        default_md_pct = float(active_rim_metrics.get("markdown_rate_pct", default_md_pct))

    with st.expander("⚙️ Customize Fleet Operational Baselines & ROI Levers", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            store_count = st.slider("Store Fleet Count", 10, 1500, 120, 10)
            avg_store_sales = st.number_input("Average Annual Sales / Store ($)", 1_000_000.0, 15_000_000.0, default_sales, 100_000.0, format="%.0f")
            cost_to_retail_ratio = st.slider("Cost-to-Retail Ratio", 0.35, 0.65, min(max(default_cr, 0.35), 0.65), 0.005, format="%.4f")
        with col2:
            baseline_md_pct = st.slider("Baseline Markdown Rate %", 15.0, 45.0, min(max(default_md_pct, 15.0), 45.0), 0.5, format="%.1f%%")
            target_md_opt_pct = st.slider("Target Markdown Optimization Lift %", 1.0, 10.0, 3.5, 0.25, format="%.2f%%")
            baseline_shrink_pct = st.slider("Baseline Shrinkage Loss %", 1.0, 4.5, 2.1, 0.1, format="%.1f%%")
            target_shrink_red_pct = st.slider("Target Shrinkage Reduction %", 5.0, 25.0, 12.0, 0.5, format="%.1f%%")
        with col3:
            weekly_stockout_hours = st.slider("Store Mgr Stockout Expediting (Hrs/Wk)", 1.0, 12.0, 4.0, 0.5)
            store_mgr_rate = st.number_input("Store Manager Hourly Wage ($/Hr)", 25.0, 65.0, 38.0, 1.0, format="%.2f")
            pricing_tier = st.selectbox("Enterprise SaaS Pricing Tier", ["Standard Fleet ($50k Base + $1,200/store/yr)", "Scale Fleet ($75k Base + $950/store/yr)"], index=0)
            base_platform_fee = 75_000.0 if "Scale Fleet" in pricing_tier else 50_000.0
            per_store_annual_fee = 950.0 if "Scale Fleet" in pricing_tier else 1_200.0

    return ValueCalculatorInputs(
        store_count=store_count,
        avg_store_sales=avg_store_sales,
        baseline_markdown_pct=baseline_md_pct,
        target_markdown_opt_pct=target_md_opt_pct,
        baseline_shrinkage_pct=baseline_shrink_pct,
        target_shrink_red_pct=target_shrink_red_pct,
        cost_to_retail_ratio=cost_to_retail_ratio,
        base_platform_fee=base_platform_fee,
        per_store_annual_fee=per_store_annual_fee,
        weekly_stockout_hours_per_store=weekly_stockout_hours,
        store_mgr_hourly_rate=store_mgr_rate,
    )


def _render_value_impact_kpis(output: ValueRealizationOutput, inputs: ValueCalculatorInputs) -> None:
    """Renders the 4 core value impact metric cards."""
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    with kpi_col1:
        st.metric("Total Annual Value Realized", f"${output.total_annual_value / 1_000_000.0:.2f}M / yr", delta=f"+${output.net_annual_benefit / 1_000_000.0:.2f}M Net Benefit")
    with kpi_col2:
        st.metric("Gross Margin & Markdown Recovery", f"${output.gross_margin_recovery / 1_000_000.0:.2f}M", delta=f"${output.annual_markdown_savings / 1_000_000.0:.2f}M Retail Savings")
    with kpi_col3:
        st.metric("Shrinkage Leakage Recaptured", f"${output.annual_shrinkage_recovery / 1_000_000.0:.2f}M", delta=f"{inputs.target_shrink_red_pct:.1f}% Loss Prevention Lift")
    with kpi_col4:
        st.metric("Estimated Payback Period", f"{output.payback_period_months:.1f} Months", delta=f"Net 3-Yr ROI: {output.net_3year_roi_pct:,.0f}%", delta_color="normal")


def _render_business_case_memo_section(inputs: ValueCalculatorInputs, output: ValueRealizationOutput) -> None:
    """Renders the executive boardroom business case synthesis button and memo container."""
    st.markdown("---")
    memo_col1, memo_col2 = st.columns([1, 2])
    with memo_col1:
        st.markdown("##### 📄 Executive Boardroom Business Case")
        gen_memo_btn = st.button("📄 Generate 1-Page Business Case Summary", type="primary", width="stretch")

    case_memo_key = "cfo_business_case_text"
    case_model_key = "cfo_business_case_model"

    if case_memo_key not in st.session_state:
        st.session_state[case_memo_key] = generate_heuristic_business_case(inputs, output)
        st.session_state[case_model_key] = "Deterministic Financial Engine (Pre-Computed)"

    if gen_memo_btn:
        with st.spinner("🤖 Consulting Google Gemini Executive Strategy Advisor..."):
            memo_result = generate_executive_business_case(inputs, output)
            st.session_state[case_memo_key] = memo_result["brief_text"]
            st.session_state[case_model_key] = memo_result["model_used"]

    current_memo = normalize_cfo_markdown(st.session_state.get(case_memo_key, ""))
    current_model = st.session_state.get(case_model_key, "Google Gemini 2.5 Flash")

    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(16, 185, 129, 0.35); border-left: 4px solid #10B981; border-radius: 8px; padding: 12px 18px; margin: 14px 0; display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 0.88rem; font-weight: 700; color: #10B981; text-transform: uppercase;">📑 Steering Committee Executive Business Case ({inputs.store_count} Store Fleet | Payback: {output.payback_period_months:.1f} Mo)</span>
            <span style="background: rgba(16, 185, 129, 0.15); color: #A7F3D0; font-size: 0.75rem; padding: 4px 12px; border-radius: 12px; font-weight: 600;">{current_model}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(current_memo)


def render_value_calculator_tab(active_rim_metrics: Optional[Dict[str, Any]] = None) -> None:
    """Renders the interactive Dynamic Value-Realization Calculator."""
    st.markdown("---")
    st.markdown("#### 💰 Sub-View D: Dynamic Enterprise Value-Realization Calculator & ROI Simulator")
    st.caption("Prospective fleet operational baseline modeling: Quantifying gross dollar savings, margin recovery, and ROI.")

    inputs = _render_fleet_input_controls(active_rim_metrics)
    output = calculate_value_realization(inputs)

    _render_value_impact_kpis(output, inputs)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(create_value_waterfall_chart(output), width="stretch")
    with chart_col2:
        st.plotly_chart(create_3year_horizon_chart(output), width="stretch")

    _render_business_case_memo_section(inputs, output)
