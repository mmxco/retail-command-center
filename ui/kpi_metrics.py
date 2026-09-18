"""
ui/kpi_metrics.py
Modular KPI metric card presentation and defensive row parsing.
"""
from __future__ import annotations
from typing import Any, Tuple, List
import streamlit as st


def _extract_metric(row: Any, keys: List[str], default: float = 0.0) -> float:
    """Safely extracts a floating-point metric value from a dictionary or pandas Series."""
    if hasattr(row, "__getitem__"):
        for k in keys:
            if k in row:
                try:
                    return float(row[k])
                except (ValueError, TypeError):
                    pass
    return default


def _resolve_row(df_store_metrics: Any) -> Any:
    """Extracts the first row if input is a non-empty DataFrame, else returns input directly."""
    if hasattr(df_store_metrics, "iloc") and hasattr(df_store_metrics, "columns") and len(df_store_metrics) > 0:
        return df_store_metrics.iloc[0]
    return df_store_metrics


def _parse_kpi_metrics(df_store_metrics: Any = None) -> Tuple[float, str, float, float, float, str]:
    """
    Defensive extraction of retail KPI metrics from polymorphic input (None, dict, Series, DataFrame).
    Returns: (gmroi_val, gmroi_delta_color, cr_val, turn_val, shrink_val, shrink_delta_color).
    """
    if df_store_metrics is None:
        return 2.42, "normal", 48.20, 4.1, 1.85, "normal"

    row = _resolve_row(df_store_metrics)

    gmroi_val = _extract_metric(row, ["gmroi", "GMROI", "normalized_gmroi"], 2.42)
    gmroi_color = "normal" if gmroi_val >= 2.0 else "inverse"

    raw_cr = _extract_metric(row, ["cost_to_retail_ratio", "cr_ratio", "avg_cr_ratio", "cost_to_retail"], 48.20)
    cr_val = raw_cr * 100.0 if raw_cr <= 1.0 else raw_cr

    raw_turn = _extract_metric(row, ["turn_rate", "inventory_turns", "sell_through_velocity"], 4.1)
    turn_val = round(raw_turn * 12.0, 1) if raw_turn < 1.0 else round(raw_turn, 1)

    raw_shrink = _extract_metric(row, ["shrinkage_rate", "shrink_rate", "shrinkage"], 1.85)
    shrink_val = raw_shrink * 100.0 if raw_shrink <= 0.15 else raw_shrink
    shrink_color = "inverse" if shrink_val > 3.0 else "normal"

    return gmroi_val, gmroi_color, cr_val, turn_val, shrink_val, shrink_color


def render_retail_kpis(df_store_metrics: Any = None) -> None:
    """
    Renders four high-visibility enterprise KPI summary metric cards with
    explanatory tooltips educating non-finance stakeholders on core retail accounting mechanics.
    """
    gmroi, gmroi_c, cr, turn, shrink, shrink_c = _parse_kpi_metrics(df_store_metrics)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="GMROI",
            value=f"{gmroi:.2f}x",
            delta="Benchmark: > 2.0x",
            delta_color=gmroi_c,
            help="Measures inventory productivity: Gross Margin ($) divided by Average Inventory at Cost.",
        )
    with col2:
        st.metric(
            label="Cost-to-Retail Ratio",
            value=f"{cr:.2f}%",
            delta="Initial Markup Target: 51.80%",
            delta_color="off",
            help="Foundational RIM multiplier: (Goods Available at Cost) / (Goods Available at Retail).",
        )
    with col3:
        st.metric(
            label="Annual Inventory Turn Rate",
            value=f"{turn:.1f} Turns/yr",
            delta="Apparel Target: 3.5 - 4.5",
            delta_color="normal",
            help="Net Sales divided by Average Inventory at Retail. Demonstrates inventory velocity.",
        )
    with col4:
        st.metric(
            label="Shrinkage Anomaly Index",
            value=f"{shrink:.2f}%",
            delta="Fleet Anomaly Threshold: < 3.0%",
            delta_color=shrink_c,
            help="Physical inventory shrinkage as a percentage of retail sales.",
        )
