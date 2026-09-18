"""
core/value_engine.py
Pure domain mathematical engine for enterprise retail value realization and ROI modeling.
Contains zero Streamlit, Plotly, or presentation dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ValueCalculatorInputs:
    """Enterprise baseline operational inputs provided by prospect executive."""
    store_count: int = 120
    avg_store_sales: float = 3_200_000.0          # $3.2M / store / year
    baseline_markdown_pct: float = 28.0           # 28.0% markdown dependency
    target_markdown_opt_pct: float = 3.5          # 3.5% promotional efficiency lift
    baseline_shrinkage_pct: float = 2.1           # 2.1% shrinkage loss
    target_shrink_red_pct: float = 12.0           # 12.0% shrinkage reduction
    cost_to_retail_ratio: float = 0.4820          # 48.20% markon cost complement
    base_platform_fee: float = 50_000.0           # $50k enterprise core SaaS fee
    per_store_annual_fee: float = 1_200.0         # $1,200 / store / year SaaS
    weekly_stockout_hours_per_store: float = 4.0  # 4.0 hrs/week store manager expediting
    store_mgr_hourly_rate: float = 38.00          # $38.00 blended hourly wage
    annual_stockout_incidents: int = 45           # 45 acute stockout events / store / yr
    avg_basket_value: float = 165.00              # $165 average missed order basket
    stockout_recapture_rate_pct: float = 25.0     # 25% digital intra-network recapture


@dataclass
class ValueRealizationOutput:
    """Synthesized annual enterprise financial outputs and multi-year ROI metrics."""
    # Scale Baselines
    total_fleet_sales: float
    annual_fleet_markdown_volume: float
    annual_fleet_shrinkage_loss: float

    # Pillar 1: Markdown Optimization & RIM Asset Protection
    annual_markdown_savings: float
    gross_margin_recovery: float

    # Pillar 2: Shrinkage Leakage Reduction
    annual_shrinkage_recovery: float

    # Pillar 3: Autonomous Inventory Escalation & Labor Savings
    hours_saved_annually: float
    annual_labor_savings: float
    annual_stockout_recapture: float

    # Net Economic Value & Solution Payback
    total_annual_value: float
    annual_software_investment: float
    net_annual_benefit: float
    net_3year_roi_pct: float
    payback_period_months: float


def calculate_value_realization(inputs: ValueCalculatorInputs) -> ValueRealizationOutput:
    """
    Executes the 5 deterministic enterprise retail ROI equations:
    1. Fleet Baseline Scale
    2. Pillar 1: Markdown Optimization & RIM Asset Protection
    3. Pillar 2: Shrinkage Leakage Reduction
    4. Pillar 3: Autonomous Inventory Escalation & Labor Savings
    5. Net Economic Value & Solution Payback
    """
    # 1. Fleet Baseline Scale
    total_fleet_sales = inputs.store_count * inputs.avg_store_sales
    annual_fleet_markdown_volume = total_fleet_sales * (inputs.baseline_markdown_pct / 100.0)
    annual_fleet_shrinkage_loss = total_fleet_sales * (inputs.baseline_shrinkage_pct / 100.0)

    # 2. Pillar 1: Markdown Optimization & RIM Asset Protection
    annual_markdown_savings = annual_fleet_markdown_volume * (inputs.target_markdown_opt_pct / 100.0)
    gross_margin_recovery = annual_markdown_savings * (1.0 - inputs.cost_to_retail_ratio)

    # 3. Pillar 2: Shrinkage Leakage Reduction
    annual_shrinkage_recovery = annual_fleet_shrinkage_loss * (inputs.target_shrink_red_pct / 100.0)

    # 4. Pillar 3: Autonomous Inventory Escalation & Labor Savings
    hours_saved_annually = inputs.store_count * inputs.weekly_stockout_hours_per_store * 52.0
    annual_labor_savings = hours_saved_annually * inputs.store_mgr_hourly_rate
    annual_stockout_recapture = (
        inputs.store_count
        * inputs.annual_stockout_incidents
        * inputs.avg_basket_value
        * (inputs.stockout_recapture_rate_pct / 100.0)
    )

    # 5. Net Economic Value & Solution Payback
    total_annual_value = (
        annual_markdown_savings
        + annual_shrinkage_recovery
        + annual_labor_savings
        + annual_stockout_recapture
    )

    annual_software_investment = (
        inputs.base_platform_fee + (inputs.store_count * inputs.per_store_annual_fee)
    )

    net_annual_benefit = total_annual_value - annual_software_investment

    # 3-Year Cumulative ROI: ((3 * Annual Value - 3 * Investment) / (3 * Investment)) * 100
    three_year_investment = 3.0 * annual_software_investment
    three_year_value = 3.0 * total_annual_value
    net_3year_roi_pct = (
        ((three_year_value - three_year_investment) / three_year_investment) * 100.0
        if three_year_investment > 0 else 0.0
    )

    # Payback Period (Months): (Year 1 Software Investment / Total Annual Value) * 12
    payback_period_months = (
        (annual_software_investment / total_annual_value) * 12.0
        if total_annual_value > 0 else 0.0
    )

    return ValueRealizationOutput(
        total_fleet_sales=total_fleet_sales,
        annual_fleet_markdown_volume=annual_fleet_markdown_volume,
        annual_fleet_shrinkage_loss=annual_fleet_shrinkage_loss,
        annual_markdown_savings=annual_markdown_savings,
        gross_margin_recovery=gross_margin_recovery,
        annual_shrinkage_recovery=annual_shrinkage_recovery,
        hours_saved_annually=hours_saved_annually,
        annual_labor_savings=annual_labor_savings,
        annual_stockout_recapture=annual_stockout_recapture,
        total_annual_value=total_annual_value,
        annual_software_investment=annual_software_investment,
        net_annual_benefit=net_annual_benefit,
        net_3year_roi_pct=net_3year_roi_pct,
        payback_period_months=payback_period_months,
    )
