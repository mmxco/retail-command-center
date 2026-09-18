"""
services/llm_advisor.py
Centralized Google Gemini API client creation, Windows SSL compatibility,
priority candidate model fallback cascade, and deterministic heuristic fallback memos.
"""
from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

# Attempt GenAI SDK import and configure AFC warning suppression
try:
    from google import genai
    from google.genai import types
    from google.genai.models import Models, AsyncModels

    # Mask upstream automatic function calling warning
    Models._logged_afc_warning = True
    AsyncModels._logged_afc_warning = True

    class _AFCWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "automatic function calling (AFC)" not in record.getMessage()

    logging.getLogger("google_genai.models").addFilter(_AFCWarningFilter())
    logging.getLogger("google_genai").addFilter(_AFCWarningFilter())
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    types = None  # type: ignore

from services.formatters import normalize_cfo_markdown
from core.value_engine import ValueCalculatorInputs, ValueRealizationOutput


# Default candidate model cascade (in priority order)
DEFAULT_CANDIDATE_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash-lite",
]


@dataclass
class RIMDashboardState:
    """Live snapshot of recalculated RIM dashboard metrics."""
    store_id: str
    store_tier: str
    region: str
    markdown_shock_pct: float             # e.g., +25% clearance shock
    cost_to_retail_ratio: float           # e.g., 0.4820 (48.20%)
    baseline_ending_retail: float
    shocked_ending_retail: float
    baseline_ending_cost: float
    shocked_ending_cost: float
    balance_sheet_asset_deflation: float  # Baseline cost - Shocked cost
    gross_margin_decay_pct: float         # Baseline GM% - Shocked GM%
    projected_gmroi: float                # Annualized GMROI under shock
    nearest_peers: List[str] = field(default_factory=list)  # Top 3 KNN peer store IDs
    archetype_label: str = "Balanced Regional Performers"
    sales_lift_pct: float = 0.0
    baseline_gm_pct: float = 58.4
    shocked_gm_pct: float = 51.4


_USE_ENV = object()


def get_gemini_client(api_key: Any = _USE_ENV) -> Optional[Any]:
    """
    Creates and returns a Google Gemini client configured with client_args={'verify': False}
    to ensure reliable execution on Windows systems with custom root certificate bundles.
    """
    if not GENAI_AVAILABLE:
        return None
    if api_key is _USE_ENV:
        key = os.environ.get("GEMINI_API_KEY")
    else:
        key = api_key
    if not key:
        return None
    try:
        return genai.Client(
            api_key=key,
            http_options=types.HttpOptions(client_args={"verify": False}),
        )
    except Exception:
        return None


def call_gemini_cascade(
    prompt: str,
    system_instruction: str = "",
    candidate_models: Optional[List[str]] = None,
    api_key: Any = _USE_ENV,
    preferred_model: Optional[str] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 4096,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Executes a prompt across candidate Gemini models in cascade order with retry fallback.
    Returns (generated_text, model_name) on success, or (None, None) on complete failure.
    """
    client = get_gemini_client(api_key)
    if not client:
        return None, None

    models = list(candidate_models or DEFAULT_CANDIDATE_MODELS)
    if preferred_model and preferred_model in models:
        models.remove(preferred_model)
        models.insert(0, preferred_model)
    elif preferred_model:
        models.insert(0, preferred_model)

    for model_name in models:
        try:
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                system_instruction=system_instruction,
            )
            chat = client.chats.create(model=model_name, config=config)
            response = chat.send_message(prompt)
            if response and response.text and len(response.text.strip()) > 50:
                return response.text.strip(), model_name
        except Exception:
            continue

    return None, None


def build_cfo_reasoning_prompt(state: RIMDashboardState) -> str:
    """Constructs prompt for executive CFO advisory brief."""
    peers_str = ", ".join(state.nearest_peers) if state.nearest_peers else "STORE_101, STORE_105"

    return rf"""You are an elite Retail Chief Financial Officer (CFO) and Merchandising Risk Strategist.
Review the following real-time Retail Inventory Method (RIM) balance sheet stress test:

### STORE OPERATIONAL PROFILE
- **Store Identifier**: {state.store_id} ({state.store_tier}, Region: {state.region})
- **ML Cluster Archetype**: {state.archetype_label}
- **Nearest Sibling Peers (KNN)**: {peers_str}

### REAL-TIME STRESS TEST METRICS (+{state.markdown_shock_pct:.0f}% CLEARANCE MARKDOWN SHOCK)
- **Clearance Markdown Shock Depth**: +{state.markdown_shock_pct:.0f}% over planned budget
- **Effective Cost-to-Retail Ratio (Markon Complement)**: {state.cost_to_retail_ratio:.4f} ({state.cost_to_retail_ratio * 100:.2f}%)
- **Ending Inventory at Retail**: Planned ${state.baseline_ending_retail:,.2f} -> Shocked ${state.shocked_ending_retail:,.2f} (Delta: -${state.baseline_ending_retail - state.shocked_ending_retail:,.2f})
- **Ending Inventory at Cost (Asset Valuation)**: Planned ${state.baseline_ending_cost:,.2f} -> Shocked ${state.shocked_ending_cost:,.2f}
- **Balance Sheet Asset Deflation (Non-Cash Write-Down)**: ${state.balance_sheet_asset_deflation:,.2f}
- **Realized Sales Volume Lift (Elasticity)**: +{state.sales_lift_pct:.1f}%
- **Gross Margin Performance**: Planned {state.baseline_gm_pct:.1f}% -> Shocked {state.shocked_gm_pct:.1f}% (Decay: {state.gross_margin_decay_pct:.2f}%)
- **Projected Annualized GMROI**: {state.projected_gmroi:.2f}x

### EXECUTIVE ASSIGNMENT
Deliver an authoritative, boardroom-ready CFO Working Capital Risk Brief composed of exactly 3 formatted bullet points:

1. **Working Capital & Balance Sheet Asset Deflation**:
   Explain how the deterministic Retail Inventory Method (RIM) accounting formula (Ending Inventory at Retail * Cost Complement) mechanically precipitates this non-cash asset impairment of **${state.balance_sheet_asset_deflation:,.2f}**. Explain the direct downstream threat to revolving credit facility borrowing bases, asset-backed loan (ABL) collateral covenants, and working capital liquidity.

2. **Top-Line Volume Illusion vs. Margin Destruction**:
   Deconstruct the commercial fallacy of the +{state.sales_lift_pct:.1f}% top-line sales volume lift against the {state.gross_margin_decay_pct:.2f}% gross margin collapse. Prove why incremental markdown velocity destroys dollar gross profit after factoring replacement wholesale purchase costs and freight handling.

3. **Actionable Merchandising Guardrails & Peer Rebalance Protocol**:
   Establish strict, quantitative operational guardrails. Recommend a hard promotional discount ceiling (e.g., 15-20% max) and propose an autonomous intra-network stock rebalancing sweep shifting slow-moving units away from {state.store_id} into high-velocity peer stores ({peers_str}) to preserve full ticket retail value.

### FORMATTING & TYPOGRAPHY MANDATE (STRICT):
- Write in authoritative, quantitative, boardroom-ready executive finance prose.
- Do NOT use LaTeX math syntax, LaTeX delimiters ($...$ or $$...$$), or LaTeX macros.
- Format all mathematical relationships in plain business text.
- Never place math mode delimiters around store names, variables, or metrics.
- Write dollar signs directly as monetary figures without LaTeX formatting.
"""


def generate_heuristic_cfo_brief(state: RIMDashboardState) -> str:
    """Generates a deterministic rule-based executive CFO risk brief."""
    peers_str = ", ".join(state.nearest_peers) if state.nearest_peers else "STORE_101, STORE_105"

    return rf"""### 1. Working Capital & Balance Sheet Asset Deflation (${state.balance_sheet_asset_deflation:,.0f} Forced Non-Cash Write-Down)
Under Retail Inventory Method (RIM) accounting rules, accelerating clearance markdowns by +{state.markdown_shock_pct:.0f}% aggressively discounts Ending Inventory at Retail without generating commensurate gross cash proceeds. Because the store's markon cost complement remains pegged at **{state.cost_to_retail_ratio:.4f}** ({state.cost_to_retail_ratio * 100:.2f}%), the corporate balance sheet is forced to absorb an immediate, non-cash inventory asset write-down of **${state.balance_sheet_asset_deflation:,.2f}**. This reduction directly compresses tangible net worth, degrades current ratios, and tightens borrowing base availability under revolving asset-backed loan (ABL) covenants.

### 2. Top-Line Volume Illusion vs. Bottom-Line Profit Collapse ({state.gross_margin_decay_pct:.2f}% Margin Decay)
While promotional price elasticity produces an apparent +{state.sales_lift_pct:.1f}% surge in unit movement, realized gross margin decays precipitously from **{state.baseline_gm_pct:.1f}%** down to **{state.shocked_gm_pct:.1f}%**. At this steep markdown cadence, each incremental unit sold fails to cover allocated distribution and inventory replacement costs. The illusion of top-line revenue acceleration conceals cumulative gross profit dollar destruction across the regional footprint.

### 3. Actionable Merchandising Guardrails & Peer Rebalance Protocol
* **Clearance Throttle**: Enforce a mandatory **15% maximum promotional discount ceiling** on {state.store_id} to halt runaway asset deflation.
* **Intra-Network Rebalance Sweep**: Rather than taking dilutive store-level clearance markdowns, trigger automated peer-to-peer inventory transfers to high-turn sister locations (**{peers_str}**) where consumer demand supports full-price ticket realization.
* **GMROI Target Floor**: Maintain a strict GMROI guardrail of **{state.projected_gmroi:.2f}x** before approving further end-of-season markdown allowances.
"""


def generate_cfo_risk_brief(
    state: RIMDashboardState,
    api_key: Any = _USE_ENV,
    preferred_model: str = "gemini-2.5-pro",
) -> Dict[str, Any]:
    """Invokes Google Gemini with model cascading or falls back to heuristic brief."""
    if api_key is not _USE_ENV and not api_key:
        return {
            "brief_text": normalize_cfo_markdown(generate_heuristic_cfo_brief(state)),
            "model_used": "Deterministic Rule-Based Risk Engine (Offline Mode)",
            "success": True,
            "engine": "heuristic",
        }

    prompt = build_cfo_reasoning_prompt(state)
    sys_instruction = (
        "You are an executive Chief Financial Officer and Senior Retail Risk Analyst. "
        "Produce rigorous, boardroom-grade financial diagnostics with quantitative precision."
    )
    raw_text, model_used = call_gemini_cascade(
        prompt=prompt,
        system_instruction=sys_instruction,
        api_key=api_key,
        preferred_model=preferred_model,
    )

    if raw_text and model_used:
        return {
            "brief_text": normalize_cfo_markdown(raw_text),
            "model_used": f"Google {model_used} (Executive Reasoning)",
            "success": True,
            "engine": "gemini",
        }

    return {
        "brief_text": normalize_cfo_markdown(generate_heuristic_cfo_brief(state)),
        "model_used": "Deterministic Rule-Based Risk Engine (Offline Mode)",
        "success": True,
        "engine": "heuristic",
    }


def build_business_case_prompt(
    inputs: Any,
    output: Any,
) -> str:
    """Constructs prompt for executive business case memo."""
    store_count = getattr(inputs, "store_count", 120)
    avg_store_sales = getattr(inputs, "avg_store_sales", 2_500_000.0)
    baseline_markdown_pct = getattr(inputs, "baseline_markdown_pct", 15.0)
    baseline_shrinkage_pct = getattr(inputs, "baseline_shrinkage_pct", 1.5)
    cost_to_retail_ratio = getattr(inputs, "cost_to_retail_ratio", 0.45)
    target_markdown_opt_pct = getattr(inputs, "target_markdown_opt_pct", 15.0)
    target_shrink_red_pct = getattr(inputs, "target_shrink_red_pct", 25.0)

    total_fleet_sales = getattr(output, "total_fleet_sales", 300_000_000.0)
    annual_fleet_markdown_volume = getattr(output, "annual_fleet_markdown_volume", 45_000_000.0)
    annual_fleet_shrinkage_loss = getattr(output, "annual_fleet_shrinkage_loss", 4_500_000.0)
    annual_markdown_savings = getattr(output, "annual_markdown_savings", 3_375_000.0)
    gross_margin_recovery = getattr(output, "gross_margin_recovery", 3_375_000.0)
    annual_shrinkage_recovery = getattr(output, "annual_shrinkage_recovery", 1_125_000.0)
    hours_saved_annually = getattr(output, "hours_saved_annually", 18_720.0)
    annual_labor_savings = getattr(output, "annual_labor_savings", 655_200.0)
    annual_stockout_recapture = getattr(output, "annual_stockout_recapture", 675_000.0)
    total_annual_value = getattr(output, "total_annual_value", 5_902_110.0)
    annual_software_investment = getattr(output, "annual_software_investment", 194_000.0)
    net_annual_benefit = getattr(output, "net_annual_benefit", 5_708_110.0)
    payback_period_months = getattr(output, "payback_period_months", 0.39)
    net_3year_roi_pct = getattr(output, "net_3year_roi_pct", 2842.0)

    return f"""You are an executive Chief Financial Officer and Senior Retail Transformation Partner.
Synthesize an authoritative, boardroom-grade 1-page Executive Business Case Memo for the Retail Steering Committee based on the following real-time fleet valuation metrics:

### FLEET OPERATIONAL BASELINE:
- Store Fleet Count: {store_count} stores
- Average Store Sales: ${avg_store_sales:,.2f}
- Total Annual Fleet Sales: ${total_fleet_sales:,.2f}
- Baseline Markdown Volume: ${annual_fleet_markdown_volume:,.2f} ({baseline_markdown_pct:.1f}%)
- Baseline Shrinkage Loss: ${annual_fleet_shrinkage_loss:,.2f} ({baseline_shrinkage_pct:.1f}%)
- Active Cost-to-Retail Ratio: {cost_to_retail_ratio:.4f}

### MODELED VALUE REALIZATION (ANNUAL):
- Markdown Optimization Savings: ${annual_markdown_savings:,.2f} (+{target_markdown_opt_pct:.1f}% lift)
- Gross Margin Recovered: ${gross_margin_recovery:,.2f}
- Shrinkage Leakage Recaptured: ${annual_shrinkage_recovery:,.2f} (-{target_shrink_red_pct:.1f}% reduction)
- Store Manager Labor Saved: {hours_saved_annually:,.0f} hours (${annual_labor_savings:,.2f})
- Stockout Walkaway Sales Recaptured: ${annual_stockout_recapture:,.2f}
- TOTAL ANNUAL VALUE REALIZED: ${total_annual_value:,.2f}

### INVESTMENT & PAYBACK METRICS:
- Annual Software Investment: ${annual_software_investment:,.2f}
- Net Annual Economic Benefit: ${net_annual_benefit:,.2f}
- Payback Period: {payback_period_months:.1f} Months
- 3-Year Cumulative ROI: {net_3year_roi_pct:.1f}%

### EXECUTIVE ASSIGNMENT:
Draft an executive 4-section business case memo structured as follows:
#### 1. Executive Financial Summary & Payback Horizon
#### 2. Cost of Inaction (COI) & Margin Bleed Analysis
#### 3. Balance Sheet Asset Protection (RIM) & Working Capital Impact
#### 4. Steering Committee Capital Allocation Recommendation

### FORMATTING & TYPOGRAPHY MANDATE (STRICT):
- Write in authoritative, quantitative C-suite prose.
- Do NOT use LaTeX math syntax, LaTeX delimiters ($...$ or $$...$$), or LaTeX macros.
- Format all formulas in plain business text.
- Write currency figures directly without LaTeX math delimiters.
"""


def generate_heuristic_business_case(
    inputs: Any,
    output: Any,
) -> str:
    """Deterministic fallback business case brief."""
    store_count = getattr(inputs, "store_count", 120)
    target_markdown_opt_pct = getattr(inputs, "target_markdown_opt_pct", 15.0)

    total_fleet_sales = getattr(output, "total_fleet_sales", 300_000_000.0)
    total_annual_value = getattr(output, "total_annual_value", 5_902_110.0)
    annual_software_investment = getattr(output, "annual_software_investment", 194_000.0)
    net_annual_benefit = getattr(output, "net_annual_benefit", 5_708_110.0)
    payback_period_months = getattr(output, "payback_period_months", 0.39)
    net_3year_roi_pct = getattr(output, "net_3year_roi_pct", 2842.0)
    annual_fleet_markdown_volume = getattr(output, "annual_fleet_markdown_volume", 45_000_000.0)
    annual_fleet_shrinkage_loss = getattr(output, "annual_fleet_shrinkage_loss", 4_500_000.0)
    gross_margin_recovery = getattr(output, "gross_margin_recovery", 3_375_000.0)
    hours_saved_annually = getattr(output, "hours_saved_annually", 18_720.0)
    annual_labor_savings = getattr(output, "annual_labor_savings", 655_200.0)
    annual_stockout_recapture = getattr(output, "annual_stockout_recapture", 675_000.0)

    three_yr_net = (3.0 * total_annual_value) - (3.0 * annual_software_investment)

    brief = rf"""### Executive Business Case: Fleet AI Value Realization Memo

#### 1. Executive Financial Summary & Payback Horizon
Across our modeled enterprise baseline of **{store_count} stores** generating **${total_fleet_sales:,.0f}** in annual sales, deploying the autonomous merchandising and replenishment cockpit produces **${total_annual_value:,.0f}** in recurring annual gross value. Against an estimated annual software investment of **${annual_software_investment:,.0f}**, the solution delivers **${net_annual_benefit:,.0f}** in net annual economic benefit, achieving full capital payback in **{payback_period_months:.1f} months** and generating a **{net_3year_roi_pct:.0f}% 3-year cumulative ROI** (${three_yr_net:,.0f} net 36-month cash value).

#### 2. Cost of Inaction (COI) & Margin Bleed Analysis
Without operational intervention, the fleet passively bleeds **${annual_fleet_markdown_volume:,.0f}** annually through clearance markdown discounting and loses an additional **${annual_fleet_shrinkage_loss:,.0f}** to inventory shrinkage and physical discrepancy write-offs. Delaying deployment by a single fiscal quarter costs the organization approximately **${total_annual_value / 4.0:,.0f}** in unrecoverable margin and working capital deterioration.

#### 3. Balance Sheet Asset Protection (RIM) & Working Capital Impact
Under Retail Inventory Method (RIM) accounting rules, the targeted +{target_markdown_opt_pct:.1f}% markdown timing optimization directly preserves **${gross_margin_recovery:,.0f}** in full-ticket gross margin while safeguarding ending inventory asset valuations on corporate balance sheets. Concurrently, recapturing {hours_saved_annually:,.0f} store manager expediting hours frees **${annual_labor_savings:,.0f}** in managerial capacity for frontline customer engagement, while automated store-to-store rebalancing recaptures **${annual_stockout_recapture:,.0f}** in walkaway stockout sales.

#### 4. Steering Committee Capital Allocation Recommendation
The Retail Steering Committee is advised to authorize immediate Phase 1 pilot implementation. With a payback timeline under {payback_period_months + 0.5:.0f} months and low risk profile, this capital commitment meets all corporate hurdle rates for high-velocity software investments.
"""
    return normalize_cfo_markdown(brief)


def generate_executive_business_case(
    inputs: Any,
    output: Any,
    api_key: Any = _USE_ENV,
) -> Dict[str, Any]:
    """Invokes Google Gemini with model cascading or falls back to heuristic business case."""
    if api_key is not _USE_ENV and not api_key:
        return {
            "brief_text": generate_heuristic_business_case(inputs, output),
            "model_used": "Deterministic Financial Engine (Offline Mode)",
            "success": True,
        }

    prompt = build_business_case_prompt(inputs, output)
    sys_instruction = (
        "You are an executive Chief Financial Officer and Senior Retail Strategy Partner. "
        "Produce rigorous, boardroom-ready commercial diagnostics with quantitative precision."
    )
    raw_text, model_used = call_gemini_cascade(
        prompt=prompt,
        system_instruction=sys_instruction,
        api_key=api_key,
    )

    if raw_text and model_used:
        return {
            "brief_text": normalize_cfo_markdown(raw_text),
            "model_used": f"Google {model_used} (Executive Reasoning)",
            "success": True,
        }

    return {
        "brief_text": generate_heuristic_business_case(inputs, output),
        "model_used": "Deterministic Financial Engine (Offline Mode)",
        "success": True,
    }


def generate_ai_cfo_brief_text(
    store_id: str,
    archetype: str,
    tier: str,
    region: str,
    shock_pct: int,
    metrics: Dict[str, Any],
) -> str:
    """
    Backward-compatibility shim for tests/test_command_center_tabs.py:test_tab_rim_ai_cfo_brief_fallback.
    Provides structured 3-bullet CFO brief without crashing.
    """
    sales_lift_pct = metrics.get("sales_lift_pct", 0.0)
    gm_decay = metrics.get("total_gm_decay", 0.0)
    asset_deflation = metrics.get("total_asset_deflation", 0.0)
    cr_ratio = metrics.get("avg_cr_ratio", 0.40)
    base_gm_pct = metrics.get("base_gm_pct", 55.0)
    shock_gm_pct = metrics.get("shock_gm_pct", 50.0)
    top_peer = metrics.get("top_peer", "STORE_101")

    return normalize_cfo_markdown(rf"""
* **Working Capital & Balance Sheet Asset Deflation (${asset_deflation:,.0f} Write-Down)**:
  Under the Retail Inventory Method (RIM), accelerating promotional markdowns by +{shock_pct}% lowers Ending Inventory at Retail without generating commensurate gross cash proceeds. Applying the store's markon cost complement of **{cr_ratio:.4f}** forces an immediate balance sheet write-down of **${asset_deflation:,.2f}** at cost, directly contracting borrowing base collateral and working capital liquidity.

* **Top-Line Volume Illusion vs. Profit Destruction ({gm_decay:.2f}% Margin Decay)**:
  While markdown elasticity generates an apparent +{sales_lift_pct:.1f}% top-line unit lift, realized Gross Margin collapses from **{base_gm_pct:.1f}%** to **{shock_gm_pct:.1f}%**. At this clearance depth, incremental units are liquidated below fully-burdened replacement and supply chain expediting costs, resulting in net economic margin destruction.

* **Executive Merchandising Guardrails & Mitigation Protocol**:
  Establish an immediate **15% maximum promotional discount throttle** on this store cluster. Rather than taking localized markdowns in {store_id}, initiate an automated intra-network stock rebalance (via Tab 2 Autonomous Ops) to transfer slow-moving styles to high-velocity Flagship locations ({top_peer}) where full ticket price can be recovered.
""")
