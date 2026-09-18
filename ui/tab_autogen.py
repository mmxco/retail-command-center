"""
ui/tab_autogen.py
Modular presentation layer for Tab 2: Autonomous Multi-Agent Replenishment Ops (AutoGen).
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union
import streamlit as st

from services.data_loaders import load_autogen_scenarios
from services.formatters import escape_dollars, sanitize_markdown_dollars

# Module-level exports for backward compatibility
SCENARIOS = load_autogen_scenarios()
CONSENSUS_TAG = "RESOLUTION CONSENSUS"

PERSONA_CONFIG = {
    "Store_Ops_Lead": {
        "name": "Store Operations Lead (Frontline Stockout Sensor)",
        "avatar": "🏬",
        "badge_color": "#2563eb",
        "subtitle": "Store #104 (Denver Downtown) • Priority Escalation",
    },
    "Inventory_Merchandising_Lead": {
        "name": "Inventory Merchandising Lead (Multi-Echelon Balancer)",
        "avatar": "📊",
        "badge_color": "#10b981",
        "subtitle": "Regional Inventory Optimizer • Stock Transfer Rebalancing",
    },
    "Vendor_Procurement_Lead": {
        "name": "Vendor Procurement Lead (Commercial Contract Negotiator)",
        "avatar": "📦",
        "badge_color": "#f59e0b",
        "subtitle": "Supplier Terms & Rush Expediting • Fast-Track Sourcing",
    },
    "Admin_Executor": {
        "name": "Autonomous Governance & ERP Executor",
        "avatar": "⚙️",
        "badge_color": "#8b5cf6",
        "subtitle": "Audit Trail & EDI 850 Generation • ERP System of Record",
    },
    "Consultant_Approver": {
        "name": "Executive Steering Committee Approver",
        "avatar": "🛡️",
        "badge_color": "#d97706",
        "subtitle": "Human-In-The-Loop Authority • Working Capital Exception Sign-Off",
    },
}


def generate_capped_resolution(
    scenario_data: Dict[str, Any],
    cap_val: float,
    target_store: str,
    sku: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Generates dynamic capped messages and metrics when user chooses Option B."""
    wholesale_cost = 39.00
    if scenario_data["id"] == "viral_spike":
        fixed_freight = 1170.00
        allowed_for_goods = max(0.0, cap_val - fixed_freight)
        vendor_units = max(10, min(scenario_data["metrics"]["vendor_units"], int(allowed_for_goods / wholesale_cost)))
        surcharges = fixed_freight
        actual_spend = round(vendor_units * wholesale_cost + fixed_freight, 2)
    else:
        freight_ratio = cap_val / scenario_data["metrics"]["total_commitment"]
        surcharges = round(5350.00 * freight_ratio, 2)
        allowed_for_goods = max(0.0, cap_val - surcharges)
        vendor_units = max(10, min(scenario_data["metrics"]["vendor_units"], int(allowed_for_goods / wholesale_cost)))
        actual_spend = round(vendor_units * wholesale_cost + surcharges, 2)

    transfer_units = scenario_data["metrics"]["transfer_units"]
    allocated_units = vendor_units + transfer_units

    pre_messages = scenario_data["messages"][:3]
    consultant_msg = {
        "sender": "Consultant_Approver",
        "content": sanitize_markdown_dollars(
            f"🛡️ **HUMAN-IN-THE-LOOP (HITL) CAPITAL GOVERNANCE OVERRIDE**:\n"
            f"As Executive Committee Approver, I have authorized a partial working capital ceiling of **${cap_val:,.2f}** "
            f"for {target_store} on {sku}. Vendor purchase requisition is capped at **{vendor_units} units** (${actual_spend:,.2f} commit). "
            f"Proceed with EDI 850 transmission."
        ),
    }
    po_trace = {
        "tool": "generate_purchase_order",
        "args": {
            "vendor_id": "VEND-PACIFIC-DENIM",
            "sku": sku,
            "units": vendor_units,
            "unit_wholesale_cost": wholesale_cost,
            "authorized_cap": cap_val,
            "consultant_approved": True,
        },
        "result": f"SUCCESS: Capped EDI 850 PO-850-HITL generated for {vendor_units} units at ${actual_spend:,.2f}.",
    }
    admin_msg = {
        "sender": "Admin_Executor",
        "content": sanitize_markdown_dollars(
            f"⚙️ **ERP EXECUTION CONFIRMED**: Generated capped Purchase Order PO-850-HITL for **{vendor_units} vendor units**. "
            f"Combined with **{transfer_units} transfer units**, total fulfillment is **{allocated_units} units** (${actual_spend:,.2f})."
        ),
        "tool_call": po_trace,
    }
    merch_msg = {
        "sender": "Inventory_Merchandising_Lead",
        "content": sanitize_markdown_dollars(
            f"{CONSENSUS_TAG}: Sourced {allocated_units} units ({transfer_units}u sister store transfer + {vendor_units}u vendor drop-ship). "
            f"Fulfillment commitment: ${actual_spend:,.2f}."
        ),
    }
    all_messages = pre_messages + [consultant_msg, admin_msg, merch_msg]
    capped_metrics = {
        "allocated_units": allocated_units,
        "transfer_units": transfer_units,
        "vendor_units": vendor_units,
        "transfer_source": scenario_data["metrics"]["transfer_source"],
        "expediting_surcharges": surcharges,
        "total_commitment": actual_spend,
        "governance_badge": f"🛡️ HITL CAPPED (${cap_val:,.0f})",
        "sla_hours": scenario_data["metrics"]["sla_hours"],
    }
    return all_messages, capped_metrics


def _render_hitl_header_and_metrics(tot_spend: float, ceiling: float, delta_over: float, sla_hours: int) -> None:
    """Renders the executive exception banner and 4 top metrics in the HITL dialog."""
    st.markdown(
        rf"""
        <div style="border-left: 4px solid #d97706; padding: 12px 18px; background: rgba(217, 119, 6, 0.08); border-radius: 6px; margin-bottom: 18px;">
            <div style="font-size: 1.05rem; font-weight: 800; color: #b45309; margin-bottom: 4px;">CAPITAL AUTHORITY EXCEPTION REQUIRED</div>
            <div style="font-size: 0.92rem; color: var(--text-color, #1e293b); line-height: 1.5;">
                The proposed replenishment requisition totaling <strong>\${tot_spend:,.2f}</strong> exceeds the corporate autonomous limit of 
                <strong>\${ceiling:,.2f}</strong> by <strong style="color: #dc2626;">+\${delta_over:,.2f}</strong>. Executive authorization is required.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric("Proposed Spend", f"${tot_spend:,.2f}")
    with k2: st.metric("Autonomous Limit", f"${ceiling:,.2f}")
    with k3: st.metric("Unbudgeted Delta", f"+${delta_over:,.2f}", delta="Ceiling Breach", delta_color="inverse")
    with k4: st.metric("Committed SLA", f"{sla_hours} Hours")


def _render_hitl_option_a_full(scenario_data: Dict[str, Any], tot_spend: float, vendor_units: int) -> None:
    """Renders Option A: Full Requisition Authorization."""
    st.markdown("##### 1️⃣ Option A: Authorize Full Requisition (100% Demand Met)")
    st.caption(f"Approve overriding ceiling to source all **{vendor_units} vendor units** (${tot_spend:,.2f}).")
    if st.button(f"🛡️ Authorize Full Requisition (${tot_spend:,.2f})", type="primary", width="stretch", key="btn_modal_auth"):
        st.session_state["autogen_hitl_pending"] = False
        st.session_state["autogen_hitl_decision"] = "APPROVED"
        st.session_state["autogen_messages"] = scenario_data["messages"]
        st.session_state["autogen_metrics"] = scenario_data["metrics"]
        st.rerun()


def _render_hitl_option_b_capped(scenario_data: Dict[str, Any], tot_spend: float, target_store: str, sku: str, vendor_units: int, transfer_units: int, total_units: int) -> None:
    """Renders Option B: Custom Working Capital Ceiling Slider."""
    st.markdown("---")
    st.markdown("##### 2️⃣ Option B: Custom Working Capital Ceiling (Interactive Overage Slider)")
    cap_val = st.slider("Authorized Executive Spending Cap ($)", min_value=25000, max_value=int(tot_spend), value=int((25000 + tot_spend) / 2), step=500, format="$%d", key="hitl_modal_cap_slider")

    if scenario_data["id"] == "viral_spike":
        allowed = max(0.0, cap_val - 1170.00)
        capped_u = max(10, min(vendor_units, int(allowed / 39.00)))
    else:
        surcharges = round(5350.00 * (cap_val / tot_spend), 2)
        allowed = max(0.0, cap_val - surcharges)
        capped_u = max(10, min(vendor_units, int(allowed / 39.00)))

    fulfilled = capped_u + transfer_units
    fill_rate = (fulfilled / total_units) * 100.0 if total_units > 0 else 100.0

    st.markdown(
        rf"""
        <div style="background: rgba(217, 119, 6, 0.07); border: 1px solid rgba(217, 119, 6, 0.25); border-radius: 6px; padding: 10px 14px; margin-bottom: 12px;">
            <div style="display: flex; justify-content: space-between; font-size: 0.88rem;">
                <span>Authorized Cap: <strong style="color: #b45309;">\${cap_val:,.2f}</strong></span>
                <span>Fulfillment Rate: <strong style="color: #059669;">{fill_rate:.1f}%</strong> ({fulfilled}/{total_units}u)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(f"⚖️ Authorize Capped Expenditure (${cap_val:,.2f})", width="stretch", key="btn_modal_cap"):
        all_msgs, cap_mets = generate_capped_resolution(scenario_data, cap_val, target_store, sku)
        st.session_state["autogen_hitl_pending"] = False
        st.session_state["autogen_hitl_decision"] = "APPROVED_CAPPED"
        st.session_state["autogen_messages"] = all_msgs
        st.session_state["autogen_metrics"] = cap_mets
        st.rerun()


def _render_hitl_option_c_reject(scenario_data: Dict[str, Any]) -> None:
    """Renders Option C: Strict Rejection of Overages."""
    st.markdown("---")
    st.markdown(r"##### 3️⃣ Option C: Strict Rejection (Enforce \$25,000 Autonomous Floor)")
    st.caption("Reject all overages and strictly enforce the baseline $25,000 limit.")
    if st.button("🚫 Strict Rejection (Enforce $25,000 Floor Limit)", width="stretch", key="btn_modal_reject"):
        st.session_state["autogen_hitl_pending"] = False
        st.session_state["autogen_hitl_decision"] = "REJECTED"
        st.session_state["autogen_messages"] = scenario_data["rejected_messages"]
        st.session_state["autogen_metrics"] = scenario_data["rejected_metrics"]
        st.rerun()


@st.dialog("🛡️ Human-In-The-Loop (HITL) Executive Governance Gate", width="large")
def render_hitl_approval_modal(scenario_data: Dict[str, Any], target_store: str, sku: str) -> None:
    """Modular dialog coordinator for executive HITL sign-off."""
    tot_spend = scenario_data["metrics"]["total_commitment"]
    ceiling = 25000.00
    delta_over = tot_spend - ceiling
    vendor_units = scenario_data["metrics"]["vendor_units"]
    transfer_units = scenario_data["metrics"]["transfer_units"]
    total_units = scenario_data["metrics"]["allocated_units"]
    sla_hours = scenario_data["metrics"]["sla_hours"]

    _render_hitl_header_and_metrics(tot_spend, ceiling, delta_over, sla_hours)
    _render_hitl_option_a_full(scenario_data, tot_spend, vendor_units)
    _render_hitl_option_b_capped(scenario_data, tot_spend, target_store, sku, vendor_units, transfer_units, total_units)
    _render_hitl_option_c_reject(scenario_data)


def _inject_autogen_styles() -> None:
    """Injects custom CSS styles for pulsing HITL gate callouts and glow buttons."""
    st.markdown(
        """
        <style>
        .hitl-gate-callout {
            background: rgba(220, 38, 38, 0.07);
            border: 2px solid #dc2626;
            border-radius: 8px;
            padding: 16px 20px;
            margin: 14px 0 18px 0;
            animation: hitl-pulse-border 2.5s infinite;
        }
        @keyframes hitl-pulse-border {
            0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
        }
        div.st-key-btn_open_hitl_modal button {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_autogen_state() -> None:
    """Initializes Tab 2 multi-agent state keys."""
    if "autogen_scenario_key" not in st.session_state:
        st.session_state["autogen_scenario_key"] = "standard"
    if "autogen_last_scenario" not in st.session_state:
        st.session_state["autogen_last_scenario"] = "standard"
    if "autogen_target_store" not in st.session_state:
        st.session_state["autogen_target_store"] = "Store #104 (Denver Downtown)"
    if "autogen_sku" not in st.session_state:
        st.session_state["autogen_sku"] = "SKU-4092 (Vintage Stretch Denim, 32x32)"
    if "autogen_demand" not in st.session_state:
        st.session_state["autogen_demand"] = 50
    if "autogen_hitl_pending" not in st.session_state:
        st.session_state["autogen_hitl_pending"] = False
    if "autogen_hitl_decision" not in st.session_state:
        st.session_state["autogen_hitl_decision"] = None
    if "autogen_messages" not in st.session_state:
        std = SCENARIOS.get("standard", {})
        st.session_state["autogen_messages"] = std.get("messages", [])
        st.session_state["autogen_metrics"] = std.get("metrics", {})
        st.session_state["autogen_executed"] = True


def render_autogen_control_panel() -> Tuple[str, str, str, int, bool]:
    """Renders the disruption parameters panel, store/sku pickers, and launch button."""
    with st.container():
        st.markdown("#### 🎛️ Supply Chain Crisis Parameters & Disruption Injection")
        col_store, col_sku, col_scenario, col_demand = st.columns([2, 2.5, 3.2, 1.8])
        with col_store:
            target_store = st.selectbox("Target Store Location", ["Store #104 (Denver Downtown)", "Store #101 (Boston Flagship)", "Store #118 (Chicago Regional)"], index=0)
        with col_sku:
            crisis_sku = st.selectbox("Crisis Apparel SKU", ["SKU-4092 (Vintage Stretch Denim, 32x32)", "SKU-1084 (Merino Wool Quarter-Zip, Navy)", "SKU-3190 (Performance Chino, Khaki)"], index=0)
        with col_scenario:
            scenario_labels = {
                "standard": "🟢 Standard Stockout (Store #109 + Vendor) [Auto-Approved <$25k]",
                "port_strike": "🚢 West Coast Port Strike (Intra-Network Sweep) [Auto-Approved]",
                "viral_spike": "📱 Viral Demand Spike (+300% / $31.2k Requisition) [🚨 Requires HITL Approval]",
                "ice_storm": "❄️ Regional DC Ice Storm Gridlock [Auto-Approved]",
                "air_charter": "✈️ Emergency Air Charter ($38.5k Fleet Rebalance) [🚨 Requires HITL Approval]",
            }
            curr_sc = st.session_state.get("autogen_scenario_key", "standard")
            sc_idx = list(scenario_labels.keys()).index(curr_sc) if curr_sc in scenario_labels else 0
            selected_scenario_id = st.selectbox("Stress Scenario Disruption Selector", options=list(scenario_labels.keys()), format_func=lambda k: scenario_labels[k], index=sc_idx)

            if selected_scenario_id != st.session_state.get("autogen_last_scenario"):
                st.session_state["autogen_last_scenario"] = selected_scenario_id
                st.session_state["autogen_scenario_key"] = selected_scenario_id
                st.session_state["autogen_hitl_pending"] = False
                st.session_state["autogen_hitl_decision"] = None
                sc_switch = SCENARIOS.get(selected_scenario_id, {})
                st.session_state["autogen_messages"] = sc_switch.get("messages", [])
                st.session_state["autogen_metrics"] = sc_switch.get("metrics", {})
                st.session_state["autogen_demand"] = sc_switch.get("default_demand", 50)
                st.session_state["autogen_executed"] = True

        with col_demand:
            max_u = 1000 if selected_scenario_id in ["viral_spike", "air_charter"] else 100
            step_u = 50 if selected_scenario_id in ["viral_spike", "air_charter"] else 10
            demand_slider = st.slider("Promotional Demand (Units)", min_value=10, max_value=max_u, value=st.session_state.get("autogen_demand", 50), step=step_u)

        st.session_state["autogen_scenario_key"] = selected_scenario_id
        st.session_state["autogen_target_store"] = target_store
        st.session_state["autogen_sku"] = crisis_sku
        st.session_state["autogen_demand"] = demand_slider

        sc_meta = SCENARIOS.get(selected_scenario_id, {})
        if sc_meta.get("requires_hitl", False):
            st.info(f"🛡️ **HITL Governance Scenario Active**: Sourcing commitment (${sc_meta['metrics']['total_commitment']:,.2f}) requires executive approval.")

        launch = st.button("🚀 Launch Autonomous Agent Negotiation", type="primary", width="stretch")
        return target_store, crisis_sku, selected_scenario_id, demand_slider, launch


def render_autogen_simulation_runner(scenario_id: str) -> None:
    """Simulates multi-agent negotiation using st.status."""
    scenario_data = SCENARIOS[scenario_id]
    requires_hitl = scenario_data.get("requires_hitl", False)

    with st.status(f"⚡ Orchestrating Multi-Agent Negotiation: {scenario_data['title']}...", expanded=True) as status:
        st.write("🏬 **Store Operations Lead**: Firing stockout alert...")
        time.sleep(0.5)
        st.write("📊 **Inventory Merchandising Lead**: Running stock reconciliation...")
        time.sleep(0.6)
        st.write("📦 **Vendor Procurement Lead**: Negotiating rush logistics...")
        time.sleep(0.7)

        if requires_hitl:
            st.warning("⚠️ **Requisition Audit Exception**: Spend exceeds ceiling ($25,000.00). Execution HALTED.")
            status.update(label="⏸️ Autonomous Execution Paused: Executive HITL Approval Required!", state="running", expanded=True)
            st.session_state["autogen_hitl_pending"] = True
            st.session_state["autogen_hitl_decision"] = None
            st.session_state["autogen_messages"] = scenario_data["messages"][:3]
            st.session_state["autogen_metrics"] = scenario_data.get("pending_metrics", scenario_data["metrics"])
        else:
            status.update(label=f"✅ Consensus Reached: {scenario_data['governance_status']}!", state="complete", expanded=False)
            st.session_state["autogen_hitl_pending"] = False
            st.session_state["autogen_hitl_decision"] = None
            st.session_state["autogen_messages"] = scenario_data["messages"]
            st.session_state["autogen_metrics"] = scenario_data["metrics"]

    st.session_state["autogen_executed"] = True


def render_autogen_governance_gate(active_scenario: Dict[str, Any]) -> None:
    """Renders the executive governance callout or post-decision resolution status."""
    hitl_pending = st.session_state.get("autogen_hitl_pending", False)
    hitl_decision = st.session_state.get("autogen_hitl_decision")

    if hitl_pending:
        tot_spend = active_scenario["metrics"]["total_commitment"]
        ceiling = 25000.00
        overage = tot_spend - ceiling

        # Raw f-string rf"""...""" cleanly eliminates Python SyntaxWarning on unescaped dollar signs
        st.markdown(
            rf"""
            <div class="hitl-gate-callout">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-size: 1.05rem; font-weight: 800; color: #dc2626;">🚨 HITL GOVERNANCE GATE: EXECUTIVE INTERVENTION REQUIRED</span>
                    <span style="background: #dc2626; color: white; padding: 2px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;">SPEND EXCEPTION</span>
                </div>
                <div style="font-size: 0.95rem; color: var(--text-color, #1e293b); line-height: 1.5; margin-bottom: 12px;">
                    The proposed replenishment requisition for <strong>{st.session_state.get('autogen_target_store')}</strong> 
                    commits <strong>\${tot_spend:,.2f}</strong>, exceeding the autonomous 
                    spending ceiling of <strong>\${ceiling:,.2f}</strong> by <strong style="color: #dc2626;">+\${overage:,.2f}</strong>. 
                    Autonomous orchestration is paused. Click the button below to open the executive governance approval gate.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(f"🚨 ACTION REQUIRED: Open HITL Governance Approval Gate (${tot_spend:,.2f} Spend)", type="primary", width="stretch", key="btn_open_hitl_modal"):
            render_hitl_approval_modal(active_scenario, st.session_state.get("autogen_target_store", "Store #104"), st.session_state.get("autogen_sku", "SKU-4092"))

    elif hitl_decision == "APPROVED":
        col_banner, col_reset = st.columns([5, 1])
        with col_banner: st.success("🛡️ **HITL Override Authorized**: Sign-off granted for full requisition.")
        with col_reset:
            if st.button("🔄 Reset Gate", key="btn_reset_hitl_appr", width="stretch"):
                st.session_state["autogen_hitl_pending"] = False
                st.session_state["autogen_hitl_decision"] = None
                st.rerun()

    elif hitl_decision == "APPROVED_CAPPED":
        col_banner, col_reset = st.columns([5, 1])
        with col_banner: st.warning(f"⚖️ **Executive Capped Authority Enforced**: Capped at **${st.session_state.get('autogen_metrics', {}).get('total_commitment', 0.0):,.2f}**.")
        with col_reset:
            if st.button("🔄 Reset Gate", key="btn_reset_hitl_cap", width="stretch"):
                st.session_state["autogen_hitl_pending"] = False
                st.session_state["autogen_hitl_decision"] = None
                st.rerun()

    elif hitl_decision == "REJECTED":
        col_banner, col_reset = st.columns([5, 1])
        with col_banner: st.info("🚫 **Overage Rejected**: Spending held strictly to $25,000 baseline ceiling.")
        with col_reset:
            if st.button("🔄 Reset Gate", key="btn_reset_hitl_rej", width="stretch"):
                st.session_state["autogen_hitl_pending"] = False
                st.session_state["autogen_hitl_decision"] = None
                st.rerun()


def render_autogen_consensus_dashboard(active_scenario: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """Renders the executive consensus dashboard and 4 resolution KPI cards."""
    st.markdown("---")
    st.markdown(f"#### 📋 Consensus Decision Dashboard — {active_scenario['title']}")

    if st.session_state.get("autogen_hitl_pending", False):
        st.warning(f"⏸️ **Autonomous Orchestration Paused**: Sourcing commitment of **${metrics['total_commitment']:,.2f}** exceeds ceiling.")
    elif st.session_state.get("autogen_hitl_decision") == "REJECTED":
        st.info(f"⚠️ **Sourcing Plan Capped**: Sourced **{metrics['allocated_units']} units** within **{metrics['sla_hours']} hours**.")
    else:
        st.success(f"**Fulfillment Consensus Certified**: Sourced **{metrics['allocated_units']} units** within **{metrics['sla_hours']} hours**.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Allocated Units", f"{metrics['allocated_units']} units")
    with m2:
        st.metric("Sourcing Allocation Split", f"{metrics['transfer_units']}u Xfer / {metrics['vendor_units']}u Drop", delta=f"Source: {metrics['transfer_source']}")
    with m3:
        st.metric("Total Expediting Surcharges", f"${metrics['expediting_surcharges']:,.2f}", delta="Priority Freight Fee", delta_color="inverse")
    with m4:
        gov_val = "PENDING HITL" if st.session_state.get("autogen_hitl_pending", False) else ("AUTO-APPROVED" if "AUTO" in metrics["governance_badge"] else "HITL SIGN-OFF")
        st.metric("Governance Gate Status", gov_val, delta=metrics["governance_badge"], delta_color="off")


def render_autogen_chat_feed(messages: List[Dict[str, Any]], hitl_pending: bool) -> None:
    """Renders the interactive chat message stream with persona avatars and ERP traces."""
    st.markdown("---")
    st.markdown("#### 💬 Live Conversational Agent Feed & ERP Tool Execution Traces")
    for msg in messages:
        sender_id = msg.get("sender", "Admin_Executor")
        cfg = PERSONA_CONFIG.get(sender_id, PERSONA_CONFIG["Admin_Executor"])
        with st.chat_message(name=cfg["name"], avatar=cfg["avatar"]):
            st.markdown(f"<div style='font-size: 0.8rem; font-weight: 700; color: {cfg['badge_color']};'>{cfg['avatar']} {cfg['name']}</div>", unsafe_allow_html=True)
            st.markdown(sanitize_markdown_dollars(msg.get("content", "")))
            if "tool_call" in msg:
                tc = msg["tool_call"]
                with st.expander(f"⚙️ ERP Tool Execution Trace: `{tc['tool']}()`", expanded=False):
                    st.json(tc.get("args", {}))
                    st.code(tc.get("result", ""), language="text")

    if hitl_pending:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: center; gap: 8px; padding: 14px; background: rgba(217, 119, 6, 0.1); border: 1px dashed #d97706; border-radius: 8px; color: #b45309; font-weight: 600; margin-top: 14px;">
                <span>⏸️</span> Multi-Agent Orchestration Halted at Governance Gate — Awaiting Executive Sign-off...
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_tab_autogen(show_header: bool = True) -> None:
    """Renders the complete Autonomous Multi-Agent Replenishment Ops Cockpit."""
    if show_header:
        st.markdown("### 🤖 Autonomous Multi-Agent Replenishment Ops (AutoGen)")
        st.caption(
            "Demonstrates autonomous inter-agent negotiation and supply chain crisis resolution. "
            "Four AI personas debate on-shelf availability, inventory transfers, and vendor terms under acute supply shocks."
        )

    _inject_autogen_styles()
    _init_autogen_state()

    store, sku, sc_id, demand, launch = render_autogen_control_panel()

    if launch:
        render_autogen_simulation_runner(sc_id)

    active_scenario = SCENARIOS.get(st.session_state["autogen_scenario_key"], SCENARIOS.get("standard", {}))
    render_autogen_governance_gate(active_scenario)

    metrics = st.session_state.get("autogen_metrics", active_scenario.get("metrics", {}))
    render_autogen_consensus_dashboard(active_scenario, metrics)

    messages = st.session_state.get("autogen_messages", [])
    render_autogen_chat_feed(messages, st.session_state.get("autogen_hitl_pending", False))
