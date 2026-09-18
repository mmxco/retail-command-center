"""
services/data_loaders.py
Authoritative cached loaders for static styles and data fixtures.
Provides resilient file reading, Streamlit @st.cache_data caching, and zero-crash fallbacks.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import streamlit as st
    cache_decorator = st.cache_data
except (ImportError, AttributeError):
    def cache_decorator(func):
        return func

from services.formatters import sanitize_markdown_dollars

# Base directory resolution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_styles(css_path: Optional[Union[str, Path]] = None) -> str:
    """
    Reads and returns the externalized CSS stylesheet.
    Resolves relative to project root by default.
    """
    if css_path is None:
        target = _PROJECT_ROOT / "assets" / "styles.css"
    else:
        target = Path(css_path)
        if not target.is_absolute():
            target = _PROJECT_ROOT / target

    if target.exists():
        try:
            return target.read_text(encoding="utf-8")
        except Exception:
            pass
    return ""


@cache_decorator
def load_discovery_presets(filepath: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Loads corporate prospect discovery dossiers from data/discovery_presets.json.
    Cached via Streamlit with defensive zero-crash fallback.
    """
    if filepath is None:
        target = _PROJECT_ROOT / "data" / "discovery_presets.json"
    else:
        target = Path(filepath)
        if not target.is_absolute():
            target = _PROJECT_ROOT / target

    if target.exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and len(data) > 0:
                    return data
        except Exception:
            pass

    # Defensive fallback (guarantees zero demo crashes)
    return {
        "https://www.buckle.com": {
            "account_name": "The Buckle, Inc.",
            "domain": "https://www.buckle.com",
            "retail_tier": "Specialty Apparel Chain",
            "annual_revenue": "$1.26 Billion",
            "headcount": "8,200 employees",
            "pain_points": [
                {
                    "category": "POS-to-ERP Sync Latency",
                    "title": "Legacy AS400 Batch Latency",
                    "metric_badge": "6-8hr Sync Lag",
                    "technical_gap": "Store POS terminals rely on end-of-day batch flat-file polling to synchronize with legacy on-prem AS400 core.",
                    "operational_friction": "Store associates sell against phantom stock during peak weekend promotional surges, leading to 14% BOPIS order cancellations.",
                    "financial_impact": "$18.4M annual revenue leakage and elevated customer churn in denim category.",
                    "affected_executives": ["Chief Information Officer", "VP of Retail Operations"],
                },
                {
                    "category": "Promotional Markdown Addiction",
                    "title": "Margin Degradation via Reactive Markdowns",
                    "metric_badge": "38% Promo Dependency",
                    "technical_gap": "Absence of real-time elasticity modeling; pricing desk manually schedules flat 25-40% discount events across regional malls.",
                    "operational_friction": "Stores over-discount slow-moving seasonal inventory instead of triggering localized intra-chain transfers to flagship stores.",
                    "financial_impact": "420 bps gross margin erosion; balance sheet asset write-downs exceed $12M annually under RIM cost complement.",
                    "affected_executives": ["VP of Merchandising", "Chief Financial Officer"],
                },
                {
                    "category": "Omnichannel Return Friction",
                    "title": "Isolated Reverse Logistics Island",
                    "metric_badge": "22% Return Rate",
                    "technical_gap": "E-commerce digital returns portal decoupled from store POS inventory ledger; store returns take 72 hours to re-enter available-to-promise (ATP).",
                    "operational_friction": "High-value returned denim sits stranded in backrooms during peak selling windows, missing full-price turnover.",
                    "financial_impact": "$6.2M in stranded working capital and unnecessary replacement vendor purchase orders.",
                    "affected_executives": ["VP of Supply Chain", "VP of Omnichannel"],
                },
            ],
            "tech_stack": {"ecommerce_platform": "SFCC", "erp_core": "AS400", "signals": []},
            "discovery_questions": [],
            "gdocs_url": "",
        },
        "https://www.ralphlauren.com": {
            "account_name": "Ralph Lauren Corporation",
            "domain": "https://www.ralphlauren.com",
            "retail_tier": "Flagship Tier 1 Enterprise",
            "annual_revenue": "$6.63 Billion",
            "headcount": "23,300 employees",
            "pain_points": [
                {
                    "category": "Global Allocation & Replenishment Latency",
                    "title": "Cross-Border ERP Stock Disconnect",
                    "metric_badge": "12-18hr Global Lag",
                    "technical_gap": "Disparate regional SAP instances (Americas, EMEA, APAC) with fragmented middleware integration.",
                    "operational_friction": "Regional merchandising managers lack unified visibility into factory production commits and port delay queues.",
                    "financial_impact": "$45M+ in expedited air-freight surcharges and unfulfilled seasonal demand.",
                    "affected_executives": ["Chief Supply Chain Officer", "Global VP of Logistics"],
                },
                {
                    "category": "Brand Protection & Full-Price Preservation",
                    "title": "Clearance Channel Cannibalization",
                    "metric_badge": "28% Off-Price Mix",
                    "technical_gap": "Disconnected pricing engines between full-price flagships and factory outlet channels.",
                    "operational_friction": "Excess seasonal styles are liquidated into outlet stores prematurely rather than balanced across high-turn regional stores.",
                    "financial_impact": "310 bps brand margin dilution and structural gross margin compression.",
                    "affected_executives": ["Chief Commercial Officer", "VP of Retail Merchandising"],
                },
                {
                    "category": "Unified In-Store Clienteling & Inventory",
                    "title": "Fragmented VIP Endless Aisle",
                    "metric_badge": "9% Missed Conversions",
                    "technical_gap": "Store associate mobile clienteling devices lack sub-second inventory reservation APIs into central DC buffers.",
                    "operational_friction": "VIP clients request sold-out runway apparel sizes that associates cannot reliably promise for next-day white-glove delivery.",
                    "financial_impact": "$14.8M in lost high-margin luxury basket value.",
                    "affected_executives": ["Head of Global Retail Stores", "Chief Customer Officer"],
                },
            ],
            "tech_stack": {"ecommerce_platform": "SFCC", "erp_core": "SAP S/4HANA", "signals": []},
            "discovery_questions": [],
            "gdocs_url": "",
        },
        "https://www.lululemon.com": {
            "account_name": "Lululemon Athletica Inc.",
            "domain": "https://www.lululemon.com",
            "retail_tier": "Specialty Athletic Retail",
            "annual_revenue": "$9.6 Billion",
            "headcount": "38,000 employees",
            "pain_points": [
                {
                    "category": "High-Velocity Core SKU Replenishment",
                    "title": "Never-Out-of-Stock (NOOS) Allocation Lag",
                    "metric_badge": "99.2% Sell-Through",
                    "technical_gap": "Forecasting algorithms fail to catch viral TikTok colorway surges in real-time, resulting in localized 0-on-hand outages.",
                    "operational_friction": "Flagship stores in major metropolitan areas run completely out of key sizes while suburban stores hold excess days of supply.",
                    "financial_impact": "$32M in missed revenue and secondary market margin capture.",
                    "affected_executives": ["VP of Merchandise Planning", "Head of Digital Retail"],
                },
                {
                    "category": "Store-as-a-Hub Fulfillment Efficiency",
                    "title": "Associate Pick & Pack Disruption",
                    "metric_badge": "34% Store-Ship Mix",
                    "technical_gap": "Store inventory pick-routing logic lacks dynamic floor-traffic pacing, overwhelming store staff during busy retail hours.",
                    "operational_friction": "Floor associates are pulled from guest assistance to pack shipping boxes, creating frontline service bottlenecks.",
                    "financial_impact": "12% decline in guest NPS and $8.5M in incremental overtime labor cost.",
                    "affected_executives": ["VP of Store Operations", "VP of Customer Experience"],
                },
                {
                    "category": "RFID-to-POS Discrepancy Reconciliation",
                    "title": "RFID Phantom Stock Drift",
                    "metric_badge": "1.8% Shrink Drift",
                    "technical_gap": "Handheld weekly RFID scans show discrepancies with POS real-time register receipts.",
                    "operational_friction": "BOPIS orders fail when system indicates 1 unit available, but physical unit was misplaced in fitting rooms.",
                    "financial_impact": "$11.2M in annual inventory shrinkage and expedited re-shipping costs.",
                    "affected_executives": ["VP of Loss Prevention & Asset Protection", "CIO"],
                },
            ],
            "tech_stack": {"ecommerce_platform": "CommerceTools", "erp_core": "SAP S/4HANA", "signals": []},
            "discovery_questions": [],
            "gdocs_url": "",
        },
    }


@cache_decorator
def load_autogen_scenarios(filepath: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Loads multi-agent autonomous replenishment scenarios from data/autogen_scenarios.json.
    Sanitizes message contents using sanitize_markdown_dollars to prevent KaTeX corruption.
    Cached via Streamlit with defensive zero-crash fallback.
    """
    if filepath is None:
        target = _PROJECT_ROOT / "data" / "autogen_scenarios.json"
    else:
        target = Path(filepath)
        if not target.is_absolute():
            target = _PROJECT_ROOT / target

    scenarios: Dict[str, Dict[str, Any]] = {}
    if target.exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict) and len(loaded) > 0:
                    scenarios = loaded
        except Exception:
            pass

    if not scenarios:
        # Minimal fallback
        scenarios = {
            "standard": {
                "id": "standard",
                "title": "Standard Stockout",
                "icon": "🟢",
                "description": "Store #104 stockout fallback.",
                "sourcing_summary": "Intra-store rebalance.",
                "default_demand": 50,
                "governance_status": "AUTO-APPROVED (< $25,000 threshold)",
                "messages": [],
                "metrics": {
                    "demand_units": 50,
                    "allocated_units": 50,
                    "transfer_units": 18,
                    "vendor_units": 32,
                    "transfer_source": "Store #109",
                    "vendor_source": "DIST-CENTRAL",
                    "expediting_surcharges": 208.0,
                    "total_commitment": 1472.0,
                    "governance_badge": "🟢 AUTO-APPROVED",
                    "sla_hours": 24,
                },
            }
        }

    # Dynamically sanitize all scenario message branches
    for sc in scenarios.values():
        for m in sc.get("messages", []):
            if "content" in m and isinstance(m["content"], str):
                m["content"] = sanitize_markdown_dollars(m["content"])
        for m in sc.get("rejected_messages", []):
            if "content" in m and isinstance(m["content"], str):
                m["content"] = sanitize_markdown_dollars(m["content"])

    return scenarios
