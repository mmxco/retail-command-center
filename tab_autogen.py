"""
tab_autogen.py
Backward-compatible root re-export for Tab 2 (Autonomous Multi-Agent Replenishment Ops).
"""
from __future__ import annotations

from ui.tab_autogen import (
    CONSENSUS_TAG,
    PERSONA_CONFIG,
    SCENARIOS,
    generate_capped_resolution,
    load_autogen_scenarios,
    render_hitl_approval_modal,
    render_tab_autogen,
)
from services.formatters import escape_dollars, sanitize_markdown_dollars

__all__ = [
    "SCENARIOS",
    "CONSENSUS_TAG",
    "PERSONA_CONFIG",
    "load_autogen_scenarios",
    "generate_capped_resolution",
    "render_hitl_approval_modal",
    "render_tab_autogen",
    "escape_dollars",
    "sanitize_markdown_dollars",
]
