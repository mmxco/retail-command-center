"""
cfo_advisor.py
Backward-compatible root re-export for services.llm_advisor and services.formatters.
"""
from __future__ import annotations

from services.formatters import (
    clean_latex_operators,
    escape_dollars,
    normalize_cfo_markdown,
    sanitize_markdown_dollars,
)
from services.llm_advisor import (
    DEFAULT_CANDIDATE_MODELS,
    GENAI_AVAILABLE,
    RIMDashboardState,
    build_cfo_reasoning_prompt,
    call_gemini_cascade,
    generate_ai_cfo_brief_text,
    generate_cfo_risk_brief,
    generate_heuristic_cfo_brief,
    get_gemini_client,
)

__all__ = [
    "RIMDashboardState",
    "build_cfo_reasoning_prompt",
    "generate_heuristic_cfo_brief",
    "generate_cfo_risk_brief",
    "normalize_cfo_markdown",
    "clean_latex_operators",
    "escape_dollars",
    "sanitize_markdown_dollars",
    "generate_ai_cfo_brief_text",
    "call_gemini_cascade",
    "get_gemini_client",
    "DEFAULT_CANDIDATE_MODELS",
    "GENAI_AVAILABLE",
]
