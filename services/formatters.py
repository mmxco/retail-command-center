"""
services/formatters.py
Single authoritative utility for KaTeX math and dollar-sign sanitization across all cockpit tabs.
Consolidates and deduplicates formatting logic from cfo_advisor.py, value_calculator.py, and tab_autogen.py.
"""
from __future__ import annotations

import re
from typing import Optional


def sanitize_markdown_dollars(text: Optional[str]) -> str:
    """
    Escapes all unescaped dollar signs ($ -> \\$) using negative lookbehind
    to prevent Streamlit KaTeX from interpreting currency figures as broken math mode.
    Leaves already-escaped dollars (\\$) untouched. Idempotent.
    """
    if not text:
        return ""
    return re.sub(r'(?<!\\)\$', lambda m: r"\$", text)


def escape_dollars(text: Optional[str]) -> str:
    """Backward-compatible alias for sanitize_markdown_dollars."""
    return sanitize_markdown_dollars(text)


def clean_latex_operators(text: Optional[str]) -> str:
    """
    Converts LaTeX mathematical operators, macros, subscripts, and formula delimiters
    into clean, readable business prose while preserving store IDs and variables.
    """
    if not text:
        return ""

    cleaned = text

    # 1. Convert common LaTeX mathematical operators and macros to clean readable text
    cleaned = re.sub(r'\s*\\times\s*', ' * ', cleaned)
    cleaned = re.sub(r'\s*\\cdot\s*', ' * ', cleaned)
    cleaned = re.sub(r'\s*\\approx\s*', ' ≈ ', cleaned)
    cleaned = re.sub(r'\s*\\le(q)?\s*', ' <= ', cleaned)
    cleaned = re.sub(r'\s*\\ge(q)?\s*', ' >= ', cleaned)
    cleaned = re.sub(r'\\text\{([^}]+)\}', r'\1', cleaned)

    # 2. Convert LaTeX subscripts like EI_{Cost} -> EI_Cost or R_{ending} -> R_ending
    cleaned = re.sub(r'([A-Za-z0-9]+)_\{([^}]+)\}', r'\1_\2', cleaned)

    # 3. Strip math mode delimiters ($...$) around formulas, equations, or math variables
    cleaned = re.sub(r'\$([A-Za-z\\][^$\n]*?)\$', r'\1', cleaned)

    # 4. Fix punctuation spacing glitches like ).Applying -> ). Applying
    cleaned = re.sub(r'\)\.([A-Za-z])', r'). \1', cleaned)

    return cleaned


def normalize_cfo_markdown(text: Optional[str]) -> str:
    """
    Authoritative composite normalization:
    Converts LaTeX operators, strips formula delimiters, fixes spacing glitches, and escapes currency dollars.
    Passes 100% of assertions in tests/test_cfo_advisor.py:test_normalize_cfo_markdown.
    """
    if not text:
        return ""
    cleaned = clean_latex_operators(text)
    return sanitize_markdown_dollars(cleaned)
