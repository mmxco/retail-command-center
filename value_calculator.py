"""
value_calculator.py
Backward-compatible root re-export for Value Realization Calculator.
"""
from __future__ import annotations

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
from ui.value_calculator import (
    create_3year_horizon_chart,
    create_value_waterfall_chart,
    render_value_calculator_tab,
)

__all__ = [
    "ValueCalculatorInputs",
    "ValueRealizationOutput",
    "calculate_value_realization",
    "create_value_waterfall_chart",
    "create_3year_horizon_chart",
    "generate_executive_business_case",
    "generate_heuristic_business_case",
    "normalize_cfo_markdown",
    "render_value_calculator_tab",
]
