"""
tab_rim_knn.py
Backward-compatible root re-export for Tab 3 (RIM Valuation Forensics & Predictive Store Clustering).
"""
from __future__ import annotations

from ui.tab_rim_knn import (
    ARCHETYPE_COLORS,
    generate_ai_cfo_brief_text,
    load_apparel_rim_dataset,
    load_or_train_clustering_pipeline,
    project_forward_12_periods,
    render_tab_rim_knn,
)

__all__ = [
    "load_apparel_rim_dataset",
    "load_or_train_clustering_pipeline",
    "project_forward_12_periods",
    "generate_ai_cfo_brief_text",
    "render_tab_rim_knn",
    "ARCHETYPE_COLORS",
]
