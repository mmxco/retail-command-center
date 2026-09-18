"""
store_clustering.py
Backward-compatible root re-export for core.store_clustering.
"""
from __future__ import annotations

from core.store_clustering import *
from core.store_clustering import (
    ARCHETYPE_LABELS,
    FEATURE_COLS,
    StoreBenchmarkKNN,
    load_and_aggregate_store_data,
    load_pipeline,
    save_pipeline,
    train_clustering_pipeline,
)

__all__ = [
    "ARCHETYPE_LABELS",
    "FEATURE_COLS",
    "StoreBenchmarkKNN",
    "load_and_aggregate_store_data",
    "load_pipeline",
    "save_pipeline",
    "train_clustering_pipeline",
]
