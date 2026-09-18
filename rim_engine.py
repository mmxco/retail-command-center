"""
rim_engine.py
Backward-compatible root re-export for core.rim_engine.
"""
from __future__ import annotations

from core.rim_engine import *
from core.rim_engine import RIMEngine, compute_period_rim

__all__ = [
    "RIMEngine",
    "compute_period_rim",
]
