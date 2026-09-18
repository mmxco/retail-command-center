"""
tab_discovery.py
Backward-compatible root re-export for Tab 1 (Strategic Account Discovery).
"""
from __future__ import annotations

from ui.tab_discovery import (
    PRESET_DOSSIERS,
    PRESET_PROFILES,
    build_generic_prospect_profile,
    load_discovery_presets,
    render_tab_discovery,
)

__all__ = [
    "PRESET_PROFILES",
    "PRESET_DOSSIERS",
    "load_discovery_presets",
    "build_generic_prospect_profile",
    "render_tab_discovery",
]
