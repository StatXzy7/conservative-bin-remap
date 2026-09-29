"""Pure-Python interval re-binning engine (standard library only)."""

from .engine import (
    ConservationDiagnostic,
    GroupResult,
    RebinResult,
    RemapError,
    remap,
)

__all__ = [
    "ConservationDiagnostic",
    "GroupResult",
    "RebinResult",
    "RemapError",
    "remap",
]
