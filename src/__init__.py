"""
NRR φ-Mapping: Text-to-State Mapping for Non-Resolution Reasoning

Reference implementation for:
    Saito, K. (2026). Text-to-State Mapping for Non-Resolution Reasoning:
    The Contradiction-Preservation Principle. arXiv:2601.19933
"""

from .state import NRRState, Interpretation
from .phi_mapping import phi, phi_batch
from .conflict_detection import detect_conflict, detect_adversative, detect_hedging

__all__ = [
    "NRRState",
    "Interpretation",
    "phi",
    "phi_batch",
    "detect_conflict",
    "detect_adversative",
    "detect_hedging",
]
