"""
Conflict Detection for NRR φ-Mapping

Detects linguistic markers of semantic ambiguity in text.
Supports two ambiguity categories:

1. Adversative conflict: Opposing claims joined by contrastive markers.
   EN: but, however, yet, although
   JP: けど (kedo), しかし (shikashi), だが (daga), でも (demo)

2. Hedging conflict: Uncertain assertions with epistemic markers.
   EN: maybe, perhaps, might, probably
   JP: かもしれない (kamoshirenai), たぶん (tabun), だろう (darou)

Reference: Saito (2026), "Text-to-State Mapping for Non-Resolution Reasoning:
The Contradiction-Preservation Principle", Section 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────
# Marker definitions
# ─────────────────────────────────────────────────────────────────────

ADVERSATIVE_MARKERS_EN = [
    # Comma-separated markers (high confidence)
    ", but ",
    ", however ",
    ", yet ",
    ". However ",
    ". But ",
]

ADVERSATIVE_MARKERS_EN_LOOSE = [
    # Lowercase fallback markers
    " but ",
    " however ",
    " yet ",
]

ADVERSATIVE_PREFIX_EN = [
    "Although ",
]

ADVERSATIVE_MARKERS_JP = [
    "けど",      # kedo — colloquial contrastive
    "しかし",    # shikashi — formal contrastive
    "だが",      # daga — formal contrastive
    "でも",      # demo — colloquial contrastive
    "が、",      # ga, — clause-final contrastive
]

HEDGING_MARKERS_EN = [
    "Maybe ",
    "Perhaps ",
    "I might ",
    "It's probably ",
    "might ",
    "probably ",
]

HEDGING_MARKERS_JP = [
    "かもしれない",  # kamoshirenai — epistemic possibility
    "たぶん",        # tabun — probabilistic hedge
    "だろう",        # darou — conjecture
]


@dataclass
class DetectionResult:
    """Result of conflict/ambiguity detection.

    Attributes:
        detected: Whether ambiguity was detected.
        category: 'adversative', 'hedging', or None.
        marker: The specific marker found, or None.
        marker_position: Character index of the marker in text.
        num_segments: Expected number of interpretation segments.
    """
    detected: bool
    category: Optional[str] = None
    marker: Optional[str] = None
    marker_position: Optional[int] = None
    num_segments: int = 1


def detect_adversative(text: str, lang: str) -> DetectionResult:
    """Detect adversative conflict markers in text.

    Args:
        text: Input text.
        lang: Language code ('EN' or 'JP').

    Returns:
        DetectionResult with marker information if found.
    """
    if lang == "EN":
        # Check prefix markers (e.g., "Although ...")
        for prefix in ADVERSATIVE_PREFIX_EN:
            if text.startswith(prefix):
                # Find the comma that separates the clauses
                rest = text[len(prefix):]
                comma_pos = rest.find(", ")
                if comma_pos >= 0:
                    return DetectionResult(
                        detected=True,
                        category="adversative",
                        marker=prefix.strip(),
                        marker_position=0,
                        num_segments=2,
                    )

        # Count contrastive markers to detect multi-marker sentences
        marker_positions = []
        for marker in ADVERSATIVE_MARKERS_EN:
            pos = text.find(marker)
            if pos >= 0:
                marker_positions.append((pos, marker))

        if not marker_positions:
            # Try loose matching
            for marker in ADVERSATIVE_MARKERS_EN_LOOSE:
                pos = text.lower().find(marker)
                if pos >= 0:
                    marker_positions.append((pos, marker))

        if marker_positions:
            marker_positions.sort(key=lambda x: x[0])
            return DetectionResult(
                detected=True,
                category="adversative",
                marker=marker_positions[0][1].strip(),
                marker_position=marker_positions[0][0],
                num_segments=len(marker_positions) + 1,
            )

    elif lang == "JP":
        marker_positions = []
        for marker in ADVERSATIVE_MARKERS_JP:
            pos = text.find(marker)
            if pos >= 0:
                marker_positions.append((pos, marker))

        if marker_positions:
            marker_positions.sort(key=lambda x: x[0])
            return DetectionResult(
                detected=True,
                category="adversative",
                marker=marker_positions[0][1],
                marker_position=marker_positions[0][0],
                num_segments=len(marker_positions) + 1,
            )

        # Fallback: sentence break with contrasting content
        if "。" in text:
            parts = [p for p in text.split("。") if p.strip()]
            if len(parts) >= 2:
                return DetectionResult(
                    detected=True,
                    category="adversative",
                    marker="。",
                    marker_position=text.find("。"),
                    num_segments=len(parts),
                )

    return DetectionResult(detected=False)


def detect_hedging(text: str, lang: str) -> DetectionResult:
    """Detect hedging/epistemic uncertainty markers in text.

    Args:
        text: Input text.
        lang: Language code ('EN' or 'JP').

    Returns:
        DetectionResult with marker information if found.
    """
    if lang == "EN":
        # Count hedge markers
        found_markers = []
        for marker in HEDGING_MARKERS_EN:
            if text.startswith(marker) or marker in text:
                pos = text.find(marker)
                found_markers.append((pos, marker))

        if found_markers:
            found_markers.sort(key=lambda x: x[0])
            num_segments = 2
            # Multiple hedges → 3 segments
            if len(found_markers) >= 2:
                num_segments = 3
            return DetectionResult(
                detected=True,
                category="hedging",
                marker=found_markers[0][1].strip(),
                marker_position=found_markers[0][0],
                num_segments=num_segments,
            )

    elif lang == "JP":
        found_markers = []
        for marker in HEDGING_MARKERS_JP:
            pos = text.find(marker)
            if pos >= 0:
                found_markers.append((pos, marker))

        if found_markers:
            found_markers.sort(key=lambda x: x[0])
            return DetectionResult(
                detected=True,
                category="hedging",
                marker=found_markers[0][1],
                marker_position=found_markers[0][0],
                num_segments=2,
            )

    return DetectionResult(detected=False)


def detect_conflict(text: str, lang: str) -> DetectionResult:
    """Detect any type of semantic conflict in text.

    Checks adversative first, then hedging. Returns the first match.

    Args:
        text: Input text.
        lang: Language code ('EN' or 'JP').

    Returns:
        DetectionResult for the detected conflict type.
    """
    result = detect_adversative(text, lang)
    if result.detected:
        return result

    result = detect_hedging(text, lang)
    if result.detected:
        return result

    return DetectionResult(detected=False)
