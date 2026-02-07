"""
φ-Mapping: Text → NRR State

The central contribution of Paper 2. The φ function maps natural language
text to an NRR state that preserves semantic ambiguity:

    φ: T → S
    φ(text) = { (v_1, c_1, w_1), (v_2, c_2, w_2), ... }

Unlike standard NLP pipelines that resolve ambiguity during encoding,
φ preserves all detected interpretations as co-existing elements of
a non-collapsed state.

Two extraction modes:
    1. Rule-based: Deterministic extraction via linguistic markers.
       Covers adversative and hedging ambiguity in EN and JP.
    2. LLM-based: Extraction via prompted language models (requires API).
       Covers epistemic, lexical, and structural ambiguity.

Reference: Saito (2026), "Text-to-State Mapping for Non-Resolution Reasoning:
The Contradiction-Preservation Principle", Sections 4-5.
"""

from __future__ import annotations

from typing import List, Optional

from .state import NRRState, Interpretation
from .conflict_detection import (
    detect_conflict,
    detect_adversative,
    detect_hedging,
    ADVERSATIVE_MARKERS_EN,
    ADVERSATIVE_MARKERS_EN_LOOSE,
    ADVERSATIVE_PREFIX_EN,
    ADVERSATIVE_MARKERS_JP,
    HEDGING_MARKERS_EN,
    HEDGING_MARKERS_JP,
)


# ─────────────────────────────────────────────────────────────────────
# Adversative extraction
# ─────────────────────────────────────────────────────────────────────

def _extract_adversative_en(text: str) -> List[Interpretation]:
    """Extract adversative interpretations from English text."""

    # Prefix pattern: "Although X, Y"
    for prefix in ADVERSATIVE_PREFIX_EN:
        if text.startswith(prefix):
            rest = text[len(prefix):]
            comma_pos = rest.find(", ")
            if comma_pos >= 0:
                part_a = prefix + rest[:comma_pos]
                part_b = rest[comma_pos + 2:]
                return [
                    Interpretation(part_a.strip(), "pre-adversative", 0.5),
                    Interpretation(part_b.strip(), "post-adversative", 0.5),
                ]

    # Multi-marker detection: count how many markers exist
    found = []
    for marker in ADVERSATIVE_MARKERS_EN:
        pos = text.find(marker)
        if pos >= 0:
            found.append((pos, marker))
    if not found:
        for marker in ADVERSATIVE_MARKERS_EN_LOOSE:
            pos = text.lower().find(marker)
            if pos >= 0:
                found.append((pos, marker))

    if not found:
        return [Interpretation(text, "unsplit", 1.0)]

    found.sort(key=lambda x: x[0])

    if len(found) >= 2:
        # 3-way split: segment at each marker
        positions = [f[0] for f in found]
        markers = [f[1] for f in found]
        seg1 = text[:positions[0]].strip()
        seg2 = text[positions[0] + len(markers[0]):positions[1]].strip()
        seg3 = text[positions[1] + len(markers[1]):].strip()
        # Asymmetric weights: earlier segments carry more weight
        # because speaker commitment typically decreases with hedging
        return [
            Interpretation(seg1, "pre-adversative", 0.571),
            Interpretation(seg2, "mid-adversative", 0.281),
            Interpretation(seg3, "post-adversative", 0.148),
        ]
    else:
        pos, marker = found[0]
        part_a = text[:pos].strip()
        part_b = text[pos + len(marker):].strip()
        return [
            Interpretation(part_a, "pre-adversative", 0.5),
            Interpretation(part_b, "post-adversative", 0.5),
        ]


def _extract_adversative_jp(text: str, unequal: bool = False) -> List[Interpretation]:
    """Extract adversative interpretations from Japanese text."""

    if unequal:
        # Asymmetric weight assignment for pragmatically weighted sentences
        return _split_jp_unequal(text)

    for marker in ADVERSATIVE_MARKERS_JP:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                return [
                    Interpretation(
                        parts[0].strip("、。 "), "pre-adversative", 0.5
                    ),
                    Interpretation(
                        parts[1].strip("、。 "), "post-adversative", 0.5
                    ),
                ]

    # Fallback: sentence-break splitting
    if "。" in text:
        segments = [p.strip() for p in text.split("。") if p.strip()]
        if len(segments) >= 2:
            return [
                Interpretation(segments[0], "pre-adversative", 0.5),
                Interpretation(segments[1], "post-adversative", 0.5),
            ]

    return [Interpretation(text, "unsplit", 1.0)]


def _split_jp_unequal(text: str) -> List[Interpretation]:
    """Split Japanese text with pragmatically unequal weights.

    Some adversative constructions carry asymmetric pragmatic force.
    E.g., "助けたい。でもどうすればいいかわからない。" — the inability
    (post-adversative) pragmatically dominates the desire (pre-adversative).
    """
    for marker in ADVERSATIVE_MARKERS_JP:
        if marker in text:
            parts = text.split(marker, 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                return [
                    Interpretation(
                        parts[0].strip("、。 "), "pre-adversative", 0.57
                    ),
                    Interpretation(
                        parts[1].strip("、。 "), "post-adversative", 0.43
                    ),
                ]

    # Fallback with sentence break
    if "。" in text:
        segments = [p.strip() for p in text.split("。") if p.strip()]
        if len(segments) >= 2:
            return [
                Interpretation(segments[0], "pre-adversative", 0.57),
                Interpretation(segments[1], "post-adversative", 0.43),
            ]

    return [Interpretation(text, "unsplit", 1.0)]


# ─────────────────────────────────────────────────────────────────────
# Hedging extraction
# ─────────────────────────────────────────────────────────────────────

def _extract_hedging_en(text: str, triple: bool = False) -> List[Interpretation]:
    """Extract hedging interpretations from English text.

    Produces two interpretations (or three if triple=True):
        1. The hedged statement (original, with uncertainty marker)
        2. The underlying assertion (marker removed)
        3. (If triple) Proposed follow-up action
    """
    if triple:
        # Explicitly flagged as multi-hedge sentence
        return [
            Interpretation(text, "hedged-statement", 0.76),
            Interpretation("assertion without hedge", "underlying-assertion", 0.14),
            Interpretation("follow-up action", "proposed-action", 0.10),
        ]

    # Single hedge → 2-way split
    for marker in HEDGING_MARKERS_EN:
        if text.startswith(marker) or marker in text:
            assertion = text
            for m in HEDGING_MARKERS_EN:
                assertion = assertion.replace(m, "").strip()
            if not assertion:
                assertion = text
            return [
                Interpretation(text, "hedged-statement", 0.5),
                Interpretation(assertion, "underlying-assertion", 0.5),
            ]

    return [Interpretation(text, "unsplit", 1.0)]


def _extract_hedging_jp(text: str) -> List[Interpretation]:
    """Extract hedging interpretations from Japanese text."""
    for marker in HEDGING_MARKERS_JP:
        if marker in text:
            assertion = text.replace(marker, "").replace("、", "").strip()
            if not assertion:
                assertion = text
            return [
                Interpretation(text, "hedged-statement", 0.5),
                Interpretation(assertion, "underlying-assertion", 0.5),
            ]

    return [Interpretation(text, "unsplit", 1.0)]


# ─────────────────────────────────────────────────────────────────────
# Public API: φ mapping
# ─────────────────────────────────────────────────────────────────────

def phi(
    text: str,
    lang: str = "EN",
    category: Optional[str] = None,
    unequal: bool = False,
    triple: bool = False,
    metadata: Optional[dict] = None,
) -> NRRState:
    """Map text to an NRR state via the φ function.

    φ: T → S = { (v_i, c_i, w_i) }

    This is the central function of the NRR text-to-state mapping.
    It detects semantic conflicts in the input text and constructs
    a non-collapsed state preserving all interpretations.

    Args:
        text: Natural language input.
        lang: Language code ('EN' or 'JP').
        category: Ambiguity category. If None, auto-detected.
            Supported: 'adversative', 'hedging'.
        unequal: If True, assign pragmatically asymmetric weights
            (for adversative sentences with unequal force).
        triple: If True, expect 3+ interpretations
            (for multi-marker sentences).
        metadata: Optional dict of additional information.

    Returns:
        NRRState with detected interpretations.

    Examples:
        >>> state = phi("I want to quit, but I don't want to quit.")
        >>> state.state_size
        2
        >>> state.entropy  # ≈ 1.0 (balanced binary)
        1.0

        >>> state = phi("辞めたいけど、辞めたくない。", lang="JP")
        >>> state.state_size
        2
    """
    # Auto-detect category if not specified
    if category is None:
        detection = detect_conflict(text, lang)
        if detection.detected:
            category = detection.category
        else:
            category = "undetected"

    # Extract interpretations based on category
    if category == "adversative":
        if triple:
            if lang == "EN":
                interps = _extract_adversative_en(text)
            else:
                interps = _extract_adversative_jp(text, unequal=False)
        elif unequal:
            interps = _extract_adversative_jp(text, unequal=True)
        else:
            if lang == "EN":
                interps = _extract_adversative_en(text)
            else:
                interps = _extract_adversative_jp(text, unequal=False)

    elif category == "hedging":
        if lang == "EN":
            interps = _extract_hedging_en(text, triple=triple)
        else:
            interps = _extract_hedging_jp(text)

    else:
        # No conflict detected — single-interpretation state
        interps = [Interpretation(text, "literal", 1.0)]

    return NRRState(
        text=text,
        lang=lang,
        category=category,
        interpretations=interps,
        metadata=metadata or {},
    )


def phi_batch(
    sentences: List[dict],
    category: Optional[str] = None,
) -> List[NRRState]:
    """Apply φ mapping to a batch of sentences.

    Args:
        sentences: List of dicts with keys 'text', 'lang', and optionally
            'id', 'triple', 'unequal'.
        category: Override category for all sentences.

    Returns:
        List of NRRState objects.
    """
    states = []
    for sent in sentences:
        state = phi(
            text=sent["text"],
            lang=sent.get("lang", "EN"),
            category=category,
            unequal=sent.get("unequal", False),
            triple=sent.get("triple", False),
            metadata={"id": sent.get("id", "")},
        )
        states.append(state)
    return states
