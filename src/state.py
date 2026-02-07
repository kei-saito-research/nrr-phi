"""
NRR State Representation

Defines the core data structure for Non-Resolution Reasoning states:
    S = { (v_i, c_i, w_i) }

where:
    v_i : semantic interpretation (meaning vector)
    c_i : contextual identifier (interpretation context)
    w_i : activation weight (non-normalized)

Key property: weights are NOT constrained to sum to 1.
Multiple interpretations can maintain high activation simultaneously,
unlike softmax-normalized probability distributions.

Reference: Saito (2026), "Text-to-State Mapping for Non-Resolution Reasoning:
The Contradiction-Preservation Principle", Section 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Interpretation:
    """A single interpretation within an NRR state.

    Attributes:
        meaning: Semantic content of this interpretation (v_i).
        context: Contextual label identifying the interpretation's role (c_i).
        weight: Activation weight, non-normalized (w_i).
    """
    meaning: str
    context: str
    weight: float

    def to_dict(self) -> dict:
        return {
            "meaning": self.meaning,
            "context": self.context,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Interpretation:
        return cls(
            meaning=d["meaning"],
            context=d["context"],
            weight=d["weight"],
        )


@dataclass
class NRRState:
    """An NRR state: a set of co-existing interpretations.

    The state preserves semantic ambiguity by maintaining multiple
    interpretations with independent weights. No normalization is
    applied — interpretations do not compete.

    Attributes:
        text: Original input text.
        lang: Language code ('EN' or 'JP').
        category: Ambiguity category (e.g., 'adversative', 'hedging').
        interpretations: List of Interpretation objects.
        metadata: Optional additional information.
    """
    text: str
    lang: str
    category: str
    interpretations: List[Interpretation] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def state_size(self) -> int:
        """Number of co-existing interpretations |S|."""
        return len(self.interpretations)

    @property
    def entropy(self) -> float:
        """Shannon entropy H(S) over interpretation weights.

        H(S) = -sum_i p_i * log2(p_i)

        where p_i = w_i / sum(w_j) are normalized weights used
        only for entropy calculation (internal weights remain
        unnormalized).
        """
        weights = [interp.weight for interp in self.interpretations]
        total = sum(weights)
        if total == 0 or len(weights) == 0:
            return 0.0

        entropy = 0.0
        for w in weights:
            if w > 0:
                p = w / total
                entropy -= p * math.log2(p)
        return entropy

    @property
    def max_entropy(self) -> float:
        """Maximum possible entropy for |S| interpretations.

        H_max = log2(|S|), achieved when all weights are equal.
        """
        n = self.state_size
        if n <= 1:
            return 0.0
        return math.log2(n)

    @property
    def entropy_ratio(self) -> float:
        """Ratio H(S) / H_max. Measures how evenly distributed
        the interpretations are. 1.0 = perfectly balanced."""
        h_max = self.max_entropy
        if h_max == 0:
            return 0.0
        return self.entropy / h_max

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "lang": self.lang,
            "category": self.category,
            "interpretations": [i.to_dict() for i in self.interpretations],
            "state_size": self.state_size,
            "entropy": round(self.entropy, 3),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> NRRState:
        interps = [Interpretation.from_dict(i) for i in d.get("interpretations", [])]
        return cls(
            text=d["text"],
            lang=d.get("lang", "EN"),
            category=d.get("category", "unknown"),
            interpretations=interps,
            metadata=d.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return (
            f"NRRState(|S|={self.state_size}, H={self.entropy:.3f}, "
            f"lang={self.lang}, cat={self.category})"
        )
