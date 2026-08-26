"""
osea_confidence.py
-------------------
Centralized, configurable confidence decision logic for OSEA results.

This module exists so that detector/classifier thresholds live in ONE place
instead of being scattered across app.py. Tune the constants below once you
have a proper calibration test set.

Decision states
---------------
NOT_A_BIRD        : detector did not find a bird with sufficient confidence.
UNABLE_TO_IDENTIFY: a bird was detected, but the classifier isn't confident
                    enough about the species to report one.
IDENTIFIED        : a bird was detected and the classifier is confident
                    enough to report a species.

Notes
-----
- OSEA's classifier score is a real softmax probability (not a cosine
  similarity like some CLIP-style models), but with ~11k fine-grained
  species classes, a *correct* top-1 prediction can still have a fairly
  low absolute probability (lots of visually similar species competing
  for probability mass). So we deliberately do NOT require a high
  absolute top-1 score by default -- we lean more on the top1/top2
  margin (how decisively the model prefers its best guess) and on the
  detector confidence (is this even a bird / a clear photo of one).
- These defaults are intentionally conservative placeholders per the
  migration request. Recalibrate against a real labeled test set
  (known birds, hard birds, non-birds) before treating them as final.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class IdentificationState(str, Enum):
    NOT_A_BIRD = "not_a_bird"
    UNABLE_TO_IDENTIFY = "unable_to_identify"
    IDENTIFIED = "identified"


@dataclass
class ConfidenceThresholds:
    # Below this detector confidence, we say "not a bird" outright,
    # regardless of what the classifier thinks.
    detector_min_confidence: float = 0.60

    # Minimum top-1 classifier probability to even consider reporting
    # a species. Kept low on purpose -- see module docstring.
    classifier_min_top1: float = 0.05

    # Minimum top1/top2 margin. A small margin means the model is torn
    # between two (often similar-looking) species, so we prefer
    # "unable to identify" over guessing.
    classifier_min_margin: float = 0.03


DEFAULT_THRESHOLDS = ConfidenceThresholds()


@dataclass
class ConfidenceDecision:
    state: IdentificationState
    detector_confidence: float
    top1_score: float
    top2_score: float
    margin: float
    reason: str

    @property
    def is_identified(self) -> bool:
        return self.state == IdentificationState.IDENTIFIED


def decide(
    detector_confidence: float,
    detector_detected: bool,
    top1_score: float,
    top2_score: float = 0.0,
    thresholds: ConfidenceThresholds = DEFAULT_THRESHOLDS,
) -> ConfidenceDecision:
    """
    Single entry point for turning raw OSEA scores into one of the three
    states BirdLens needs to react to. Call this from app.py instead of
    comparing scores inline.
    """
    margin = max(0.0, top1_score - top2_score)

    if not detector_detected or detector_confidence < thresholds.detector_min_confidence:
        return ConfidenceDecision(
            state=IdentificationState.NOT_A_BIRD,
            detector_confidence=detector_confidence,
            top1_score=top1_score,
            top2_score=top2_score,
            margin=margin,
            reason=(
                f"Detector confidence {detector_confidence:.3f} below "
                f"threshold {thresholds.detector_min_confidence:.3f}"
            ),
        )

    if top1_score < thresholds.classifier_min_top1:
        return ConfidenceDecision(
            state=IdentificationState.UNABLE_TO_IDENTIFY,
            detector_confidence=detector_confidence,
            top1_score=top1_score,
            top2_score=top2_score,
            margin=margin,
            reason=(
                f"Top-1 classifier score {top1_score:.3f} below "
                f"threshold {thresholds.classifier_min_top1:.3f}"
            ),
        )

    if margin < thresholds.classifier_min_margin:
        return ConfidenceDecision(
            state=IdentificationState.UNABLE_TO_IDENTIFY,
            detector_confidence=detector_confidence,
            top1_score=top1_score,
            top2_score=top2_score,
            margin=margin,
            reason=(
                f"Top1/Top2 margin {margin:.3f} below "
                f"threshold {thresholds.classifier_min_margin:.3f} "
                "(model is torn between competing species)"
            ),
        )

    return ConfidenceDecision(
        state=IdentificationState.IDENTIFIED,
        detector_confidence=detector_confidence,
        top1_score=top1_score,
        top2_score=top2_score,
        margin=margin,
        reason="Confident identification",
    )