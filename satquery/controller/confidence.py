"""Three-component confidence (plan task 1.3, extended in 3.4).

Confidence is not the model's softmax. A model can be certain and wrong, so
the score combines three independent signals:

* **model**      - the tool's own reported confidence
* **agreement**  - do the deterministic physics indices corroborate the claim?
* **input_quality** - were the inputs clean enough to support any answer?

They are combined with a *geometric* mean rather than an arithmetic one: the
geometric mean collapses toward zero if any single component collapses, which
is the behaviour we want. A confident model on a corrupt input should not
score 0.66 because two of three components were fine.

Phase 1 uses equal weights and no calibration; task 3.3/3.4 fits the weights
and applies temperature scaling.
"""

from __future__ import annotations

from satquery.contracts.input_manifest import InputManifest
from satquery.contracts.trace import (
    ConfidenceCalibrationTrace,
    ConfidenceComponentsTrace,
    ConfidenceTrace,
)

HIGH_BAND = 0.75
MEDIUM_BAND = 0.45

# Per-check penalties applied to the input-quality component.
WARN_PENALTY = 0.15
FAIL_PENALTY = 0.5

# Sentinel for "expected calibration error has not been measured yet".
# A real ECE is in [0, 1], so a negative value is unambiguous, and unlike NaN
# it survives JSON serialisation.
UNMEASURED_ECE = -1.0


def input_quality(manifest: InputManifest) -> float:
    """Quality of the inputs themselves, in [0, 1], from the check results."""
    if not manifest.checks:
        return 1.0
    score = 1.0
    for check in manifest.checks:
        if check.status == "WARN":
            score -= WARN_PENALTY
        elif check.status == "FAIL":
            score -= FAIL_PENALTY
    return max(0.0, min(1.0, score))


def physics_agreement(agreements: dict[str, float]) -> float:
    """Mean agreement across whatever physics checks were possible.

    With no applicable physics check the component is neutral (1.0) rather
    than zero - absence of corroboration is not evidence of contradiction,
    and the trace records that no check ran.
    """
    if not agreements:
        return 1.0
    values = list(agreements.values())
    return max(0.0, min(1.0, sum(values) / len(values)))


def geometric_mean(*values: float) -> float:
    """Geometric mean, zero if any component is zero."""
    product = 1.0
    for v in values:
        if v <= 0:
            return 0.0
        product *= v
    return product ** (1.0 / len(values))


def band(score: float) -> str:
    if score >= HIGH_BAND:
        return "HIGH"
    if score >= MEDIUM_BAND:
        return "MEDIUM"
    return "LOW"


def compute_confidence(
    model_confidence: float,
    manifest: InputManifest,
    agreements: dict[str, float],
) -> ConfidenceTrace:
    """Combine the three components into the reported confidence."""
    model = max(0.0, min(1.0, float(model_confidence)))
    agreement = physics_agreement(agreements)
    quality = input_quality(manifest)
    final = geometric_mean(model, agreement, quality)

    return ConfidenceTrace(
        final=round(final, 6),
        band=band(final),  # type: ignore[arg-type]
        components=ConfidenceComponentsTrace(
            model=round(model, 6),
            agreement=round(agreement, 6),
            input_quality=round(quality, 6),
        ),
        calibration=ConfidenceCalibrationTrace(
            # Honest placeholder: calibration is task 3.3 and has not been
            # fitted yet. T=1.0 means "no temperature scaling applied", and
            # ece_after=-1.0 is the documented "not measured" sentinel - NaN
            # would serialise to invalid JSON and break the SSE stream.
            method="uncalibrated",
            T=1.0,
            ece_after=UNMEASURED_ECE,
        ),
    )
