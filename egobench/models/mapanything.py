"""MapAnything, feed-forward metric world reconstruction + camera trajectory.

Meta, 3DV 2026. Ships an Apache-2.0 checkpoint (the commercially-clean default
this project adopts over VGGT's apply-for-commercial checkpoint). AWS-only
(1B-class, OOMs 6GB). Emits the camera trajectory scored against Stera ARKit GT,
and metric scale for the scale-error metric.

STATUS: stub. Implement in Phase 3 (Step 11). Confirm output frame + metric-scale
convention; route the world<->camera transform through egobench.frames.
"""

from __future__ import annotations

from egobench.data.base import Clip
from egobench.models.base import Prediction

name = "mapanything"
space = "world"


def run(clip: Clip) -> Prediction:
    raise NotImplementedError("stub, Phase 3, Step 11 (AWS)")
