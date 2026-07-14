"""HaWoR, world-space hand motion (SLAM + hand net + occlusion infill).

The reference architecture for camera+hand fusion; produces world-frame hands
directly. CC-BY-NC-ND, benchmark use only. AWS-only (DROID-SLAM is VRAM-heavy).
Gives a second world-frame estimate to cross-check MapAnything's trajectory.

STATUS: stub. Implement in Phase 3 (Step 11).
"""

from __future__ import annotations

from egobench.data.base import Clip
from egobench.models.base import Prediction

name = "hawor"
space = "world"


def run(clip: Clip) -> Prediction:
    raise NotImplementedError("stub, Phase 3, Step 11 (AWS)")
