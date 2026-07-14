"""Common model interface.

A HandModel consumes a Clip and emits per-frame HandPose predictions. A WorldModel
additionally emits a camera trajectory. The stage runner (scripts/run_stage.py)
treats every model through this interface and writes predictions to a .npz keyed
by clip_id, so local and AWS stages merge cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from egobench.data.base import Clip


@dataclass
class Prediction:
    """What one model produced for one clip. Saved as .npz, synced across machines."""

    clip_id: str
    model: str
    space: str                                  # "camera" | "world"
    hand_joints: np.ndarray | None = None       # (T, J, 3)
    hand_conf: np.ndarray | None = None          # (T, J) if reported
    camera_t: np.ndarray | None = None           # (T, 3) camera positions, world
    camera_R: np.ndarray | None = None           # (T, 3, 3) world-from-camera
    detected: np.ndarray | None = None           # (T,) bool, hand found this frame
    meta: dict = field(default_factory=dict)


class Model(Protocol):
    name: str
    space: str

    def run(self, clip: Clip) -> Prediction:
        ...
