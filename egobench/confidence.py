"""Failure detection v0, the differentiator.

Nobody in the free chain measures whether a method knows when it is wrong
(PROJECT_CONTEXT §3). This computes cheap per-clip confidence signals and, in
scoring, correlates them against the ACTUAL measured W-MPJPE. If confidence
predicts error, that correlation alone is a sellable artifact, and it is the
number the X thread leads on.

Signals (all computable without ground truth, that is the point):
  - reprojection_residual : 3D joints reprojected vs 2D detections (px)
  - temporal_jerk         : 3rd derivative of joint / camera trajectory (smoothness)
  - occlusion_fraction    : fraction of frames with heavy hand occlusion
  - detection_dropout     : fraction of frames with no hand detected

STATUS: signal functions are real where they need no model internals; the
reprojection + occlusion signals are wired in Phase 5 (Step 14) once predictions
exist. Correlation against W-MPJPE lives in scripts/score.py.
"""

from __future__ import annotations

import numpy as np


def temporal_jerk(traj: np.ndarray) -> float:
    """Mean magnitude of the 3rd time-difference of a (T,3) trajectory.

    High jerk = physically implausible motion = a good cheap wrongness signal.
    Works on hand-joint centroids or camera positions alike.
    """
    traj = np.asarray(traj, float)
    if traj.shape[0] < 4:
        return float("nan")
    jerk = np.diff(traj, n=3, axis=0)
    return float(np.linalg.norm(jerk, axis=1).mean())


def detection_dropout(detected: np.ndarray) -> float:
    """Fraction of frames with no hand detected. (T,) bool -> [0,1]."""
    detected = np.asarray(detected, bool)
    if detected.size == 0:
        return float("nan")
    return float(1.0 - detected.mean())


def reprojection_residual(joints_2d_pred: np.ndarray, joints_2d_det: np.ndarray) -> float:
    """Mean pixel distance between reprojected 3D joints and 2D detections.

    joints_2d_pred: (T,J,2) from projecting predicted 3D joints (egobench.frames.project)
    joints_2d_det:  (T,J,2) raw 2D detections
    """
    a = np.asarray(joints_2d_pred, float)
    b = np.asarray(joints_2d_det, float)
    mask = ~(np.isnan(a).any(-1) | np.isnan(b).any(-1))
    if mask.sum() == 0:
        return float("nan")
    return float(np.linalg.norm(a[mask] - b[mask], axis=-1).mean())


def occlusion_fraction(occluded: np.ndarray) -> float:
    """Fraction of frames flagged as heavily occluded. (T,) bool -> [0,1]."""
    occluded = np.asarray(occluded, bool)
    if occluded.size == 0:
        return float("nan")
    return float(occluded.mean())


def clip_confidence(signals: dict) -> dict:
    """Bundle the per-clip signals. Combination weighting is fit in Step 14
    against measured error, do not hand-weight blindly here."""
    return {k: v for k, v in signals.items()}
