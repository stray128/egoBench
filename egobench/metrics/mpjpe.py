"""Mean Per-Joint Position Error, world-frame and Procrustes-aligned.

The headline is W-MPJPE. PA-MPJPE is reported ONLY for contrast: it Procrustes-
aligns away exactly the scale + world-placement error this project cares about,
so a method can look great in PA-MPJPE while placing the hand 20cm wrong in the
world (PROJECT_CONTEXT §14, anti-pattern #4). Showing the gap between the two IS
part of the thesis.
"""

from __future__ import annotations

import numpy as np

from egobench.frames import umeyama, apply_similarity


def _per_joint_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(J,) Euclidean distance per joint, metres."""
    pred = np.asarray(pred, float)
    gt = np.asarray(gt, float)
    assert pred.shape == gt.shape and pred.shape[-1] == 3
    return np.linalg.norm(pred - gt, axis=-1)


def w_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """World-frame MPJPE (metres). No alignment. Headline metric.

    Both pred and gt MUST already be in the same world frame, do the frame
    conversion via egobench.frames upstream, not here.
    """
    return float(_per_joint_error(pred, gt).mean())


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Procrustes-aligned MPJPE (metres). Similarity-aligns pred to gt first.
    Reported for contrast only, never as the headline."""
    s, R, t = umeyama(pred, gt, with_scale=True)
    aligned = apply_similarity(s, R, t, pred)
    return float(_per_joint_error(aligned, gt).mean())


def mpjpe_breakdown(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Both numbers + the gap, for one clip. The gap is the story."""
    w = w_mpjpe(pred, gt)
    pa = pa_mpjpe(pred, gt)
    return {"w_mpjpe": w, "pa_mpjpe": pa, "gap": w - pa}
