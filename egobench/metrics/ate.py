"""Absolute Trajectory Error for the recovered camera path.

Scored against Stera-10M's ARKit + LiDAR trajectory (real sensor GT). Standard
practice (TUM RGB-D): align the estimated trajectory to GT with a similarity
transform, then RMSE of translation residuals. Report BOTH:
  - Sim(3)-aligned ATE  -> shape/drift error with the monocular scale gauge freed.
  - SE(3)-aligned ATE    -> includes the metric-scale error (the number that bites).
The gap between them is the scale contribution, same spirit as W vs PA-MPJPE.
"""

from __future__ import annotations

import numpy as np

from egobench.frames import umeyama, apply_similarity


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(axis=1).mean()))


def ate(pred_xyz: np.ndarray, gt_xyz: np.ndarray, with_scale: bool = False) -> float:
    """ATE (metres). pred_xyz / gt_xyz are (N,3) camera positions, time-aligned.

    with_scale=False -> SE(3) alignment (keeps metric scale error in the number).
    with_scale=True  -> Sim(3) alignment (factors scale out; drift/shape only).
    """
    pred_xyz = np.asarray(pred_xyz, float)
    gt_xyz = np.asarray(gt_xyz, float)
    assert pred_xyz.shape == gt_xyz.shape and pred_xyz.shape[1] == 3
    s, R, t = umeyama(pred_xyz, gt_xyz, with_scale=with_scale)
    aligned = apply_similarity(s, R, t, pred_xyz)
    return _rmse(aligned, gt_xyz)


def ate_breakdown(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> dict:
    se3_ate = ate(pred_xyz, gt_xyz, with_scale=False)
    sim3_ate = ate(pred_xyz, gt_xyz, with_scale=True)
    return {
        "ate_se3_m": se3_ate,       # includes scale error
        "ate_sim3_m": sim3_ate,     # drift/shape only
        "scale_contribution_m": se3_ate - sim3_ate,
    }
