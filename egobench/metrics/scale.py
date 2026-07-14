"""Absolute metric scale error.

The scale wall (PROJECT_CONTEXT §14): a monocular camera observes only bearing,
never absolute distance, scale is unobservable from images alone. This metric
asks how wrong the recovered scale is against metric ground truth. Measured two
ways; report whichever the clip supports, ideally both.
"""

from __future__ import annotations

import numpy as np


def scale_from_bones(joints: np.ndarray, bone_pairs: list[tuple[int, int]]) -> float:
    """Characteristic hand size = mean length over a fixed set of bone pairs (metres)."""
    joints = np.asarray(joints, float)
    lengths = [np.linalg.norm(joints[a] - joints[b]) for a, b in bone_pairs]
    return float(np.mean(lengths))


def scale_error_ratio(pred_size: float, gt_size: float) -> float:
    """Signed relative scale error: pred/gt - 1. 0 = perfect; +0.5 = 50% too big."""
    return float(pred_size / gt_size - 1.0)


def scale_error(
    pred_joints: np.ndarray,
    gt_joints: np.ndarray,
    bone_pairs: list[tuple[int, int]],
) -> dict:
    """Absolute + relative scale error from hand bone lengths.

    bone_pairs is the MANO joint-index pairs defining stable bones; the exact
    indices are locked in Phase 1 once the joint ordering is confirmed.
    """
    pred_size = scale_from_bones(pred_joints, bone_pairs)
    gt_size = scale_from_bones(gt_joints, bone_pairs)
    return {
        "pred_size_m": pred_size,
        "gt_size_m": gt_size,
        "abs_error_m": abs(pred_size - gt_size),
        "rel_error": scale_error_ratio(pred_size, gt_size),
    }
