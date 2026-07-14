"""Coordinate-frame transforms, the one place bugs are allowed to live.

Every alignment, composition, and change-of-frame in egobench routes through
here so there is a single, tested definition of each convention. PROJECT_CONTEXT
§16 flags coordinate-frame alignment as where the bugs live; this module exists
to make that surface small and auditable.

Conventions:
  - SE(3) as 4x4 homogeneous matrices, world-from-camera (T_wc) unless named otherwise.
  - Rotations 3x3, right-handed.
  - Points as (N,3). Translations in metres.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------------------- #
# Build / decompose
# --------------------------------------------------------------------------- #

def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compose a 4x4 SE(3) from rotation (3x3) and translation (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    """Invert an SE(3) matrix cheaply (R^T, -R^T t)."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def quat_to_R(q_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion (x,y,z,w) -> 3x3 rotation. ARKit/most SDKs use xyzw or wxyz -
    confirm order per dataset in Phase 1 before trusting this."""
    return Rotation.from_quat(np.asarray(q_xyzw).reshape(4)).as_matrix()


# --------------------------------------------------------------------------- #
# Apply
# --------------------------------------------------------------------------- #

def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 SE(3) to (N,3) points."""
    pts = np.asarray(pts)
    homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (homo @ T.T)[:, :3]


def camera_to_world(pts_cam: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """Lift camera-frame points into world using the camera-to-world extrinsic."""
    return transform_points(T_wc, pts_cam)


def world_to_camera(pts_world: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """Project world points into camera frame."""
    return transform_points(se3_inv(T_wc), pts_world)


def project(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pinhole-project (N,3) camera-frame points to (N,2) pixels. For the
    reprojection-residual confidence signal and GT-overlay sanity checks."""
    pts_cam = np.asarray(pts_cam)
    z = np.clip(pts_cam[:, 2:3], 1e-6, None)
    uv = (K @ (pts_cam / z).T).T
    return uv[:, :2]


# --------------------------------------------------------------------------- #
# Align (for scoring, Procrustes / Sim(3) / SE(3))
# --------------------------------------------------------------------------- #

def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares similarity transform mapping src -> dst (Umeyama 1991).

    Returns (s, R, t) with dst ≈ s * R @ src + t. Used two ways:
      - PA-MPJPE: with_scale=True on hand joints (Procrustes alignment).
      - ATE: with_scale on trajectories to factor out the monocular scale gauge.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]

    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c = src - mu_s
    dst_c = dst - mu_d

    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    if with_scale:
        var_s = (src_c ** 2).sum() / n
        s = np.trace(np.diag(D) @ S) / var_s
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_similarity(s: float, R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return (s * (R @ np.asarray(pts).T).T) + t
