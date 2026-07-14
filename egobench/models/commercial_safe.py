"""Commercial-safe hand -> world pipeline (the Apache floor of the tax ladder).

Produces metric, world-frame 21-joint hands from a monocular clip using ONLY
permissively-licensed parts (MediaPipe Apache-2.0 + OpenCV + our geometry). No
MANO, no NC weights. Scored against EgoDex GT to get the commercial-safe W-MPJPE
- the number the tax is measured from.

The scale problem: MediaPipe gives metric hand *shape* (root-relative, wrist-origin)
and 2D keypoints, but no absolute placement. We need the wrist's metric depth along
its camera ray. Three ways to get it (`scale` arg):

  "depth"  E0  back-project the wrist pixel at a metric-depth-net reading.
               BASELINE / deliberately weak, metric depth nets are blind to
               near-field hands (see memory: metric-depth-blind-to-hands), off 2-3x.
               Included to show the naive Apache floor is broken.

  "pnp"    E1  solve PnP between MediaPipe's METRIC 3D hand model and its own 2D
               keypoints, with intrinsics K. Because the 3D model is metric, PnP's
               translation is metric -> wrist depth from anthropometry + geometry,
               the depth net never consulted. The real Apache-safe estimator.

  (E2 known-motion triangulation across frames, and E-fusion, land in Milestone B.)

Camera-frame joints are lifted to world with the clip's camera extrinsic (GT on
EgoDex; real ARKit pose on Stera). Output mirrors wilor.py: meta["per_frame"] holds
one dict per detected hand with world-frame joints, so scoring matches to GT by
nearest 2D wrist (MediaPipe handedness is unreliable egocentric).
"""

from __future__ import annotations

import numpy as np

from egobench import frames as F
from egobench.data.base import Clip
from egobench.models.base import Prediction

name = "commercial_safe"
space = "world"


def _pnp_camera_joints(obj_metric: np.ndarray, img_2d: np.ndarray, K: np.ndarray):
    """PnP a metric 3D hand model to its 2D keypoints -> (21,3) camera-frame joints.

    obj_metric: (21,3) root-relative metric shape (MediaPipe world landmarks, m).
    img_2d:     (21,2) pixel keypoints. K: (3,3). Returns None if PnP fails.
    """
    import cv2

    obj = np.ascontiguousarray(obj_metric, dtype=np.float64)
    img = np.ascontiguousarray(img_2d, dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        obj, img, K.astype(np.float64), None, flags=cv2.SOLVEPNP_SQPNP
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return (R @ obj.T).T + tvec.reshape(1, 3)          # camera-frame metric joints


def _depth_camera_joints(obj_metric, img_2d, K, depth_map, patch):
    """E0: place the metric hand so its wrist sits at the depth-net reading.

    Orientation still solved by PnP (shape->2D); only the wrist DEPTH is overridden
    with the depth net, so this isolates 'what if scale came from the depth net'.
    """
    import cv2
    from egobench.models.depth_anything import DepthEstimator

    d = DepthEstimator.sample(depth_map, img_2d[0], patch=patch)   # wrist pixel
    if not np.isfinite(d):
        return None
    obj = np.ascontiguousarray(obj_metric, dtype=np.float64)
    img = np.ascontiguousarray(img_2d, dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(obj, img, K.astype(np.float64), None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    # wrist camera ray at metric depth d -> wrist camera position
    u, v = float(img_2d[0][0]), float(img_2d[0][1])
    ray = np.linalg.inv(K.astype(np.float64)) @ np.array([u, v, 1.0])
    wrist_cam = ray / ray[2] * d
    rotated = (R @ obj.T).T                              # root-relative, oriented
    return rotated - rotated[0] + wrist_cam.reshape(1, 3)  # anchor wrist at depth


def run(clip: Clip, scale: str = "pnp", device: str | None = None,
        depth_variant: str = "small", depth_patch: int = 9) -> Prediction:
    import mediapipe as mp

    if scale not in ("pnp", "depth"):
        raise ValueError("scale must be 'pnp' or 'depth'")

    depth_est = None
    if scale == "depth":
        from egobench.models.depth_anything import DepthEstimator
        depth_est = DepthEstimator(variant=depth_variant, device=device)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=2,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    per_frame: list[list[dict]] = []
    try:
        for cf in clip.frames(with_rgb=True):
            dets: list[dict] = []
            if cf.rgb is not None:
                H, W = cf.rgb.shape[:2]
                K = cf.camera.K
                T_wc = F.se3(cf.camera.R_wc, cf.camera.t_wc)
                depth_map = depth_est.infer(cf.rgb) if depth_est is not None else None
                res = hands.process(cf.rgb)
                if res.multi_hand_landmarks:
                    world = res.multi_hand_world_landmarks or res.multi_hand_landmarks
                    for lm2d, lmw, handed in zip(
                        res.multi_hand_landmarks, world, res.multi_handedness
                    ):
                        kpts_2d = np.array([[p.x * W, p.y * H] for p in lm2d.landmark])
                        obj = np.array([[p.x, p.y, p.z] for p in lmw.landmark])  # metric, wrist-origin
                        if scale == "pnp":
                            cam_j = _pnp_camera_joints(obj, kpts_2d, K)
                        else:
                            cam_j = _depth_camera_joints(obj, kpts_2d, K, depth_map, depth_patch)
                        if cam_j is None:
                            continue
                        world_j = F.transform_points(T_wc, cam_j)     # camera -> world
                        dets.append({
                            "joints_world": world_j,                  # (21,3)
                            "joints_cam": cam_j,
                            "kpts_2d": kpts_2d,
                            "side": handed.classification[0].label.lower(),  # unreliable; scoring re-matches
                            "wrist_depth": float(cam_j[0, 2]),
                        })
            per_frame.append(dets)
    finally:
        hands.close()

    return Prediction(
        clip_id=clip.clip_id, model=f"{name}:{scale}", space=space,
        meta={"per_frame": per_frame, "scale": scale,
              "note": f"Apache floor; scale={scale}; world-frame via clip extrinsic"},
    )
