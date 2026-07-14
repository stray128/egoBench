"""WiLoR, multi-hand MANO regressor, camera-space, real-time.

Accuracy reference for camera-space hand pose (CC-BY-NC-ND weights, benchmark
use only, never a commercial ship). Runs on the 6GB laptop GPU (~60ms/frame,
fp16) via the WiLoR-mini inference package.

Output per detected hand (confirmed against real inference):
  pred_keypoints_3d  (21,3)  root-relative 3D joints, camera frame, metres
  pred_cam_t_full    (3,)    full-frame camera translation (metric placement)
  scaled_focal_length ()     focal WiLoR assumed for the depth estimate
  pred_keypoints_2d  (21,2)  2D keypoints
  is_right           scalar  1.0 right / 0.0 left

Absolute camera-frame joints = pred_keypoints_3d + pred_cam_t_full. Lifting those
to world with a GT camera extrinsic yields a world-frame prediction and thus a
real W-MPJPE. WiLoR's depth uses its own assumed focal, not the dataset's, that
mismatch is a genuine monocular scale error, not a bug: it is exactly what
W-MPJPE is meant to expose.

NOTE: Stera-10M's own hand annotations ARE WiLoR output, comparing WiLoR to
Stera "GT" would be circular. WiLoR is scored against EgoDex GT only.

Joint order matches MANO-21 (wrist, then thumb/index/middle/ring/pinky x4),
aligning with the EgoDex loader subset.
"""

from __future__ import annotations

import numpy as np

from egobench.data.base import Clip
from egobench.models.base import Prediction

name = "wilor"
space = "camera"

_PIPE = None


def _pipeline(device: str, dtype_str: str = "float16"):
    global _PIPE
    if _PIPE is None:
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline,
        )

        dt = torch.float16 if (dtype_str == "float16" and device == "cuda") else torch.float32
        _PIPE = WiLorHandPose3dEstimationPipeline(device=torch.device(device), dtype=dt)
    return _PIPE


def run(clip: Clip, device: str = "cuda") -> Prediction:
    pipe = _pipeline(device)
    per_frame: list[list[dict]] = []
    for cf in clip.frames(with_rgb=True):
        dets: list[dict] = []
        if cf.rgb is not None:
            for h in pipe.predict(cf.rgb):
                wp = h["wilor_preds"]
                j_rel = np.asarray(wp["pred_keypoints_3d"][0], float)          # (21,3)
                cam_t = np.asarray(wp["pred_cam_t_full"][0], float)            # (3,)
                dets.append({
                    "joints_cam": j_rel + cam_t,                              # (21,3) absolute cam frame
                    "joints_rel": j_rel,                                       # root-relative
                    "kpts_2d": np.asarray(wp["pred_keypoints_2d"][0], float),  # (21,2)
                    "side": "right" if float(h["is_right"]) >= 0.5 else "left",
                    "focal": float(np.asarray(wp["scaled_focal_length"])),
                })
        per_frame.append(dets)
    return Prediction(
        clip_id=clip.clip_id, model=name, space=space,
        meta={"per_frame": per_frame, "note": "abs cam-frame joints via pred_cam_t_full"},
    )
