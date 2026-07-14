"""MediaPipe Hands baseline, camera-space 21-joint hand landmarks.

The commercial-safe floor (Apache-2.0, no MANO). Runs on CPU / 6GB laptop GPU.

What MediaPipe emits per detected hand:
  - image landmarks  : (21,2) normalized x,y -> pixels. 2D detection.
  - world landmarks  : (21,3) metres, WRIST-ORIGIN, canonical hand space.
      root-relative + weak-perspective. It has NO absolute camera-frame
      translation and NO world placement -> W-MPJPE is not even attemptable.
      That structural gap is the point: the commercial-safe baseline cannot
      produce the headline metric at all.

Landmark order (0 wrist, then thumb/index/middle/ring/pinky x4) matches the
EgoDex MANO-21 subset ordering used by the loader.

Hand selection: MediaPipe handedness is unreliable on egocentric footage, so we
match each detection to the GT hand by nearest wrist in 2D (done in scoring),
not by MediaPipe's Left/Right label.
"""

from __future__ import annotations

import numpy as np

from egobench.data.base import Clip
from egobench.models.base import Prediction

name = "mediapipe"
space = "camera"


def run(clip: Clip, max_hands: int = 2, min_conf: float = 0.5) -> Prediction:
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=max_hands,
        min_detection_confidence=min_conf, min_tracking_confidence=0.5,
    )
    per_frame: list[list[dict]] = []
    try:
        for cf in clip.frames(with_rgb=True):
            dets: list[dict] = []
            if cf.rgb is not None:
                H, W = cf.rgb.shape[:2]
                res = hands.process(cf.rgb)
                if res.multi_hand_landmarks:
                    world = res.multi_hand_world_landmarks or res.multi_hand_landmarks
                    for lm2d, lmw, handed in zip(
                        res.multi_hand_landmarks, world, res.multi_handedness
                    ):
                        kpts_2d = np.array([[p.x * W, p.y * H] for p in lm2d.landmark])
                        joints_rel = np.array([[p.x, p.y, p.z] for p in lmw.landmark])  # wrist-origin m
                        dets.append({
                            "kpts_2d": kpts_2d, "joints_rel": joints_rel,
                            "label": handed.classification[0].label,
                            "score": float(handed.classification[0].score),
                        })
            per_frame.append(dets)
    finally:
        hands.close()

    return Prediction(
        clip_id=clip.clip_id, model=name, space=space,
        meta={"per_frame": per_frame, "note": "root-relative only; no world placement"},
    )
