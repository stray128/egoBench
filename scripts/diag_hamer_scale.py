"""Is HaMeR's W-94 a metric-scale bias? Compare hand SIZE: HaMeR vs MediaPipe vs GT.

size = sum of the 20 MANO bone lengths (scale-only, pose-invariant proxy).
If MediaPipe/GT ~= 1.0 and HaMeR/GT is biased, rescaling HaMeR shape to MediaPipe
metric size before PnP should fix W (win-both, still MIT/Apache).
"""
from __future__ import annotations
import sys
import numpy as np
from egobench import frames as F
from egobench.data.egodex import discover_clips, load_clip
import scripts.score_hamer_pnp as SH

MATCH_PX = 200.0
# MANO/OpenPose-21 kinematic edges (wrist=0, 4 joints per finger)
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def bone_sum(j):
    return sum(np.linalg.norm(j[a] - j[b]) for a, b in EDGES)


def mp_world(rgb):
    """MediaPipe metric world landmarks per hand: (wrist2d, world_joints_m)."""
    H, W = rgb.shape[:2]
    res = SH._mp().process(rgb)
    out = []
    if res.multi_hand_landmarks and res.multi_hand_world_landmarks:
        for lm, wl in zip(res.multi_hand_landmarks, res.multi_hand_world_landmarks):
            w2 = np.array([lm.landmark[0].x * W, lm.landmark[0].y * H])
            wj = np.array([[p.x, p.y, p.z] for p in wl.landmark])
            out.append((w2, wj))
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    clips = discover_clips("data/egodex/test")[:n]
    hr, mr = [], []   # HaMeR/GT and MediaPipe/GT size ratios
    for hp in clips:
        clip = load_clip(hp)
        for cf in clip.frames(with_rgb=True):
            if cf.rgb is None:
                continue
            T_wc = F.se3(cf.camera.R_wc, cf.camera.t_wc)
            gt = cf.hands
            gpx = [F.project(F.world_to_camera(h.joints[:1], T_wc), cf.camera.K)[0] for h in gt]
            gsize = [bone_sum(h.joints) for h in gt]
            for j3d, k2d in SH.hamer_hands(cf.rgb):
                cand = [(np.hypot(*(k2d[0] - gpx[gi])), gi) for gi in range(len(gt))]
                if cand:
                    d, gi = min(cand)
                    if d <= MATCH_PX and gsize[gi] > 0:
                        hr.append(bone_sum(j3d) / gsize[gi])
            for w2, wj in mp_world(cf.rgb):
                cand = [(np.hypot(*(w2 - gpx[gi])), gi) for gi in range(len(gt))]
                if cand:
                    d, gi = min(cand)
                    if d <= MATCH_PX and gsize[gi] > 0:
                        mr.append(bone_sum(wj) / gsize[gi])
    print(f"HaMeR/GT   size ratio: median={np.median(hr):.3f}  (n={len(hr)})  [1.0=perfect]")
    print(f"MediaPipe/GT size ratio: median={np.median(mr):.3f}  (n={len(mr)})")


if __name__ == "__main__":
    main()
