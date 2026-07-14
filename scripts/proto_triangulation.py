"""Prototype the motion-parallax cue: triangulate the wrist from known camera motion.

Physics: a static world point = intersection of camera rays from several known poses.
We already have camera poses, so this is 'structure from known motion', zero learned
parts, metric scale straight from geometry. Catch: the hand moves, so we only trust
windows where the wrist is quasi-stationary (grasp holds).

Two passes per window:
  (a) triangulate from GT 2D wrist  -> sanity: should recover GT depth near-exactly
                                        (proves math + that camera baseline suffices)
  (b) triangulate from MediaPipe 2D -> real accuracy, compared to PnP's ~56mm and GT.
"""
from __future__ import annotations

import numpy as np

from egobench import frames as F
from egobench.data.egodex import discover_clips, load_clip
from egobench.models import commercial_safe as CS


def triangulate(centers, dirs):
    """Least-squares point closest to a bundle of world-space rays (C_i, d_i unit)."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for C, d in zip(centers, dirs):
        P = np.eye(3) - np.outer(d, d)      # projector onto plane perp to ray
        A += P; b += P @ C
    return np.linalg.solve(A, b)


def ray_world(px, K, R_wc, t_wc):
    d_cam = np.linalg.inv(K) @ np.array([px[0], px[1], 1.0])
    d_world = R_wc @ d_cam
    return t_wc, d_world / np.linalg.norm(d_world)


def main():
    import os
    W = int(os.environ.get("TRI_W", 8))
    MOVE_MAX = float(os.environ.get("TRI_MOVE", 0.03))
    BASE_MIN = float(os.environ.get("TRI_BASE", 0.04))
    DIAG = os.environ.get("TRI_DIAG") == "1"

    best = None
    diag = []   # (span, base) over all full-coverage windows
    for hp in discover_clips("data/egodex/test"):
        clip = load_clip(hp)
        cs = CS.run(clip, scale="pnp").meta["per_frame"]
        frames = list(clip.frames(with_rgb=False))
        # follow RIGHT hand: per frame collect GT wrist world, GT 2D, MP 2D, PnP world wrist
        rec = []
        for cf, cdets in zip(frames, cs):
            K = cf.camera.K; R = cf.camera.R_wc; t = cf.camera.t_wc
            T = F.se3(R, t)
            gtr = next((h for h in cf.hands if h.side == "right"), None)
            if gtr is None:
                rec.append(None); continue
            gpx = F.project(F.world_to_camera(gtr.joints[:1], T), K)[0]
            # match MP det to this GT hand
            mp = None
            best_d = 120
            for d in cdets:
                dd = np.hypot(*(d["kpts_2d"][0] - gpx))
                if dd < best_d:
                    best_d, mp = dd, d
            rec.append({"K": K, "R": R, "t": t, "gt_w": gtr.joints[0], "gt_px": gpx,
                        "mp_px": None if mp is None else mp["kpts_2d"][0],
                        "pnp_w": None if mp is None else mp["joints_world"][0]})
        # slide window: need full MP coverage, stationary wrist, enough baseline
        for i in range(len(rec) - W):
            win = rec[i:i + W]
            if any(r is None or r["mp_px"] is None for r in win):
                continue
            gw = np.array([r["gt_w"] for r in win])
            span = np.linalg.norm(gw.max(0) - gw.min(0))
            cams = np.array([r["t"] for r in win])
            base = np.linalg.norm(cams.max(0) - cams.min(0))
            diag.append((span, base))
            if span < MOVE_MAX and base > BASE_MIN:
                if best is None or base > best[4]:      # keep MAX-baseline stationary window
                    best = (clip.clip_id, i, win, span, base)

    if DIAG and diag:
        d = np.array(diag)
        print(f"full-coverage windows: {len(d)}")
        print(f"  wrist span (cm):    p25={np.percentile(d[:,0],25)*100:.1f} p50={np.percentile(d[:,0],50)*100:.1f} p75={np.percentile(d[:,0],75)*100:.1f}")
        print(f"  cam baseline (cm):  p25={np.percentile(d[:,1],25)*100:.1f} p50={np.percentile(d[:,1],50)*100:.1f} p75={np.percentile(d[:,1],75)*100:.1f} max={d[:,1].max()*100:.1f}")
        print(f"  windows with span<3cm: {(d[:,0]<0.03).sum()} | baseline>4cm: {(d[:,1]>0.04).sum()} | both: {((d[:,0]<0.03)&(d[:,1]>0.04)).sum()}")

    if not best:
        print("no stationary window found"); return
    cid, i, win, span, base = best
    gt_w = np.mean([r["gt_w"] for r in win], axis=0)
    tri_gt = triangulate(*zip(*[ray_world(r["gt_px"], r["K"], r["R"], r["t"]) for r in win]))
    tri_mp = triangulate(*zip(*[ray_world(r["mp_px"], r["K"], r["R"], r["t"]) for r in win]))
    pnp_w = np.mean([r["pnp_w"] for r in win], axis=0)

    def err(p): return np.linalg.norm(p - gt_w) * 1000
    print(f"clip {cid}  frames[{i}:{i+W}]  wrist span={span*100:.1f}cm  cam baseline={base*100:.1f}cm\n")
    print(f"  GT wrist (world)          {np.round(gt_w,3)}")
    print(f"  triangulated from GT 2D   err = {err(tri_gt):5.1f} mm   (sanity: math + baseline)")
    print(f"  triangulated from MP 2D   err = {err(tri_mp):5.1f} mm   (real parallax cue)")
    print(f"  PnP wrist (this window)   err = {err(pnp_w):5.1f} mm   (current estimator)")


if __name__ == "__main__":
    main()
