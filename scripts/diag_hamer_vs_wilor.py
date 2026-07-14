"""Triangulate the HaMeR PA-26 bug: HaMeR vs WiLoR vs GT, root-relative PA.

Two-pass (6GB GPU can't hold both models): run HaMeR over all frames and cache
matched joints, free GPU, then run WiLoR. Match both to the same GT hand by 2D
wrist, compute root-relative PA of each pair.

- WiLoR~10, HaMeR~26, HaMeR-vs-WiLoR large (~25) -> HaMeR predicts different
  (worse) poses on egocentric EgoDex. Integration fine; HaMeR quality is the cap.
- HaMeR-vs-WiLoR small but both-vs-GT differ -> a definition/frame mismatch.
"""
from __future__ import annotations
import sys
import numpy as np
import torch
from egobench import frames as F
from egobench.data.egodex import discover_clips, load_clip
from egobench.metrics.mpjpe import pa_mpjpe
from egobench.models import wilor
import scripts.score_hamer_pnp as SH

MATCH_PX = 200.0


def rel(j):
    return j - j[:1]


def _flip(j, ax):
    p = j.copy(); p[:, ax] *= -1
    return p


def pa_reflect(pred, gt):
    """Min PA over identity + each single-axis reflection (catches chirality bugs
    that rotation-only Procrustes cannot undo)."""
    best = pa_mpjpe(pred, gt)
    for ax in range(3):
        p = pred.copy(); p[:, ax] *= -1
        best = min(best, pa_mpjpe(p, gt))
    return best


def match(dets2d3d, gpx, ngt):
    """dets2d3d: list of (wrist2d, joints). -> dict gi -> joints."""
    out = {}
    for w2, j in dets2d3d:
        cand = [(np.hypot(*(w2 - gpx[gi])), gi) for gi in range(ngt)]
        if cand:
            d, gi = min(cand)
            if d <= MATCH_PX and gi not in out:
                out[gi] = j
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    clips = discover_clips("data/egodex/test")[:n]

    # ---- pass 1: HaMeR, cache per (clip, frame) ----
    hcache = {}
    for ci, hp in enumerate(clips):
        clip = load_clip(hp)
        for fi, cf in enumerate(clip.frames(with_rgb=True)):
            if cf.rgb is None:
                continue
            hcache[(ci, fi)] = [(k2d[0], j3d) for j3d, k2d in SH.hamer_hands(cf.rgb)]

    SH._MODEL = None
    torch.cuda.empty_cache()

    # ---- pass 2: WiLoR + compare ----
    hg, wg, hw, hgr, hwr = [], [], [], [], []
    _HG, _HW = [], []   # raw (rel HaMeR, rel GT/WiLoR) pairs for fixed-axis test
    for ci, hp in enumerate(clips):
        clip = load_clip(hp)
        pf = wilor.run(clip).meta["per_frame"]
        for fi, cf in enumerate(clip.frames(with_rgb=True)):
            if cf.rgb is None:
                continue
            K = cf.camera.K
            T_wc = F.se3(cf.camera.R_wc, cf.camera.t_wc)
            gt = cf.hands
            gpx = [F.project(F.world_to_camera(h.joints[:1], T_wc), K)[0] for h in gt]
            hh = match(hcache.get((ci, fi), []), gpx, len(gt))
            wdets = pf[fi] if fi < len(pf) else []
            ww = match([(d["kpts_2d"][0], d["joints_rel"]) for d in wdets], gpx, len(gt))
            for gi in range(len(gt)):
                g = rel(gt[gi].joints)
                if gi in hh:
                    hg.append(pa_mpjpe(rel(hh[gi]), g))
                    hgr.append(pa_reflect(rel(hh[gi]), g))
                    _HG.append((rel(hh[gi]), g))
                if gi in ww:
                    wg.append(pa_mpjpe(rel(ww[gi]), g))
                if gi in hh and gi in ww:
                    hw.append(pa_mpjpe(rel(hh[gi]), rel(ww[gi])))
                    hwr.append(pa_reflect(rel(hh[gi]), rel(ww[gi])))
                    _HW.append((rel(hh[gi]), rel(ww[gi])))
    f = lambda a: (np.median(a) * 1000 if a else -1, len(a))
    print(f"HaMeR vs GT    PA median={f(hg)[0]:.1f}mm (n={f(hg)[1]})")
    print(f"WiLoR vs GT    PA median={f(wg)[0]:.1f}mm (n={f(wg)[1]})")
    print(f"HaMeR vs WiLoR PA median={f(hw)[0]:.1f}mm (n={f(hw)[1]})")
    print(f"HaMeR vs GT    PA+reflect(per-sample min) median={f(hgr)[0]:.1f}mm")
    print(f"HaMeR vs WiLoR PA+reflect(per-sample min) median={f(hwr)[0]:.1f}mm")
    # which single fixed axis fixes it uniformly? (the real hardcodeable fix)
    for ax in range(3):
        gg = [pa_mpjpe(_flip(h, ax), g) for h, g in _HG]
        ll = [pa_mpjpe(_flip(h, ax), w) for h, w in _HW]
        print(f"  fixed flip axis {ax}: HaMeR-vs-GT={np.median(gg)*1000:.1f}mm  HaMeR-vs-WiLoR={np.median(ll)*1000:.1f}mm")


if __name__ == "__main__":
    main()
