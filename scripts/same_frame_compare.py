"""Apples-to-apples: WiLoR-NC vs commercial-safe on IDENTICAL hand-frames.

Kills the selection-bias objection. For every GT hand that BOTH models detect on
the same frame, score both. Median W-MPJPE on that shared set is the honest number.

WiLoR world joints: undo its wrong-focal depth (it assumes ~37500px; EgoDex true
focal ~736px, depth is linear in focal) then lift camera->world with GT extrinsic.
"""
from __future__ import annotations

import numpy as np

from egobench import frames as F
from egobench.data.egodex import discover_clips, load_clip
from egobench.metrics.mpjpe import pa_mpjpe, w_mpjpe
from egobench.models import commercial_safe as CS
from egobench.models import wilor as WL

MATCH_PX = 200.0


def _match(dets_px, gt_px):
    """Greedy nearest match det-index -> gt-index by 2D wrist pixel."""
    out = {}
    used = set()
    for di, wpx in enumerate(dets_px):
        best, bj = MATCH_PX, -1
        for gj, gpx in enumerate(gt_px):
            if gj in used:
                continue
            d = np.hypot(*(wpx - gpx))
            if d < best:
                best, bj = d, gj
        if bj >= 0:
            used.add(bj)
            out[di] = bj
    return out


def main():
    pairs_w, pairs_c = [], []          # W-MPJPE on the shared set
    pa_w, pa_c = [], []
    n_shared = 0
    for hp in discover_clips("data/egodex/test"):
        clip = load_clip(hp)
        cs = CS.run(clip, scale="pnp").meta["per_frame"]
        wl = WL.run(clip).meta["per_frame"]
        for cf, cdets, wdets in zip(clip.frames(with_rgb=False), cs, wl):
            if not cdets and not wdets:
                continue
            K = cf.camera.K
            T_wc = F.se3(cf.camera.R_wc, cf.camera.t_wc)
            f_e = float(K[0, 0])
            gt = cf.hands
            gt_px = [F.project(F.world_to_camera(h.joints[:1], T_wc), K)[0] for h in gt]

            cmap = _match([d["kpts_2d"][0] for d in cdets], gt_px)   # CS det->gt
            # WiLoR: correct focal, lift to world
            w_world = []
            w_px = []
            for d in wdets:
                jrel = d["joints_rel"]; jcam = d["joints_cam"]; foc = d["focal"]
                cam_t = jcam - jrel
                cam_corr = jrel + cam_t * (f_e / foc)               # depth ~ focal
                w_world.append(F.transform_points(T_wc, cam_corr))
                w_px.append(d["kpts_2d"][0])
            wmap = _match(w_px, gt_px)

            shared = set(cmap.values()) & set(wmap.values())         # gt hands both hit
            for gj in shared:
                di_c = [k for k, v in cmap.items() if v == gj][0]
                di_w = [k for k, v in wmap.items() if v == gj][0]
                g = gt[gj].joints
                pairs_c.append(w_mpjpe(cdets[di_c]["joints_world"], g))
                pairs_w.append(w_mpjpe(w_world[di_w], g))
                pa_c.append(pa_mpjpe(cdets[di_c]["joints_world"], g))
                pa_w.append(pa_mpjpe(w_world[di_w], g))
                n_shared += 1

    pc, pw = np.array(pairs_c) * 1000, np.array(pairs_w) * 1000
    ac, aw = np.array(pa_c) * 1000, np.array(pa_w) * 1000
    print(f"\n=== SAME-FRAME (n={n_shared} GT hands both models detected) ===")
    print(f"  W-MPJPE  commercial-safe  median={np.median(pc):6.1f}mm  mean={pc.mean():6.1f}mm")
    print(f"  W-MPJPE  WiLoR-NC         median={np.median(pw):6.1f}mm  mean={pw.mean():6.1f}mm")
    print(f"  PA-MPJPE commercial-safe  median={np.median(ac):6.1f}mm")
    print(f"  PA-MPJPE WiLoR-NC         median={np.median(aw):6.1f}mm")
    win = (pc < pw).mean() * 100
    print(f"\n  commercial-safe better placement on {win:.0f}% of shared hands")
    print(f"  tax (same-frame): {np.median(pc)-np.median(pw):+.1f}mm ({np.median(pc)/np.median(pw):.2f}x)")
    np.savez("outputs/same_frame_compare.npz", cs_w=pc, wl_w=pw, cs_pa=ac, wl_pa=aw)


if __name__ == "__main__":
    main()
