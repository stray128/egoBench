"""Unified benchmark driver: datasets x methods -> per-(dataset,task,method) W/PA/scale.

The one entry point that produces the tax ladder. Dataset-agnostic (loops registered
loaders) and method-agnostic (loops runners). Add a loader or a runner -> it is in
the benchmark. Scores hand pose against paired GT ONLY, matched by 2D wrist
(handedness is unreliable egocentric, see hamer.py), per-task breakdown, no lone
aggregate.

Architecture: OUTER loop over runners, INNER over clips. Each model loads once, runs
all clips, then the GPU is freed before the next runner, so WiLoR and HaMeR never
co-reside (fits the 6GB laptop AND a 24GB A10G). GT is cached per clip (cheap,
rgb-free) and reused across methods.

Methods (license tier in parens):
  mediapipe_pnp   MediaPipe shape + PnP            (Apache, placement winner, W~57)
  hamer_pnp       HaMeR shape + PnP, chirality-fix (NC via MANO, shape winner, PA~11)
  wilor_owncam    WiLoR + its own weak-persp cam_t (ND, the monocular-scale-bias baseline, W~197)
  wilor_pnp       WiLoR shape + our PnP            (ND, ceiling, W~57/PA~10; benchmark-only)

Usage: python scripts/run_benchmark.py [--datasets egodex] [--methods all] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from egobench import frames as F
from egobench.metrics import w_mpjpe, pa_mpjpe
from egobench.metrics.scale import scale_error_ratio, scale_from_bones

MATCH_PX = 200.0
# MANO/OpenPose-21 kinematic edges (wrist=0, 4 joints/finger), bones for scale.
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
         (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]

# ---- dataset registry: name -> (discover_fn, load_fn, root) ----
def _datasets():
    from egobench.data import egodex
    reg = {"egodex": (egodex.discover_clips, egodex.load_clip, "data/egodex/test")}
    try:
        from egobench.data import hot3d
        reg["hot3d"] = (hot3d.discover_clips, hot3d.load_clip, "data/hot3d")
    except Exception:
        pass
    try:
        from egobench.data import assemblyhands as ah
        reg["assemblyhands"] = (ah.discover_clips, ah.load_clip, "data/assemblyhands")
    except Exception:
        pass
    return reg


# ---- runners: clip -> {method_name: per_frame}  (per_frame[i] = list of dets w/ joints_world, kpts_2d) ----
def run_mediapipe(clip):
    from egobench.models import commercial_safe
    return {"mediapipe_pnp": commercial_safe.run(clip, scale="pnp").meta["per_frame"]}


def run_hamer(clip):
    from egobench.models import hamer
    return {"hamer_pnp": hamer.run(clip).meta["per_frame"]}


def run_wilor(clip):
    """One WiLoR pass -> two method streams: own-cam_t placement, and our-PnP placement."""
    import cv2
    from egobench.models import wilor
    pf = wilor.run(clip).meta["per_frame"]
    # need K + T_wc per frame -> re-read GT camera (rgb-free)
    cams = [cf.camera for cf in clip.frames(with_rgb=False)]
    owncam, pnp = [], []
    for fi, dets in enumerate(pf):
        cam = cams[fi] if fi < len(cams) else None
        oc, pp = [], []
        if cam is not None:
            T_wc = F.se3(cam.R_wc, cam.t_wc); K = cam.K
            f_e = float(K[0, 0])
            for d in dets:
                # WiLoR's pred_cam_t_full assumes ITS OWN focal; rescale by f_dataset/f_wilor
                # to place at the true camera. The residual is the monocular scale bias (~197mm).
                f_w = float(d.get("focal") or f_e)
                cam_t = d["joints_cam"] - d["joints_rel"]              # = pred_cam_t_full (per-joint const)
                cam_corr = d["joints_rel"] + cam_t * (f_e / f_w)
                oc.append({"joints_world": F.transform_points(T_wc, cam_corr),
                           "kpts_2d": d["kpts_2d"]})
                obj = np.ascontiguousarray(d["joints_rel"], np.float64)
                img = np.ascontiguousarray(d["kpts_2d"], np.float64)
                ok, rvec, tvec = cv2.solvePnP(obj, img, K.astype(np.float64), None,
                                              flags=cv2.SOLVEPNP_SQPNP)
                if ok:
                    R, _ = cv2.Rodrigues(rvec)
                    camj = (R @ obj.T).T + tvec.reshape(1, 3)
                    pp.append({"joints_world": F.transform_points(T_wc, camj),
                               "kpts_2d": d["kpts_2d"]})
        owncam.append(oc); pnp.append(pp)
    return {"wilor_owncam": owncam, "wilor_pnp": pnp}


RUNNERS = {"mediapipe": run_mediapipe, "hamer": run_hamer, "wilor": run_wilor}


def _free_gpu():
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    # drop cached model globals so the next runner reloads fresh (no co-residence)
    for mod in ("egobench.models.hamer", "egobench.models.wilor"):
        m = sys.modules.get(mod)
        if m is not None:
            for g in ("_MODEL", "_PIPE", "_MP"):
                if hasattr(m, g):
                    setattr(m, g, None)


def gt_frames(clip):
    """Per-frame GT: list of (joints_world (21,3), side, wrist_2d), cached, rgb-free."""
    out = []
    for cf in clip.frames(with_rgb=False):
        if cf.camera is None or not cf.hands:
            out.append([]); continue
        T_wc = F.se3(cf.camera.R_wc, cf.camera.t_wc); K = cf.camera.K
        hs = []
        for h in cf.hands:
            w2 = F.project(F.world_to_camera(h.joints[:1], T_wc), K)[0]
            hs.append((h.joints, h.side, w2))
        out.append(hs)
    return out


def score_stream(per_frame, gts, dataset, task, method, recs):
    for fi, dets in enumerate(per_frame):
        gt = gts[fi] if fi < len(gts) else []
        if not gt:
            continue
        used = set()
        for det in dets:
            w2 = det["kpts_2d"][0]
            cand = [(np.hypot(*(w2 - gt[gi][2])), gi) for gi in range(len(gt)) if gi not in used]
            if not cand:
                continue
            d, gi = min(cand)
            if d > MATCH_PX:
                continue
            used.add(gi)
            gj = gt[gi][0]; pj = det["joints_world"]
            recs.append({
                "dataset": dataset, "task": task, "method": method,
                "w": w_mpjpe(pj, gj) * 1000.0,
                "pa": pa_mpjpe(pj, gj) * 1000.0,
                "rel_scale": scale_error_ratio(scale_from_bones(pj, BONES),
                                               scale_from_bones(gj, BONES)),
            })


def summarize(recs):
    from collections import defaultdict
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        by[r["method"]]["w"].append(r["w"]); by[r["method"]]["pa"].append(r["pa"])
    print("\n=== benchmark summary (median over all matched hands) ===")
    for m, d in sorted(by.items()):
        w = np.array(d["w"]); pa = np.array(d["pa"])
        print(f"  {m:16s}  W={np.median(w):6.1f}mm  PA={np.median(pa):5.1f}mm  (n={w.size})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="egodex")
    ap.add_argument("--methods", default="all")
    ap.add_argument("--limit", type=int, default=0, help="cap clips per dataset (0=all)")
    ap.add_argument("--out", default="outputs/bench_records.jsonl")
    args = ap.parse_args()

    reg = _datasets()
    ds_names = list(reg) if args.datasets == "all" else args.datasets.split(",")
    run_names = list(RUNNERS) if args.methods == "all" else args.methods.split(",")

    recs = []
    out_f = open(args.out, "w")  # incremental: write+flush per clip so a crash keeps partial results
    for ds in ds_names:
        discover, load, root = reg[ds]
        clips = discover(root)
        if args.limit:
            clips = clips[:args.limit]
        print(f"[{ds}] {len(clips)} clips", flush=True)
        gtc = {}  # clip_id -> gt_frames (compute once, reuse across methods)
        for rn in run_names:
            runner = RUNNERS[rn]
            for ci, hp in enumerate(clips):
                clip = load(hp)
                task = clip.condition.get("task") or clip.clip_id.split("/")[-2]
                if clip.clip_id not in gtc:
                    gtc[clip.clip_id] = gt_frames(clip)
                try:
                    streams = runner(clip)
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {rn} failed on {clip.clip_id}: {e}", flush=True)
                    continue
                n0 = len(recs)
                for method, pf in streams.items():
                    score_stream(pf, gtc[clip.clip_id], ds, task, method, recs)
                for r in recs[n0:]:
                    out_f.write(json.dumps(r) + "\n")
                out_f.flush()
                if (ci + 1) % 20 == 0:
                    print(f"  [{rn}] {ci+1}/{len(clips)} clips, {len(recs)} recs", flush=True)
            _free_gpu()
            print(f"  done runner={rn}", flush=True)
    out_f.close()

    print(f"\nwrote {len(recs)} hand records -> {args.out}", flush=True)
    summarize(recs)


if __name__ == "__main__":
    main()
