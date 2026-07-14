"""Score the commercial-safe pipeline on EgoDex -> the tax ladder.

Runs commercial_safe (PnP + depth-baseline) over every EgoDex test clip, matches
each detection to a GT hand by nearest 2D wrist, computes W-MPJPE / PA-MPJPE, and
plots the ladder against the WiLoR-NC reference (191mm). Detection rate is reported
because MediaPipe drops hands on egocentric footage, a real condition, not hidden.

Usage: python scripts/score_commercial_safe.py [scale ...]   # default: pnp depth
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from egobench import frames as F
from egobench.data.egodex import discover_clips, load_clip
from egobench.metrics.mpjpe import pa_mpjpe, w_mpjpe
from egobench.models import commercial_safe as CS

WILOR_REF_MM = 190.8   # median W-MPJPE, WiLoR-NC on same clips (outputs/wilor_egodex_scores.npz)


def _match_and_score(clip, pred):
    """Yield (w, pa) per matched hand-frame: match dets to GT hands by 2D wrist."""
    pf = pred.meta["per_frame"]
    n_det = n_gt = 0
    for cf, dets in zip(clip.frames(with_rgb=False), pf):
        K = cf.camera.K
        T_wc = F.se3(cf.camera.R_wc, cf.camera.t_wc)
        gt_hands = [h for h in cf.hands]
        n_gt += len(gt_hands)
        n_det += len(dets)
        # GT wrist pixels for matching
        gt_px = [F.project(F.world_to_camera(h.joints[:1], T_wc), K)[0] for h in gt_hands]
        used = set()
        for d in dets:
            wpx = d["kpts_2d"][0]
            best, bj = 1e9, -1
            for j, gpx in enumerate(gt_px):
                if j in used:
                    continue
                dist = np.hypot(*(wpx - gpx))
                if dist < best:
                    best, bj = dist, j
            if bj < 0 or best > 200:      # no plausible GT match (px)
                continue
            used.add(bj)
            yield (w_mpjpe(d["joints_world"], gt_hands[bj].joints),
                   pa_mpjpe(d["joints_world"], gt_hands[bj].joints),
                   best)
    _match_and_score.rate = n_det / max(n_gt, 1)


def score_all(scale: str):
    clips = discover_clips("data/egodex/test")
    per_clip, ws, pas = {}, [], []
    total_det = total_gt = 0
    for hp in clips:
        clip = load_clip(hp)
        pred = CS.run(clip, scale=scale)
        cw, cpa = [], []
        # recompute det/gt for rate
        pf = pred.meta["per_frame"]
        for cf, dets in zip(clip.frames(with_rgb=False), pf):
            total_gt += len(cf.hands); total_det += len(dets)
        for w, pa, _ in _match_and_score(clip, pred):
            cw.append(w); cpa.append(pa)
        if cw:
            per_clip[clip.clip_id] = (np.median(cw) * 1000, np.median(cpa) * 1000, len(cw))
            ws += cw; pas += cpa
    ws = np.array(ws) * 1000; pas = np.array(pas) * 1000
    return {
        "scale": scale, "per_clip": per_clip,
        "w_median": float(np.median(ws)) if ws.size else float("nan"),
        "w_mean": float(np.mean(ws)) if ws.size else float("nan"),
        "pa_median": float(np.median(pas)) if pas.size else float("nan"),
        "n": int(ws.size), "det_rate": total_det / max(total_gt, 1),
    }


def main():
    scales = sys.argv[1:] or ["pnp", "depth"]
    results = {}
    for s in scales:
        r = score_all(s)
        results[s] = r
        print(f"\n=== commercial_safe scale={s} ===")
        print(f"  W-MPJPE  median={r['w_median']:.1f}mm  mean={r['w_mean']:.1f}mm  (n={r['n']} matched hands)")
        print(f"  PA-MPJPE median={r['pa_median']:.1f}mm    detection-rate={r['det_rate']*100:.0f}%")

    print(f"\n=== TAX LADDER (median W-MPJPE) ===")
    print(f"  WiLoR-NC (reference, MANO)     {WILOR_REF_MM:6.1f} mm")
    for s in scales:
        r = results[s]
        tax = r["w_median"] - WILOR_REF_MM
        print(f"  commercial-safe [{s:5s}]        {r['w_median']:6.1f} mm   tax {tax:+.1f} mm ({r['w_median']/WILOR_REF_MM:.2f}x)")

    _plot(results, scales)
    np.savez("outputs/commercial_safe_scores.npz",
             **{s: results[s] for s in scales}, wilor_ref=WILOR_REF_MM)
    print("\nsaved outputs/commercial_safe_scores.npz + outputs/tax_ladder.png")


def _plot(results, scales):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["WiLoR-NC\n(MANO, can't ship)"] + [f"Commercial-safe\n[{s}]" for s in scales]
    vals = [WILOR_REF_MM] + [results[s]["w_median"] for s in scales]
    colors = ["#c0392b"] + ["#2980b9", "#7f8c8d"][: len(scales)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v:.0f}mm", ha="center", fontweight="bold")
    ax.set_ylabel("W-MPJPE (mm) , world-frame error, lower better")
    ax.set_title("The Commercial-Safety Tax: accuracy cost of a license-clean hand→world pipeline")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path("outputs").mkdir(exist_ok=True)
    fig.savefig("outputs/tax_ladder.png", dpi=130)


if __name__ == "__main__":
    main()
