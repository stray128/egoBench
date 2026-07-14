"""Diagnose HaMeR PA-26 bug: is it joint-order or crop/pose quality?

Collect (hamer_j3d, gt_j3d) pairs on matched hands. For each pair align HaMeR->GT
(Procrustes) and record per-joint residual. Then find the single global joint
permutation minimizing mean residual (ICP-style: align, Hungarian, re-align).

- If best-perm PA << identity PA  -> ordering bug, hardcode the perm.
- If best-perm PA ~= identity PA ~= 26 -> HaMeR pose itself is off -> crop port.
"""
from __future__ import annotations
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from egobench import frames as F
from egobench.frames import umeyama, apply_similarity
from egobench.data.egodex import discover_clips, load_clip
from egobench.metrics.mpjpe import pa_mpjpe
from scripts.score_hamer_pnp import hamer_hands

MATCH_PX = 200.0


def collect(nclips):
    clips = discover_clips("data/egodex/test")[:nclips]
    pairs = []
    for hp in clips:
        clip = load_clip(hp)
        for cf in clip.frames(with_rgb=True):
            if cf.rgb is None:
                continue
            K = cf.camera.K
            T_wc = F.se3(cf.camera.R_wc, cf.camera.t_wc)
            gt = cf.hands
            gpx = [F.project(F.world_to_camera(h.joints[:1], T_wc), K)[0] for h in gt]
            used = set()
            for j3d, k2d in hamer_hands(cf.rgb):
                wpx = k2d[0]
                cand = [(np.hypot(*(wpx - gpx[gi])), gi) for gi in range(len(gt)) if gi not in used]
                if not cand:
                    continue
                d, gi = min(cand)
                if d > MATCH_PX:
                    continue
                used.add(gi)
                # both root-relative, metric; gt.joints world-frame -> subtract wrist
                g = gt[gi].joints - gt[gi].joints[:1]
                h = j3d - j3d[:1]
                pairs.append((h, g))
    return pairs


def perm_pa(pairs, perm):
    return np.mean([pa_mpjpe(h[perm], g) for h, g in pairs]) * 1000


def find_perm(pairs, iters=4):
    perm = np.arange(21)
    for _ in range(iters):
        # accumulate cost[i,j] = mean dist(aligned hamer joint i, gt joint j)
        cost = np.zeros((21, 21))
        for h, g in pairs:
            s, R, t = umeyama(h[perm], g, with_scale=True)
            ha = apply_similarity(s, R, t, h)  # align full (unpermuted) hamer using current perm's transform
            # dist between every aligned hamer joint and every gt joint
            d = np.linalg.norm(ha[:, None, :] - g[None, :, :], axis=-1)
            cost += d
        row, col = linear_sum_assignment(cost.T)  # gt row j <- hamer col; want perm so h[perm[j]] ~ g[j]
        newperm = np.empty(21, int)
        newperm[row] = col
        if np.array_equal(newperm, perm):
            break
        perm = newperm
    return perm


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    pairs = collect(n)
    print(f"collected {len(pairs)} matched hands")
    if not pairs:
        return
    ident = np.arange(21)
    print(f"identity PA:  {perm_pa(pairs, ident):.1f} mm")
    perm = find_perm(pairs)
    print(f"best-perm PA: {perm_pa(pairs, perm):.1f} mm")
    print(f"perm = {perm.tolist()}")
    print(f"is identity: {np.array_equal(perm, ident)}")


if __name__ == "__main__":
    main()
