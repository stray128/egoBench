"""Interim analysis of the live bench jsonl. Read-only; safe while run appends.

Records: {dataset, task, method, w, pa, rel_scale}, one per hand-frame.
Aggregation: per-clip is not in the record, so we report (a) pooled frame medians
and (b) per-task medians (median over that task's frames), then the median-of-tasks
(macro) so a few huge tasks don't dominate. Headline = W-MPJPE. PA for contrast.
"""
import json, sys, collections
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "outputs/bench_egodex_local.jsonl"

rows = collections.defaultdict(list)          # method -> list of (task,w,pa,scale)
with open(PATH) as f:
    for line in f:
        try:
            r = json.loads(line)
        except Exception:
            continue                            # tolerate a half-written trailing line
        if r.get("w") is None:
            continue
        rows[r["method"]].append((r["task"], r["w"], r["pa"], r.get("rel_scale")))

def med(x): return float(np.median(x)) if len(x) else float("nan")
def iqr(x):
    if not len(x): return (float("nan"), float("nan"))
    return float(np.percentile(x, 25)), float(np.percentile(x, 75))

print(f"\n=== egobench interim  ({PATH}) ===")
for m in ("mediapipe_pnp", "hamer_pnp", "wilor_owncam", "wilor_pnp"):
    data = rows.get(m)
    if not data:
        print(f"\n[{m}] (no records yet)")
        continue
    tasks = sorted({t for t, *_ in data})
    W = np.array([w for _, w, _, _ in data], float)
    PA = np.array([pa for _, _, pa, _ in data], float)
    SC = np.array([s for _, _, _, s in data if s is not None], float)
    # per-task medians (macro)
    byt = collections.defaultdict(lambda: ([], []))
    for t, w, pa, _ in data:
        byt[t][0].append(w); byt[t][1].append(pa)
    task_W = np.array([med(v[0]) for v in byt.values()])
    task_PA = np.array([med(v[1]) for v in byt.values()])
    q1w, q3w = iqr(W)
    print(f"\n[{m}]  {len(data)} frames over {len(tasks)} tasks")
    print(f"  W-MPJPE  pooled median {med(W):7.1f} mm   IQR [{q1w:.0f},{q3w:.0f}]   macro(median-of-task-medians) {med(task_W):7.1f} mm")
    print(f"  PA-MPJPE pooled median {med(PA):7.1f} mm   macro {med(task_PA):7.1f} mm")
    if len(SC):
        print(f"  rel_scale median {med(SC):+.3f}  (|err| median {med(np.abs(SC)):.3f})")
    print(f"  frac W<50mm {np.mean(W<50):.1%}   W<100mm {np.mean(W<100):.1%}")

# per-task table for the completed method (mediapipe)
m = "mediapipe_pnp"
if rows.get(m):
    byt = collections.defaultdict(lambda: ([], []))
    for t, w, pa, _ in rows[m]:
        byt[t][0].append(w); byt[t][1].append(pa)
    tab = sorted(((t, med(v[0]), med(v[1]), len(v[0])) for t, v in byt.items()),
                 key=lambda x: -x[1])
    print(f"\n=== {m}: per-task median W  (worst 12 / best 12 of {len(tab)}) ===")
    print("  WORST:")
    for t, w, pa, n in tab[:12]:
        print(f"    {w:7.1f} mm  PA {pa:5.1f}  n={n:5d}  {t}")
    print("  BEST:")
    for t, w, pa, n in tab[-12:]:
        print(f"    {w:7.1f} mm  PA {pa:5.1f}  n={n:5d}  {t}")
