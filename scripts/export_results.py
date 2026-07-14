"""Export publishable analytics from bench jsonl:
   results/summary.csv        one row per method (aggregate)
   results/per_task.csv       task x method medians (the real per-task analytics)
   results/per_task_W.png     sorted per-task W-MPJPE, one line per method
"""
import json, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATH = "outputs/bench_egodex_local.jsonl"
os.makedirs("results", exist_ok=True)

rec = collections.defaultdict(lambda: collections.defaultdict(lambda: ([], [], [])))  # task->method->(W,PA,scale)
allm = collections.defaultdict(lambda: ([], [], []))
with open(PATH) as f:
    for line in f:
        try: r = json.loads(line)
        except Exception: continue
        if r.get("w") is None: continue
        m, t = r["method"], r["task"]
        rec[t][m][0].append(r["w"]); rec[t][m][1].append(r["pa"])
        if r.get("rel_scale") is not None: rec[t][m][2].append(r["rel_scale"])
        allm[m][0].append(r["w"]); allm[m][1].append(r["pa"])
        if r.get("rel_scale") is not None: allm[m][2].append(r["rel_scale"])

def med(x): return float(np.median(x)) if len(x) else float("nan")
METHODS = ["mediapipe_pnp", "hamer_pnp", "wilor_pnp", "wilor_owncam"]
LABEL = {"mediapipe_pnp":"MediaPipe+PnP","hamer_pnp":"HaMeR+PnP",
         "wilor_pnp":"WiLoR+PnP","wilor_owncam":"WiLoR (native)"}

# summary.csv
with open("results/summary.csv", "w") as f:
    f.write("method,license,shippable,n_frames,n_tasks,W_median_mm,PA_median_mm,scale_err_median,pct_W_lt_50,pct_W_lt_100\n")
    LIC = {"mediapipe_pnp":("Apache-2.0","yes"),"hamer_pnp":("MANO-NC","no"),
           "wilor_pnp":("CC-BY-NC-ND","no"),"wilor_owncam":("CC-BY-NC-ND","no")}
    for m in METHODS:
        W = np.array(allm[m][0]); PA = np.array(allm[m][1]); SC = np.array(allm[m][2])
        nt = len({t for t in rec if m in rec[t]})
        lic, ship = LIC[m]
        f.write(f"{m},{lic},{ship},{len(W)},{nt},{med(W):.1f},{med(PA):.1f},"
                f"{med(SC):.3f},{np.mean(W<50):.3f},{np.mean(W<100):.3f}\n")

# per_task.csv
tasks = sorted(rec)
with open("results/per_task.csv", "w") as f:
    hdr = ["task"] + [f"{m}_W" for m in METHODS] + [f"{m}_PA" for m in METHODS] + ["n_clips_frames"]
    f.write(",".join(hdr) + "\n")
    for t in tasks:
        row = [t.replace(",", " ")]
        for m in METHODS: row.append(f"{med(rec[t][m][0]):.1f}" if m in rec[t] else "")
        for m in METHODS: row.append(f"{med(rec[t][m][1]):.1f}" if m in rec[t] else "")
        n = max((len(rec[t][m][0]) for m in rec[t]), default=0)
        row.append(str(n))
        f.write(",".join(row) + "\n")

# per_task_W.png : sort tasks by mediapipe W, plot each method
base = "mediapipe_pnp"
order = sorted([t for t in tasks if base in rec[t]], key=lambda t: med(rec[t][base][0]))
x = np.arange(len(order))
plt.figure(figsize=(13, 5.5))
COL = {"mediapipe_pnp":"#2e7d32","hamer_pnp":"#f9a825","wilor_pnp":"#1565c0","wilor_owncam":"#c62828"}
for m in METHODS:
    y = [med(rec[t][m][0]) if m in rec[t] else np.nan for t in order]
    plt.plot(x, y, ".-", ms=4, lw=1, color=COL[m], label=LABEL[m], alpha=0.85)
plt.axhline(50, ls=":", c="gray", lw=1); plt.text(1, 52, "50 mm", color="gray", fontsize=8)
plt.xlabel("112 EgoDex tasks (sorted by MediaPipe+PnP difficulty)")
plt.ylabel("per-task median absolute world MPJPE (no alignment, mm)")
plt.title("Per-task world-placement error across 112 tasks, every method, every task")
plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
plt.savefig("results/per_task_W.png", dpi=140)
print("wrote results/summary.csv, results/per_task.csv, results/per_task_W.png")
print(open("results/summary.csv").read())
