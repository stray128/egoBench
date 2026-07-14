"""Thesis figure from bench jsonl.
 Panel A: W-MPJPE tax ladder (bars, colored by shippable).
 Panel B: placement (W) vs shape (PA) per method, with the PnP-fix arrow.
"""
import json, sys, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATH = sys.argv[1] if len(sys.argv) > 1 else "outputs/bench_egodex_local.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "outputs/thesis_plot.png"

rows = collections.defaultdict(lambda: ([], []))
with open(PATH) as f:
    for line in f:
        try: r = json.loads(line)
        except Exception: continue
        if r.get("w") is None: continue
        rows[r["method"]][0].append(r["w"]); rows[r["method"]][1].append(r["pa"])

def med(x): return float(np.median(x))
M = {m: (med(w), med(pa), len(w)) for m, (w, pa) in rows.items()}

# label, license, shippable
META = {
    "mediapipe_pnp": ("MediaPipe + PnP", "Apache-2.0", True),
    "hamer_pnp":     ("HaMeR + PnP",     "NC (MANO)",  False),
    "wilor_pnp":     ("WiLoR + PnP",     "CC-BY-NC-ND", False),
    "wilor_owncam":  ("WiLoR (native cam)", "CC-BY-NC-ND", False),
}
GREEN, RED = "#2e7d32", "#c62828"

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.6))

# ---- Panel A: tax ladder (sorted by W) ----
order = sorted([m for m in M if m in META], key=lambda m: M[m][0])
labels = [META[m][0] for m in order]
Ws = [M[m][0] for m in order]
colors = [GREEN if META[m][2] else RED for m in order]
bars = axA.barh(range(len(order)), Ws, color=colors)
axA.set_yticks(range(len(order))); axA.set_yticklabels(labels)
axA.invert_yaxis()
axA.set_xlabel("Absolute world MPJPE  (no alignment, mm) , lower is better")
axA.set_title("Placement ladder\ngreen = commercially shippable (Apache)", fontsize=11)
for i, m in enumerate(order):
    axA.text(M[m][0] + 2, i, f"{M[m][0]:.1f} mm", va="center", fontsize=10)
axA.margins(x=0.15)

# ---- Panel B: W vs PA scatter ----
for m in META:
    if m not in M: continue
    w, pa, n = M[m]
    c = GREEN if META[m][2] else RED
    axB.scatter(pa, w, s=140, c=c, edgecolor="k", zorder=3)
    dy = -12 if m != "wilor_owncam" else 6
    axB.annotate(META[m][0], (pa, w), textcoords="offset points",
                 xytext=(8, dy), fontsize=9)
# PnP-fix arrow: wilor_owncam -> wilor_pnp
if "wilor_owncam" in M and "wilor_pnp" in M:
    o, p = M["wilor_owncam"], M["wilor_pnp"]
    axB.annotate("", xy=(p[1], p[0]), xytext=(o[1], o[0]),
                 arrowprops=dict(arrowstyle="->", color="#1565c0", lw=2, ls="--"))
    axB.text((o[1]+p[1])/2 + 0.3, (o[0]+p[0])/2,
             f"PnP fixes camera\n{o[0]:.0f} → {p[0]:.0f} mm  ({o[0]/p[0]:.1f}×)",
             color="#1565c0", fontsize=9, va="center")
axB.set_xlabel("PA-MPJPE  (Procrustes shape error, mm) , lower is better")
axB.set_ylabel("Absolute world MPJPE  (no alignment, mm)")
axB.set_title("Shape vs placement are separable\n(112 EgoDex tasks, 530 clips)", fontsize=11)
axB.grid(alpha=0.3, zorder=0)

fig.suptitle("egobench, monocular hand pose on Apple EgoDex test (112 tasks, 240k frames)",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=140, bbox_inches="tight")
print("wrote", OUT)
print("medians:", {m: (round(M[m][0],1), round(M[m][1],1)) for m in order})
