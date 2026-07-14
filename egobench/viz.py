"""The thesis plot, predicted vs ground-truth wrist trajectory over time,
with scale error and drift annotated. This single figure is the X-postable
artifact (PROJECT_CONTEXT §16, deliverable #4).

STATUS: scaffold. The plotting function signature is fixed; fill the annotation
details in Phase 6 (Step 15) once real predictions + GT are scored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def thesis_plot(
    pred_wrist: np.ndarray,      # (T,3) predicted wrist position, world frame
    gt_wrist: np.ndarray,        # (T,3) ground-truth wrist position, world frame
    out_path: str | Path,
    scale_error: float | None = None,
    ate_m: float | None = None,
    title: str = "Monocular wrist trajectory vs sensor ground truth",
) -> Path:
    """Render predicted vs GT wrist trajectory with drift/scale annotations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred_wrist = np.asarray(pred_wrist, float)
    gt_wrist = np.asarray(gt_wrist, float)
    t = np.arange(len(gt_wrist))
    err = np.linalg.norm(pred_wrist - gt_wrist, axis=1)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]}, sharex=True
    )
    for i, axis_name in enumerate("xyz"):
        ax_top.plot(t, gt_wrist[:, i], lw=1.6, label=f"GT {axis_name}")
        ax_top.plot(t, pred_wrist[:, i], lw=1.2, ls="--", label=f"pred {axis_name}")
    ax_top.set_ylabel("position (m)")
    ax_top.set_title(title)
    ax_top.legend(ncol=3, fontsize=8, loc="upper right")

    ax_bot.fill_between(t, 0, err, alpha=0.3)
    ax_bot.plot(t, err, lw=1.2)
    ax_bot.set_ylabel("world error (m)")
    ax_bot.set_xlabel("frame")

    ann = []
    if scale_error is not None:
        ann.append(f"scale err: {scale_error:+.1%}")
    if ate_m is not None:
        ann.append(f"ATE: {ate_m*100:.1f} cm")
    if ann:
        ax_bot.text(
            0.01, 0.92, "   ".join(ann), transform=ax_bot.transAxes,
            fontsize=9, va="top", family="monospace",
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
