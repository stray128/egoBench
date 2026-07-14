# Report 1, The Commercial-Safety Tax (results log)

Dated 2026-07-13. All numbers on EgoDex `test/` (11 clips), egobench harness.
Reproduce: `python scripts/score_commercial_safe.py` · `python scripts/same_frame_compare.py`

## Headline

A fully license-clean monocular hand→world pipeline (**MediaPipe Apache-2.0 + OpenCV
PnP + our geometry**, no MANO / no NC weights) places hands in **world coordinates
3.5× more accurately** than the non-commercial SOTA (WiLoR / MANO).

### Apples-to-apples (same-frame, n=1183 GT hands both models detect)
| metric | commercial-safe (Apache) | WiLoR-NC (MANO) |
|---|---|---|
| **W-MPJPE (world placement)** | **56.4 mm** | 196.7 mm |
| PA-MPJPE (shape) | 17.1 mm | 10.0 mm |
| better placement | **95% of hands** | 5% |

**Commercial-safety tax = −140 mm (0.29×). Negative.** Being license-clean is *free*
(better, even) for world-frame placement. WiLoR keeps the shape edge.

### Robustness
- Per-clip median W-MPJPE: worst 114 mm (`add_remove_lid`), best 43 mm (`soft_legos`), all beat WiLoR.
- Pooled percentiles: p50=57, p75=90, p90=135, **p95=168 mm** (< WiLoR median). 97% beat WiLoR.
- Selection bias ruled out: number barely moved (56.6→56.4 mm) restricting to shared frames.

## Mechanism
WiLoR wins hand *shape* (MANO prior) but loses metric *placement* to monocular scale
bias (its scale error −60 mm). PnP recovers absolute scale from **anthropometry** -
MediaPipe's metric hand model makes PnP's translation metric, so wrist depth comes
from geometry, never a depth net. Different failure modes; placement is what
world-frame robot action data needs.

## Method (the stack)
`RGB → MediaPipe (2D kpts + metric 3D hand) → cv2.solvePnP(metric model ↔ 2D, K) →
camera-frame metric hand → ×GT camera extrinsic → world-frame 21 joints → W-MPJPE`.
All Apache/MIT/BSD. Code: `egobench/models/commercial_safe.py` (scale="pnp").

## Sub-findings
1. **Metric depth nets are blind to near-field hands.** DA-V2 metric-indoor (Small &
   Large) place the wrist ~1.0–1.3 m when GT is 0.31–0.37 m, off 2–3×, total near-field
   collapse (min over 91px window still ~1 m). Scale cannot come from a depth net.
   `scale="depth"` is the deliberately-weak baseline.
2. **Motion parallax is baseline-starved for egocentric.** Triangulation from known
   camera poses is geometrically exact (GT-2D → 0.3 mm error) but egocentric heads
   translate ~4 mm/8 frames (max 2.1 cm). Real MediaPipe-2D triangulation → 45 mm,
   worse than PnP's 15 mm on the same window. Use only as a fusion partner in big-
   translation moments + as a GT-free confidence cue (PnP-vs-triangulation disagreement).

## Status / next
- Core result **publish-ready** (MIT repo + X thread).
- Open: confidence-QC overlay; visual overlays; shape improvement (HaMeR slot-1 swap);
  Report 2 = EgoVerse cross-dataset audit.
