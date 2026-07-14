"""Score predictions against ground truth, split by which truth backs which claim.

The discipline that makes the number defensible (PROJECT_CONTEXT §9):
  - hand pose   -> scored against EgoDex GT ONLY (Vision Pro tracking).
                   Never against Stera's WiLoR-derived hands (that would be circular).
  - camera traj -> scored against Stera ARKit GT (and EgoDex extrinsics).
  - report per-clip AND per-condition (motion/lighting/occlusion). No lone aggregate.
  - lead with the out-of-distribution / hard-condition numbers, not the flattering one.
  - correlate confidence signals (egobench.confidence) against measured W-MPJPE.

STATUS: stub. Wire in Phase 4 (Steps 12-13) once loaders + predictions exist.
Metrics themselves are already implemented and tested in egobench.metrics.
"""

from __future__ import annotations

import sys

from egobench import config
from egobench.metrics import w_mpjpe, pa_mpjpe, ate, scale_error  # noqa: F401  (real, wired next)


def main() -> int:
    cfg = config.load_config()
    print(f"headline metric: {cfg['metrics']['headline']}")
    print(f"breakdown by: {cfg['metrics']['breakdown_by']}")
    raise NotImplementedError(
        "stub, Phase 4, Steps 12-13. Load GT + preds, apply metrics per clip, "
        "build the per-condition table, correlate confidence vs W-MPJPE."
    )


if __name__ == "__main__":
    sys.exit(main())
