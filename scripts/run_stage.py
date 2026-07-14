"""Run ONE model stage over the configured clips, write predictions to .npz.

Decoupled by design: local runs the light stages (mediapipe, wilor), AWS runs
the heavy ones (hamer, mapanything, hawor). Each writes OUTPUT_ROOT/preds/
<model>/<clip_id>.npz, small files that sync across machines and merge at scoring
time. No dataset ever ships to AWS; only clip ids + code, and AWS re-fetches its
own samples.

Usage:  python scripts/run_stage.py --model wilor [--dataset egodex]

STATUS: stub. Wire in Phase 2 (Steps 8-9) for local models, Phase 3 (Step 11)
for AWS models. The model registry + .npz writer are the only shared plumbing.
"""

from __future__ import annotations

import argparse
import sys

from egobench import config

MODELS = {
    "mediapipe": "egobench.models.mediapipe_hands",
    "wilor": "egobench.models.wilor",
    "hamer": "egobench.models.hamer",
    "mapanything": "egobench.models.mapanything",
    "hawor": "egobench.models.hawor",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--dataset", choices=["egodex", "stera"])
    ap.add_argument("--config")
    args = ap.parse_args()

    cfg = config.load_config(args.config)
    dev = config.device()
    print(f"model={args.model}  device={dev}  run_on={cfg['models'][args.model]['run_on']}")

    raise NotImplementedError(
        f"stub, import {MODELS[args.model]}, run over configured clips, "
        "write .npz per clip. Phase 2/3."
    )


if __name__ == "__main__":
    sys.exit(main())
