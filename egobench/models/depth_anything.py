"""Depth-Anything-V2 metric-depth estimator, the scale-wall breaker.

A monocular RGB frame gives bearing (which pixel) but not range (how far). To place
a root-relative hand in metric camera space we need ONE scalar the hand estimator
can't give: the metric depth of the wrist along its camera ray. This module supplies
per-pixel metric depth (metres) so `commercial_safe` can sample it at the wrist.

Not a `Model` (it emits no hand joints), it's a component the world-frame pipeline
calls. Kept variant-selectable so the license axis is a one-line swap:

  variant   checkpoint (HF, metric-indoor)                          license
  --------  ------------------------------------------------------  -------------
  "small"   Depth-Anything-V2-Metric-Indoor-Small-hf   (24.8M)      Apache-2.0   <- the Apache floor (tax slice)
  "large"   Depth-Anything-V2-Metric-Indoor-Large-hf   (335M)       CC-BY-NC-4.0 <- best depth (NC, no ND -> usable)

Indoor checkpoints: EgoDex/Stera are indoor egocentric manipulation. Swap to the
outdoor metric variant only for outdoor footage.

Weights cache to HF_HOME (-> UbuntuStorage2 via .env), never the tight root.
"""

from __future__ import annotations

import numpy as np

from egobench import config  # noqa: F401  -- imports .env (HF_HOME) BEFORE transformers

_CKPT = {
    "small": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "Apache-2.0"),
    "large": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", "CC-BY-NC-4.0"),
}


class DepthEstimator:
    """Lazy-loaded metric depth. `infer(rgb)` -> (H,W) metres; `sample()` -> robust wrist depth."""

    def __init__(self, variant: str = "small", device: str | None = None, fp16: bool = True):
        if variant not in _CKPT:
            raise ValueError(f"variant must be one of {list(_CKPT)}, got {variant!r}")
        self.variant = variant
        self.checkpoint, self.license = _CKPT[variant]
        self.device = device or config.device()
        self.fp16 = fp16 and self.device == "cuda"
        self._proc = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        dt = torch.float16 if self.fp16 else torch.float32
        self._proc = AutoImageProcessor.from_pretrained(self.checkpoint)
        try:
            self._model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint, dtype=dt)
        except TypeError:  # older transformers (<5) uses torch_dtype
            self._model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint, torch_dtype=dt)
        self._model = self._model.to(self.device).eval()

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        """RGB (H,W,3) uint8 -> metric depth (H,W) float32 metres, at input resolution."""
        self._load()
        import torch
        from PIL import Image

        H, W = rgb.shape[:2]
        img = Image.fromarray(rgb) if rgb.dtype == np.uint8 else Image.fromarray(rgb.astype(np.uint8))
        inputs = self._proc(images=img, return_tensors="pt").to(self.device)
        if self.fp16:
            inputs = {k: (v.half() if v.is_floating_point() else v) for k, v in inputs.items()}
        with torch.no_grad():
            depth = self._model(**inputs).predicted_depth  # (1, h, w) metres
        depth = torch.nn.functional.interpolate(
            depth[:, None].float(), size=(H, W), mode="bilinear", align_corners=False
        )[0, 0]
        return depth.cpu().numpy().astype(np.float32)

    @staticmethod
    def sample(depth: np.ndarray, xy: np.ndarray, patch: int = 5) -> float:
        """Robust metric depth at pixel `xy=(x,y)`: median over a patch, NaN/<=0 rejected.

        Median (not mean) so a boundary pixel bleeding onto the far background can't
        drag the estimate, the single biggest failure mode of depth-at-a-keypoint.
        """
        H, W = depth.shape
        x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
        if not (0 <= x < W and 0 <= y < H):
            return float("nan")
        r = patch // 2
        x0, x1 = max(0, x - r), min(W, x + r + 1)
        y0, y1 = max(0, y - r), min(H, y + r + 1)
        vals = depth[y0:y1, x0:x1].reshape(-1)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        return float(np.median(vals)) if vals.size else float("nan")
