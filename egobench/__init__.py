"""egobench, in-the-wild metric-scale egocentric reconstruction benchmark.

Scores the free open-source monocular reconstruction chain (MediaPipe / WiLoR /
HaMeR / MapAnything / HaWoR) against real sensor ground truth:

  - camera trajectory  -> Stera-10M (ARKit + LiDAR fusion)
  - 3D hand pose       -> EgoDex   (Vision Pro on-device tracking)

Headline metric is W-MPJPE (world-frame). PA-MPJPE is reported only for contrast.
"""

__version__ = "0.0.1"
