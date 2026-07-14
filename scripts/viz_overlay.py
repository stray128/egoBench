"""Overlay GT (green) vs commercial-safe reconstruction (blue) on real frames.

Reprojects both world-frame skeletons back into the image via the camera pose -
a direct visual of the geometric reconstruction landing on the hand.
"""
from __future__ import annotations
import sys
import cv2
import numpy as np
from egobench import frames as F
from egobench.data.egodex import load_clip
from egobench.models import commercial_safe as CS

EDGES = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),
         (11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]

def draw(img, j2d, color):
    for a, b in EDGES:
        pa, pb = j2d[a].astype(int), j2d[b].astype(int)
        cv2.line(img, tuple(pa), tuple(pb), color, 2)
    for p in j2d.astype(int):
        cv2.circle(img, tuple(p), 3, color, -1)

def main():
    hp = sys.argv[1] if len(sys.argv) > 1 else "data/egodex/test/assemble_disassemble_soft_legos/1.hdf5"
    clip = load_clip(hp)
    pf = CS.run(clip, scale="pnp").meta["per_frame"]
    frames = list(clip.frames(with_rgb=True))
    # pick frames spread across clip that have both a GT hand and a matched detection
    picks, tiles = [], []
    idxs = np.linspace(20, len(frames)-20, 6).astype(int)
    for fi in idxs:
        cf, dets = frames[fi], pf[fi]
        if cf.rgb is None or not dets:
            continue
        K = cf.camera.K; T = F.se3(cf.camera.R_wc, cf.camera.t_wc)
        img = cv2.cvtColor(cf.rgb, cv2.COLOR_RGB2BGR).copy()
        for h in cf.hands:                                   # GT green
            j2d = F.project(F.world_to_camera(h.joints, T), K)
            draw(img, j2d, (0, 220, 0))
        for d in dets:                                       # ours blue
            j2d = F.project(F.world_to_camera(d["joints_world"], T), K)
            draw(img, j2d, (255, 120, 0))
        cv2.putText(img, f"frame {fi}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
        tiles.append(cv2.resize(img, (640, 360)))
    if not tiles:
        print("no drawable frames"); return
    rows = [np.hstack(tiles[i:i+3]) for i in range(0, len(tiles), 3)]
    montage = np.vstack([r for r in rows if r.shape[1] == rows[0].shape[1]])
    cv2.putText(montage, "GREEN = GT   BLUE = commercial-safe (MediaPipe+PnP, Apache)",
                (20, montage.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    out = "outputs/overlay_commercial_safe.png"
    cv2.imwrite(out, montage)
    print("saved", out)

if __name__ == "__main__":
    main()
