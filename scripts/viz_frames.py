"""Qualitative comparison as an animated GIF (1 s per frame).

Each GIF frame is a 3x2 grid, Input | GT | MediaPipe | HaMeR | WiLoR+PnP |
WiLoR native, for one video timestep. Each method's predicted 3D-world hand is
reprojected onto the frame through the TRUE camera, so bad world placement lands
OFF the hand. Skeletons only (no MANO mesh) => license-safe. Thick bones + joint
dots, drawn at panel resolution so they stay crisp.

Outputs:
  results/qualitative.gif          animated, 1 s/frame
  results/qualitative_frames.png   static 3-frame stack (README fallback)
"""
import os, sys
os.environ.setdefault("EGOBENCH_FRAME_STRIDE", "12")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import cv2
from PIL import Image

import run_benchmark as RB
from egobench import frames as F
from egobench.data import egodex

BONES = RB.BONES
CLIP_TASK = "type_keyboard"          # clean two-hand clip; native placement clearly off
N_FRAMES = 8                         # GIF length in seconds (1 s each)
PANEL_W = 520                        # per-panel width
# (stream key, label, RGB color)
STREAMS = [
    ("input",         "Input",          (235, 235, 235)),
    ("gt",            "Ground truth",   (60, 220, 60)),
    ("mediapipe_pnp", "MediaPipe + PnP", (60, 140, 255)),
    ("hamer_pnp",     "HaMeR + PnP",    (255, 170, 40)),
    ("wilor_pnp",     "WiLoR + PnP",    (200, 80, 235)),
    ("wilor_owncam",  "WiLoR (native)", (240, 60, 60)),
]

def project(joints_world, cam):
    T_wc = F.se3(cam.R_wc, cam.t_wc)
    return F.project(F.world_to_camera(joints_world, T_wc), cam.K)

def draw(img, pts, color, sc):
    """pts in original px; sc = panel/orig scale. Thick bones + joint dots."""
    h, w = img.shape[:2]
    P = pts * sc
    for a, b in BONES:
        pa, pb = P[a], P[b]
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2, cv2.LINE_AA)
    for p in P:
        if np.all(np.isfinite(p)) and -w < p[0] < 2 * w and -h < p[1] < 2 * h:
            cv2.circle(img, (int(p[0]), int(p[1])), 7, color, -1, cv2.LINE_AA)
            cv2.circle(img, (int(p[0]), int(p[1])), 7, (25, 25, 25), 2, cv2.LINE_AA)

def panel(rgb, hands_pts, label, color, sc, W, H):
    SS = 2                                        # supersample: draw skeleton at 2x,
    im = cv2.resize(rgb, (W * SS, H * SS))        # downscale -> sub-pixel thin AA lines
    for pts in hands_pts:
        draw(im, pts, color, sc * SS)
    im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    bar = im.copy()
    cv2.rectangle(bar, (0, 0), (W, 26), (25, 25, 25), -1)
    cv2.addWeighted(bar, 0.55, im, 0.45, 0, im)
    cv2.putText(im, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return im

def collect(task):
    hdf5 = f"data/egodex/test/{task}/0.hdf5"
    clip = egodex.load_clip(hdf5)
    rgb, cams, gts = [], [], []
    for cf in clip.frames(with_rgb=True):
        rgb.append(cf.rgb.copy() if cf.rgb is not None else None)
        cams.append(cf.camera)
        gts.append([h.joints for h in cf.hands] if cf.hands else [])
    streams = {}
    for rn in ("mediapipe", "hamer", "wilor"):
        streams.update(RB.RUNNERS[rn](egodex.load_clip(hdf5)))
        RB._free_gpu()
    return rgb, cams, gts, streams

def method_pts(streams, key, fi, cam):
    sf = streams.get(key, [])
    out = []
    if fi < len(sf) and cam is not None:
        for d in sf[fi]:
            jw = d.get("joints_world")
            if jw is not None:
                out.append(project(np.asarray(jw), cam))
    return out

def grid_frame(rgb, cam, gt_hands, streams, fi):
    orig_w = rgb.shape[1]
    sc = PANEL_W / orig_w
    H = int(rgb.shape[0] * sc)
    panels = []
    for key, label, col in STREAMS:
        if key == "input":
            hp = []
        elif key == "gt":
            hp = [project(np.asarray(j), cam) for j in gt_hands]
        else:
            hp = method_pts(streams, key, fi, cam)
        panels.append(panel(rgb, hp, label, col, sc, PANEL_W, H))
    row1 = np.hstack(panels[0:3])
    row2 = np.hstack(panels[3:6])
    return np.vstack([row1, row2])

def main():
    rgb, cams, gts, streams = collect(CLIP_TASK)
    cand = [i for i in range(len(rgb)) if rgb[i] is not None and gts[i] and cams[i] is not None]
    if len(cand) > N_FRAMES:
        idx = np.linspace(0, len(cand) - 1, N_FRAMES).astype(int)
        cand = [cand[i] for i in idx]
    frames = [grid_frame(rgb[fi], cams[fi], gts[fi], streams, fi) for fi in cand]
    pil = [Image.fromarray(f) for f in frames]
    pil[0].save("results/qualitative.gif", save_all=True, append_images=pil[1:],
                duration=1000, loop=0, optimize=True)
    # static stack of first 3 frames for a still fallback
    stack = np.vstack(frames[:3])
    Image.fromarray(stack).save("results/qualitative_frames.png")
    print(f"wrote results/qualitative.gif ({len(frames)} frames, 1s each) + qualitative_frames.png")

if __name__ == "__main__":
    main()
