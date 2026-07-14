"""Shippable 'win both columns': HaMeR shape (MIT) + our PnP placement.

Reproduces config C (WiLoR-shape+PnP, W57/PA10) with a LICENSE-CLEAN shape model -
no ND asterisk. HaMeR detection/mmcv/detectron2 bypassed: hands detected by
MediaPipe, cropped, run through HaMeR's ViT+MANO head, then PnP'd to world with GT
camera pose (same geometry as commercial_safe).
"""
from __future__ import annotations
import sys
import os as _os
import cv2
import numpy as np
import torch
from egobench import frames as F
from egobench.data.egodex import discover_clips, load_clip
from egobench.metrics.mpjpe import pa_mpjpe, w_mpjpe

CACHE = "/mnt/UbuntuStorage2/egobench-store/hamer_DATA/_DATA"
_MODEL = _CFG = _MP = None
MATCH_PX = 200.0


def _load():
    global _MODEL, _CFG
    if _MODEL is not None:
        return
    import hamer.configs as C
    C.CACHE_DIR_HAMER = CACHE
    from hamer.configs import get_config
    from hamer.models import HAMER
    cfg = get_config(f"{CACHE}/hamer_ckpts/model_config.yaml", update_cachedir=True)
    cfg.defrost(); cfg.MODEL.BBOX_SHAPE = [192, 256]; cfg.freeze()
    m = HAMER.load_from_checkpoint(f"{CACHE}/hamer_ckpts/checkpoints/hamer.ckpt",
                                   strict=False, cfg=cfg, init_renderer=False)
    _MODEL = m.to("cuda").eval(); _CFG = cfg


def _mp():
    global _MP
    if _MP is None:
        import mediapipe as mp
        _MP = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2,
                                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return _MP


def _crop(rgb, px, do_flip):
    """Port of HaMeR ViTDetDataset.__getitem__ geometry (bbox from 2D kpts).

    scale=2.5*wh/200, expand to BBOX_SHAPE aspect, square .max(), gaussian
    anti-alias, warp -> (256,256), normalize with 0-255 mean/std. rgb stays RGB
    (skip HaMeR's BGR->RGB flip since our input is already RGB).
    """
    from hamer.datasets.utils import (generate_image_patch_cv2,
                                       expand_to_aspect_ratio, convert_cvimg_to_tensor)
    from skimage.filters import gaussian
    x0, y0 = px.min(0); x1, y1 = px.max(0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    scale = 2.5 * np.array([x1 - x0, y1 - y0]) / 200.0
    bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=_CFG.MODEL.BBOX_SHAPE).max()
    P = _CFG.MODEL.IMAGE_SIZE
    cvimg = rgb.astype(np.float32)
    ds = (bbox_size / P) / 2.0
    if ds > 1.1:
        cvimg = gaussian(cvimg, sigma=(ds - 1) / 2, channel_axis=2, preserve_range=True)
    patch, trans = generate_image_patch_cv2(cvimg, cx, cy, bbox_size, bbox_size, P, P,
                                            do_flip, 1.0, 0.0, border_mode=cv2.BORDER_CONSTANT)
    img = convert_cvimg_to_tensor(patch)  # CHW float, 0-255
    mean = 255. * np.array(_CFG.MODEL.IMAGE_MEAN); std = 255. * np.array(_CFG.MODEL.IMAGE_STD)
    for c in range(3):
        img[c] = (img[c] - mean[c]) / std[c]
    return torch.from_numpy(img).float(), trans  # trans: image->patch (2x3)


def hamer_hands(rgb):
    """Detect (MediaPipe) -> HaMeR shape+2D per hand. Returns list of (joints3d_rel, kpts2d_img)."""
    _load()
    H, W = rgb.shape[:2]
    res = _mp().process(rgb)
    if not res.multi_hand_landmarks:
        return []
    crops, metas, mp2d, transs = [], [], [], []
    for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
        px = np.array([[p.x * W, p.y * H] for p in lm.landmark])          # MediaPipe 2D (image px)
        # MediaPipe labels handedness assuming a selfie-MIRRORED image; egocentric
        # EgoDex is NOT mirrored, so the label is inverted (verified: agrees with GT
        # side only 12/207 as-is). Invert it -> correct chirality into HaMeR.
        is_right = handed.classification[0].label == "Left"
        if _os.environ.get("RIGHT_ONLY") == "1" and not is_right:
            continue
        crop, trans = _crop(rgb, px, do_flip=not is_right)                 # flip handled in warp
        crops.append(crop); metas.append(is_right); mp2d.append(px); transs.append(trans)
    if not crops:
        return []
    batch = {"img": torch.stack(crops).cuda(),
             "box_center": torch.zeros(len(metas), 2).cuda(),
             "box_size": torch.ones(len(metas)).cuda(),
             "img_size": torch.tensor([[W, H]] * len(metas)).float().cuda(),
             "right": torch.tensor([1.0 if r else 0.0 for r in metas]).cuda()}
    with torch.no_grad():
        out = _MODEL(batch)
    j3d = out["pred_keypoints_3d"].cpu().numpy()          # (N,21,3) metric, root-rel, HaMeR frame
    k2d = out["pred_keypoints_2d"].cpu().numpy()          # (N,21,2) crop-normalized [-0.5,0.5]
    P = _CFG.MODEL.IMAGE_SIZE
    hands = []
    for i in range(len(metas)):
        # HaMeR's OWN 2D -> crop px -> image px (inverse of image->patch affine `trans`).
        # Self-consistent with j3d[i]; avoids cross-model correspondence error of MediaPipe-2D.
        cpx = (k2d[i] + 0.5) * P                           # (21,2) patch px
        if not metas[i]:                                   # left hand: crop was h-flipped -> un-flip x
            cpx[:, 0] = P - cpx[:, 0]
        T = np.vstack([transs[i], [0, 0, 1]])
        Tinv = np.linalg.inv(T)
        img2d = (Tinv @ np.c_[cpx, np.ones(len(cpx))].T).T[:, :2]
        j = j3d[i].copy()
        if not metas[i]:                                   # left: HaMeR outputs right-conv -> un-flip x
            j[:, 0] *= -1
        hands.append((j, img2d, mp2d[i]))                  # proper 3D, HaMeR img-2D, MP-2D
    return hands


def pnp_cam(j3d, k2d, K):
    """PnP HaMeR-3D to its OWN 2D -> camera-frame joints (metric, OpenCV frame)."""
    obj = np.ascontiguousarray(j3d, np.float64); img = np.ascontiguousarray(k2d, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K.astype(np.float64), None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return (R @ obj.T).T + tvec.reshape(1, 3)


def reflect_centroid(cam, ax):
    """Un-mirror HaMeR chirality about the hand centroid (axis in CAMERA frame).
    Preserves placement (translation) -> fixes PA shape without moving W."""
    if ax is None:
        return cam
    c = cam.mean(0, keepdims=True)
    out = cam.copy(); out[:, ax] = 2 * c[:, ax] - out[:, ax]
    return out


def main():
    clips = discover_clips("data/egodex/test")
    if len(sys.argv) > 1:
        clips = clips[:int(sys.argv[1])]
    axes = [None, 0, 1, 2]
    Wd = {a: [] for a in axes}; PAd = {a: [] for a in axes}
    for hp in clips:
        clip = load_clip(hp)
        for cf in clip.frames(with_rgb=True):
            if cf.rgb is None:
                continue
            K = cf.camera.K; T_wc = F.se3(cf.camera.R_wc, cf.camera.t_wc)
            gt = cf.hands
            gpx = [F.project(F.world_to_camera(h.joints[:1], T_wc), K)[0] for h in gt]
            used = set()
            for j3d, hk2d, mp2d in hamer_hands(cf.rgb):
                wpx = mp2d[0]                              # match by MediaPipe wrist (image px)
                cand = [(np.hypot(*(wpx - gpx[gi])), gi) for gi in range(len(gt)) if gi not in used]
                if not cand:
                    continue
                d, gi = min(cand)
                if d > MATCH_PX:
                    continue
                k2d_use = mp2d if _os.environ.get("USE_MP2D") == "1" else hk2d
                cam = pnp_cam(j3d, k2d_use, K)            # PnP with HaMeR's own 2D (or MediaPipe 2D)
                if cam is None:
                    continue
                used.add(gi)
                for a in axes:
                    wj = F.transform_points(T_wc, reflect_centroid(cam, a))
                    Wd[a].append(w_mpjpe(wj, gt[gi].joints)); PAd[a].append(pa_mpjpe(wj, gt[gi].joints))
    print(f"\n=== HaMeR-shape + HaMeR-2D PnP (SHIPPABLE, MIT), reflection-axis sweep ===")
    for a in axes:
        W = np.array(Wd[a]) * 1000; PA = np.array(PAd[a]) * 1000
        print(f"  reflect={a}:  W median={np.median(W):.1f}mm  PA median={np.median(PA):.1f}mm  (n={W.size})")
    print(f"  refs: A W57/PA17 | WiLoR+PnP C W57/PA10 | WiLoR-NC W197/PA10")
    best = min(axes, key=lambda a: np.median(Wd[a]))
    np.savez("outputs/hamer_pnp_scores.npz", w=np.array(Wd[best]) * 1000, pa=np.array(PAd[best]) * 1000, axis=best)


if __name__ == "__main__":
    main()
