"""HaMeR shape + our PnP placement -> metric world-frame hands.

The SHAPE winner of the tax ladder. HaMeR gives an accurate root-relative hand;
our PnP lifts it to metric world frame with the clip's intrinsics + extrinsic.
Detection is MediaPipe (Apache), NOT HaMeR's detectron2/ViTDet.

LICENSE (be precise, do not overclaim): HaMeR *code* is MIT, but the hand model
is MANO, which is NON-COMMERCIAL (research-only). So this pipeline is NC-tier:
usable under an NC release, NOT commercially shippable. It is still strictly
freer than WiLoR (CC-BY-NC-ND -> cannot redistribute modified weights at all).
Only the MediaPipe+PnP pipeline (commercial_safe) is Apache/commercial-clean.
So the two-model story is: PLACEMENT is fully-Apache (MediaPipe, W~57); SHAPE
needs an NC hand model (HaMeR/MANO, PA~10.7); the ND SOTA (WiLoR) can't ship at all.

Two non-obvious things had to be right (both verified on EgoDex, see diag_hamer_*):
  1. CHIRALITY. MediaPipe labels handedness assuming a selfie-MIRRORED image;
     egocentric video is not mirrored, so the label is INVERTED (agrees with GT
     side only ~12/207). Feeding wrong chirality into HaMeR's left/right flip
     mirrors every 3D hand -> PA 26mm. Inverting the label -> PA 10.7mm (matches
     WiLoR 10), no reflection hack.
  2. SELF-CONSISTENT 2D. PnP uses HaMeR's OWN projected 2D (mapped crop->image via
     the crop affine), not MediaPipe's, so the 2D-3D correspondence is exact.

Result on EgoDex: W ~100mm / PA ~10.7mm. Wins SHAPE (PA) vs MediaPipe (PA 17);
placement (W) stays best from MediaPipe/PnP (W57).
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from egobench import frames as F
from egobench.data.base import Clip
from egobench.models.base import Prediction

name = "hamer_pnp"
space = "world"

CACHE = os.environ.get("HAMER_CACHE", "/mnt/UbuntuStorage2/egobench-store/hamer_DATA/_DATA")
_MODEL = _CFG = _MP = None


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
    """Port of HaMeR ViTDetDataset crop geometry (bbox from 2D kpts). Returns
    (CHW normalized tensor, image->patch affine trans)."""
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
    img = convert_cvimg_to_tensor(patch)
    mean = 255. * np.array(_CFG.MODEL.IMAGE_MEAN); std = 255. * np.array(_CFG.MODEL.IMAGE_STD)
    for c in range(3):
        img[c] = (img[c] - mean[c]) / std[c]
    return torch.from_numpy(img).float(), trans


def _hamer_hands(rgb):
    """Detect (MediaPipe) -> HaMeR shape per hand. Returns [(joints3d_rel, hamer_img2d, mp2d)]."""
    _load()
    H, W = rgb.shape[:2]
    res = _mp().process(rgb)
    if not res.multi_hand_landmarks:
        return []
    crops, is_rights, mp2d, transs = [], [], [], []
    for lm, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
        px = np.array([[p.x * W, p.y * H] for p in lm.landmark])
        # invert MediaPipe handedness: selfie-mirror assumption is wrong for egocentric
        is_right = handed.classification[0].label == "Left"
        crop, trans = _crop(rgb, px, do_flip=not is_right)
        crops.append(crop); is_rights.append(is_right); mp2d.append(px); transs.append(trans)
    batch = {"img": torch.stack(crops).cuda(),
             "box_center": torch.zeros(len(is_rights), 2).cuda(),
             "box_size": torch.ones(len(is_rights)).cuda(),
             "img_size": torch.tensor([[W, H]] * len(is_rights)).float().cuda(),
             "right": torch.tensor([1.0 if r else 0.0 for r in is_rights]).cuda()}
    with torch.no_grad():
        out = _MODEL(batch)
    j3d = out["pred_keypoints_3d"].cpu().numpy()      # (N,21,3) metric root-rel
    k2d = out["pred_keypoints_2d"].cpu().numpy()      # (N,21,2) crop-norm [-0.5,0.5]
    P = _CFG.MODEL.IMAGE_SIZE
    hands = []
    for i, is_right in enumerate(is_rights):
        cpx = (k2d[i] + 0.5) * P
        if not is_right:                              # left crop was h-flipped -> un-flip x
            cpx[:, 0] = P - cpx[:, 0]
        T = np.vstack([transs[i], [0, 0, 1]])
        img2d = (np.linalg.inv(T) @ np.c_[cpx, np.ones(len(cpx))].T).T[:, :2]
        j = j3d[i].copy()
        if not is_right:                              # HaMeR outputs right-conv -> un-flip x for left
            j[:, 0] *= -1
        hands.append((j, img2d, mp2d[i]))
    return hands


def _pnp_cam(j3d, k2d, K):
    obj = np.ascontiguousarray(j3d, np.float64); img = np.ascontiguousarray(k2d, np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K.astype(np.float64), None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return (R @ obj.T).T + tvec.reshape(1, 3)


def run(clip: Clip, device: str | None = None) -> Prediction:
    per_frame: list[list[dict]] = []
    for cf in clip.frames(with_rgb=True):
        dets: list[dict] = []
        if cf.rgb is not None:
            K = cf.camera.K
            T_wc = F.se3(cf.camera.R_wc, cf.camera.t_wc)
            for j3d, hk2d, mp2d in _hamer_hands(cf.rgb):
                cam = _pnp_cam(j3d, hk2d, K)          # PnP with HaMeR's OWN 2D
                if cam is None:
                    continue
                dets.append({
                    "joints_world": F.transform_points(T_wc, cam),
                    "joints_cam": cam,
                    "kpts_2d": mp2d,                  # MediaPipe 2D for GT matching (image px)
                    "side": "right",                  # unreliable; scoring re-matches by 2D wrist
                })
        per_frame.append(dets)
    return Prediction(
        clip_id=clip.clip_id, model=name, space=space,
        meta={"per_frame": per_frame, "note": "HaMeR shape (MIT code, MANO NC) + PnP; chirality-corrected"},
    )
