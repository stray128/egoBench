"""Hero visual: shaded MANO hand mesh reprojected onto real frames.

Uses WiLoR's mesh vertices (already installed), NO detectron2, NO pytorch3d.
Renders with a plain painter's-algorithm flat shader in OpenCV. This is the
'reconstruction landing on the hand' shot for the post.
"""
from __future__ import annotations
import pickle
import cv2
import numpy as np
import torch
from egobench.data.egodex import load_clip
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

MANO_PKL = "venv/lib/python3.12/site-packages/wilor_mini/pretrained_models/MANO_RIGHT.pkl"


def load_faces():
    with open(MANO_PKL, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return np.asarray(d["f"], dtype=np.int32)          # (1538,3)


def render_mesh(img, verts_cam, faces, K, base_color=(235, 170, 130)):
    """Flat-shade + reproject a camera-frame mesh onto img (painter's algo)."""
    uv = (K @ (verts_cam / verts_cam[:, 2:3]).T).T[:, :2]      # (V,2)
    tri = verts_cam[faces]                                      # (F,3,3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    light = np.array([0.3, 0.4, -0.85]); light /= np.linalg.norm(light)
    shade = np.clip(0.35 + 0.65 * np.abs(n @ light), 0, 1)     # (F,)
    order = np.argsort(-tri[:, :, 2].mean(1))                   # far -> near
    overlay = img.copy()
    for fi in order:
        pts = uv[faces[fi]].astype(np.int32)
        c = tuple(int(base_color[k] * shade[fi]) for k in range(3))
        cv2.fillConvexPoly(overlay, pts, c, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.75, img, 0.25, 0)


def main():
    faces = load_faces()
    clip = load_clip("data/egodex/test/assemble_disassemble_soft_legos/1.hdf5")
    frames = list(clip.frames(with_rgb=True))
    pipe = WiLorHandPose3dEstimationPipeline(device=torch.device("cuda"), dtype=torch.float16)
    tiles = []
    for fi in np.linspace(40, len(frames) - 40, 6).astype(int):
        cf = frames[fi]
        if cf.rgb is None:
            continue
        H, W = cf.rgb.shape[:2]
        img = cv2.cvtColor(cf.rgb, cv2.COLOR_RGB2BGR).copy()
        for h in pipe.predict(cf.rgb):
            wp = h["wilor_preds"]
            verts = np.asarray(wp["pred_vertices"][0], float)
            cam_t = np.asarray(wp["pred_cam_t_full"][0], float)
            foc = float(np.asarray(wp["scaled_focal_length"]))
            K = np.array([[foc, 0, W / 2], [0, foc, H / 2], [0, 0, 1]])
            img = render_mesh(img, verts + cam_t, faces, K)
        tiles.append(cv2.resize(img, (640, 360)))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    montage = np.vstack([r for r in rows if r.shape[1] == rows[0].shape[1]])
    cv2.putText(montage, "MANO mesh reconstruction reprojected onto video  |  egobench",
                (20, montage.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.imwrite("outputs/mesh_hero.png", montage)
    print("saved outputs/mesh_hero.png")


if __name__ == "__main__":
    main()
