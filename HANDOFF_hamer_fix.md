# HANDOFF, egobench state + HaMeR fix (2026-07-13)

Resume doc. Everything needed to continue after /compact. Read this first.

## TL;DR
- **Core result is DONE + publish-ready** (config A + C below). Numbers validated, robust, selection-bias killed.
- **HaMeR is installed + runs** (the hard part). But my quick integration gives **PA 26 mm (should be ~10)**, a crop/joint-order bug. Fixing this = the one open task the user wants done **before tweeting**.

## THE RESULTS (all on EgoDex test, 11 clips, egobench harness)

| config | what | W-MPJPE | PA-MPJPE | license |
|---|---|---|---|---|
| WiLoR-NC (ref) | MANO SOTA | 197 mm | 10 mm | CC-BY-NC-ND |
| **A** commercial-safe | MediaPipe + PnP | **56 mm** | 17 mm | **Apache (shippable)** |
| **C** ceiling | WiLoR-shape + our PnP | **57 mm** | **10 mm** | ND (benchmark-only) |
| B | A + One-Euro smoothing | 56 mm | 17 mm | Apache |
| HaMeR+PnP (BROKEN) | HaMeR-shape + our PnP | 95–150 mm | 26 mm | MIT (target: shippable C) |

**Headline (proven):** a fully license-clean MediaPipe+PnP pipeline places hands in world 3.5× better than NC WiLoR (56 vs 197 mm, same-frame n=1183, wins 95%). WiLoR wins shape (PA 10 vs 17). Config C shows: swap in a real shape model → win BOTH (57/10). HaMeR is meant to make C shippable (MIT, no ND).

**Mechanism:** WiLoR loses world placement to monocular scale bias (scale err −60mm). PnP gets metric scale from anthropometry (hand-as-ruler) → no depth net, no camera motion needed.

**3 sub-findings:** (1) metric depth nets blind to near-field hands (~1m when GT 0.3m). (2) motion parallax baseline-starved for egocentric (head translates ~4mm/8fr; GT-2D triangulation exact 0.3mm but MP-2D 45mm > PnP 15mm). (3) PnP anthropometric scale is the right primary estimator.

## THE HAMER BUG (the fix task)

**Symptom:** `scripts/score_hamer_pnp.py` → HaMeR-shape+PnP gives PA 26mm (right-only also 26 → NOT the left-flip). W 95–150mm. Should be ~57/10 like config C.

**PA is shape-only (Procrustes) → the problem is HaMeR's 3D joints vs GT.** Ranked hypotheses to test IN ORDER:

1. **Joint-order mismatch (MOST LIKELY, test first, cheap).** HaMeR uses `mano_to_openpose` order (hamer/models/mano_wrapper.py:20). WiLoR gets PA 8.5 vs EgoDex, so WiLoR's order matches EgoDex. **Verify HaMeR order actually matches** by finding the permutation minimizing PA between HaMeR `pred_keypoints_3d` and GT (or vs WiLoR's joints on the same hand). If a non-identity permutation drops PA to ~10 → hardcode it. WiLoR-mini may reorder internally in a way HaMeR doesn't.
2. **Crop preprocessing mismatch.** My crop (`generate_image_patch_cv2(img, cx, cy, size, size, 256,256,...)`) forces a **square** box; HaMeR's `ViTDetDataset` uses the real bbox aspect + BBOX_SHAPE [192,256] rescale. Port `ViTDetDataset.__getitem__` (hamer/datasets/vitdet_dataset.py) or `get_example` (hamer/datasets/utils.py:491) exactly. NOTE: vitdet_dataset may import detectron2, check; if so, copy just the crop logic.
3. **Left-hand flip**, ruled out (right-only still 26) but re-verify after 1&2.

**Fastest path:** do #1 first (a 20-line permutation diagnostic). If order was the bug, done. Then #2 if PA still >12.

## ENVIRONMENT / RECIPES (critical, non-obvious)

- **venv:** `/home/ashwith/EI/egobench/venv` (py3.12). Always `source` it. cwd resets to `/home/ashwith/EI` between Bash calls → use absolute paths or `cd /home/ashwith/EI/egobench`.
- **Big disk:** `/mnt/UbuntuStorage2/egobench-store/` (488G). Root `/` only ~40G, keep weights OFF it.
- **Env vars for runs:** `HF_HOME=/mnt/UbuntuStorage2/egobench-store/hf_cache TORCH_HOME=/mnt/UbuntuStorage2/egobench-store/torch_cache PYTHONPATH=/home/ashwith/EI/egobench`
- **Network (pip / HF / wget):** sandbox blocks it → use `dangerouslyDisableSandbox: true` on those Bash calls.
- **numpy 2.x:** `.ptp()` method removed → use `np.ptp(x)`. `np.bool/int` shims already patched in chumpy.
- **Noisy stdout:** filter with `grep -vE "Warning|warn|InitializeLog|gl_context|inference_feedback|landmark_proj|XNNPACK|feedback tensor|deprecat|shape coeff|cross|\.py:[0-9]"`. WiLoR YOLO logs per-frame → 114KB, always filter.

### HaMeR load recipe (WORKS, py3.12/6GB, no detectron2/mmcv/pytorch3d)
```python
import hamer.configs as C
C.CACHE_DIR_HAMER = "/mnt/UbuntuStorage2/egobench-store/hamer_DATA/_DATA"
from hamer.configs import get_config
from hamer.models import HAMER
cfg = get_config(f"{C.CACHE_DIR_HAMER}/hamer_ckpts/model_config.yaml", update_cachedir=True)
cfg.defrost(); cfg.MODEL.BBOX_SHAPE=[192,256]; cfg.freeze()
model = HAMER.load_from_checkpoint(f"{C.CACHE_DIR_HAMER}/hamer_ckpts/checkpoints/hamer.ckpt",
                                   strict=False, cfg=cfg, init_renderer=False).to("cuda").eval()
```
Forward output keys: `pred_keypoints_3d (N,21,3)` metric root-rel, `pred_vertices (N,778,3)`, `pred_keypoints_2d`, `pred_cam_t`, `focal_length`, `pred_mano_params`.

### Patches applied to `/mnt/UbuntuStorage2/egobench-store/hamer_src/` (editable install)
1. `hamer/utils/__init__.py`, renderer imports wrapped in try/except (pyrender optional).
2. `hamer/models/hamer.py:33`, backbone PRETRAINED_WEIGHTS load guarded with `os.path.exists`.
3. MANO copied to `hamer_DATA/_DATA/data/mano/MANO_RIGHT.pkl` (from wilor-mini pretrained_models).

Deps added (`--no-deps`): smplx, yacs, einops, pytorch_lightning, lightning_utilities, torchmetrics, lightning, fsspec, pyrender, trimesh, PyOpenGL, networkx, imageio, freetype-py, webdataset, braceexpand.

## FILES (all in /home/ashwith/EI/egobench/)
Real code: `egobench/models/{depth_anything,commercial_safe,mediapipe_hands,wilor}.py`, `egobench/frames.py`, `egobench/metrics/mpjpe.py`, `egobench/data/{egodex,stera,base}.py`, `egobench/confidence.py`.
Scripts: `scripts/{score_commercial_safe,same_frame_compare,improve_shape,proto_triangulation,viz_overlay,viz_mesh,score_hamer_pnp}.py`.
Outputs: `outputs/{tax_ladder,overlay_commercial_safe,mesh_hero}.png`, `outputs/*.npz`.
Docs: `RESULTS_report1.md` (the numbers), `PLAN_commercial_safety_tax.md`, `LANDSCAPE_UPDATE_2026-07-12.md`.

## AFTER HAMER FIX, remaining to tweet
1. Re-run `score_hamer_pnp.py` full 11 clips → confirm ~57/10 shippable (MIT).
2. Regenerate tax_ladder chart with HaMeR bar. Optional: mesh_hero via HaMeR (currently WiLoR).
3. Package: public MIT repo (**rotate HF token first**, it's in `.env`, gitignored; user must rotate at hf.co/settings/tokens). Draft X thread around: the number + config A (shippable) + config C/HaMeR (win both) + the 3 sub-findings + mesh visual.
4. Positioning: "independent measurement authority for physical-AI human data" (see memory).

## COMPETITIVE CONTEXT (for the thread / conversations)
Mecka AI ($60M, EgoVerse 1362h, Seat A, funded/crowded). FPV Labs/Stera (capture→data). ArenaX/RLMesh (RL eval runtime, adjacent, Seat-B downstream). EgoVerse license = CC-BY-SA (usable to eval/train, exclude license:null episodes). All hand datasets are pseudo-labelled → the audit wedge.
