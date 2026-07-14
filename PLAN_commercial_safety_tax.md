# PLAN, The Commercial-Safety Tax → a shippable hand reconstructor

> Aim: produce the one genuinely new, tweetable, sellable result, **how much accuracy
> does a fully commercial-safe monocular hand→world pipeline cost vs the non-commercial
> chain?**, and, if the gap justifies it, build our own commercial-safe reconstructor
> to close it. Gated so we never over-build (PROJECT_CONTEXT §13).
>
> Status of what exists: egobench harness, EgoDex + Stera loaders, `frames.py`, metrics,
> the GT-free reprojection confidence method, WiLoR (NC) baseline at **W-MPJPE 191mm**.

## 0-pre. Positioning frame (locked 2026-07-13, do not reopen until Report 1 ships)

**Identity: the independent measurement authority for physical-AI human data**
(MLPerf/AnandTech-style). Vendors (Mecka $60M/EgoVerse, FPV/Stera, Macrodata) sell
data, they cannot credibly grade their own output. We sell no data → structurally
un-conflicted. Each shipped report compounds the authority.

- **Report 1 (this week): the tax number**, §2 below. Fastest, safest, unpublished.
- **Report 2: cross-dataset robustness / failure taxonomy** (EgoDex, Stera, EgoVerse
  if license allows). Confidence signal is the *instrument inside* the study, never
  the headline, reprojection residual alone is standard CV, not a contribution.
- **Report 3 (gated on 1+2 landing): reprojection self-correction**, use the signal
  to *shrink* the 191mm cliff at inference. Method novelty; high risk.

**Honest-novelty rules learned this session:**
- The confidence signal is table-stakes as an algorithm; novelty lives in the
  finding / number / position, not the residual.
- Tax number is computable **only on EgoDex** (real ARKit hand GT). Stera hands are
  WiLoR pseudo-labels → circular as GT; Stera is wild-demo + camera-GT geometry only.
- The A2 *method* ladder is where technical novelty can live: depth-net wrist
  sampling (baseline, common) → bone-length anthropometric prior → **known-motion
  triangulation of the wrist using GT camera poses** (zero learned parts) → fused
  w/ confidence-weighted filtering. Founder (CV engineer) contributes here.

---

## 0. The license logic that drives the whole plan

This is the constraint everything follows from:

- **Evaluation on NC data is fine.** We can benchmark *any* model against EgoDex/Stera GT, that's research use. So the *tax number* is computable today.
- **Training a commercial model on NC data is NOT.** EgoDex (CC-BY-NC-ND) and Stera (CC-BY-NC-4.0) cannot be training data for a shippable model. Neither can MANO-derived corpora.
- **Therefore the only clean large-scale training source is data we generate ourselves**, synthetic (sim) with perfect GT, or self-captured. Synthetic wins: infinite GT, free, and we own the rights.

This is exactly why "build our own reconstructor" and "generate our own data in sim" are the *same* move: sim is the only legal way to get labelled hand data at scale for a commercial model.

---

## 1. The deliverable ladder (each gate decides the next)

| Milestone | What | Effort | Gate |
|---|---|---|---|
| **A** | The Tax Number, off-the-shelf commercial-safe pipeline vs WiLoR-NC | **days** | is the gap ship-small or invest-large? |
| **B** | Engineer the gap down, fusion / temporal / tiny correction head (no NC data) | 1–2 wk | good enough to ship? |
| **C** | Synthetic-data + train our own MANO-free reconstructor | weeks | beats off-the-shelf on egobench? |
| **D** | Publish the benchmark + our model's number | days |, |

**Milestone A is the whole point of this week.** It gives the tweetable number AND tells us whether B/C are worth it. Do not skip to C.

---

## 2. Milestone A, The Tax Number (this week, local, 6GB)

**Goal:** first W-MPJPE of a 100%-Apache/MIT hand→world pipeline, side-by-side with WiLoR-NC on the same egobench clips.

**Components (all commercial-safe, licenses verified):**
| stage | tool | license |
|---|---|---|
| hand 2D + metric root-relative 3D | MediaPipe Hands | Apache-2.0 |
| metric depth (breaks the scale wall) | Depth-Anything-V2-**Small** (metric variant, 24.8M) | **Apache-2.0** ✅ |
| confidence / QC | our reprojection signal | ours |
| eval harness | egobench | ours (MIT) |

**Pipeline design (minimal-dependency fusion):**
1. MediaPipe → 21 keypoints in 2D (px) + metric hand *shape* (root-relative, metres). Apache.
2. DA-V2-Small → metric depth map. Sample depth **at the wrist keypoint only**, one robust point, avoid trusting the whole noisy map.
3. Place MediaPipe's metric hand so its wrist sits at that metric depth along the camera ray (back-project wrist pixel through K at depth d). → **metric camera-frame hand.**
4. Lift to world via EgoDex GT camera extrinsic. → W-MPJPE.
5. Run our reprojection confidence on it (works on any hand output).

**Steps:**
- A1. `egobench/models/depth_anything.py` wrapper (HF `depth-anything/Depth-Anything-V2-Small`, metric head). ~1hr.
- A2. `egobench/models/commercial_safe.py`, the MediaPipe+depth fusion producing world-frame joints. The wrist-depth back-projection is the core; document the frame math in `frames.py`.
- A3. Score on the 11 EgoDex clips. Report W-MPJPE, PA-MPJPE, per-condition, same protocol as WiLoR.
- A4. **The chart:** WiLoR-NC bar (191mm) next to Commercial-Safe bar (X mm). Annotate the gap = "the tax."
- A5. Sanity overlay (green GT / blue commercial-safe) on a few clips.

**Deliverable:** the tax number + one chart. If it holds up → the tweet.

**Honest expectation:** commercial-safe will be *worse* (MediaPipe < WiLoR on shape; depth fusion adds noise). Fine, the gap is the finding, and it sets the target for B/C.

---

## 3. Gate A, the decision the number makes

- **Tax small** (say < ~1.5× WiLoR's error, and both usable): the commercial-safe pipeline is basically shippable. Publish it, ship it, done. Milestone C not needed. *Best outcome, cheapest.*
- **Tax large** (commercial-safe is 2–3×+ worse): now you have a *measured* reason to build our own model, and a target to beat. Proceed to B, then C. This is the only condition under which the big build is justified (avoids the §13 over-build trap).

---

## 4. Milestone B, engineer the gap down (commercial-safe, no training)

Cheap wins before any model training, all Apache/MIT:
- **Temporal smoothing** of the wrist-depth track (One-Euro / Kalman), kills the per-frame depth jitter.
- **Multi-point depth**, median depth over palm keypoints, not just wrist, when they're valid.
- **Hand-size metric prior**, MediaPipe's metric shape gives an expected bone length; use it to cross-check / correct the depth-derived scale (two independent scale estimates → fuse).
- **Better commercial depth**, swap DA-V2-Small for MapAnything's Apache checkpoint or ZoeDepth (MIT) if metric accuracy is better; benchmark all three.
- **A tiny learned correction head**, trained *only on synthetic* (Milestone C's data), that maps (MediaPipe joints + depth) → corrected world joints. Small MLP, commercial-safe.

Re-score after each. Gate B: is it now good enough to ship?

---

## 5. Milestone C, our own commercial-safe reconstructor (the product)

Only if Gate A says the tax is large. This is the real moat: a hand→world model with **zero NC dependency**, trained on data we own.

### 5.1 Synthetic data generation (the legal training corpus)
Perfect GT (joints, mesh, depth, camera, contact), infinite, commercial-safe because self-generated.

**Options, ranked by how quickly we can start:**
1. **Blender headless + a CC0/CC-BY hand rig**, fully open, scriptable in Python, domain randomization (textures, lighting, backgrounds, HDRI, camera jitter, object clutter). Runs on the AWS GPU. **Lowest friction, no MCP needed.** Recommended start.
2. **NVIDIA Isaac Sim + Replicator** (self-hosted on AWS GPU), physics-accurate hand-object contact + RTX rendering + built-in synthetic-data GT/randomization. Robotics-grade; heavier setup. Where your NVIDIA/Isaac access slots in.
3. **Unreal + MetaHuman**, photoreal hands. ⚠️ **Verify the EULA** for using rendered frames as ML training data, MetaHuman licensing is nuanced; confirm before relying on it.

**Key design freedom:** since we render our own data, **we define our own hand skeleton/mesh convention**, no MANO topology, no MANO license, ever. We can regress plain 3D keypoints (+ our own mesh if a "reconstructor" mesh is needed, using a CC0 hand mesh).

**Asset license is a real gate**, the hand rig/mesh we render must itself be commercial-safe (CC0/CC-BY, or one we build). Verify before generating at scale.

### 5.2 The model (MANO-free)
- **Backbone:** ViT or CNN (timm, permissive). Detect-then-regress like WiLoR, but the head outputs **direct 3D keypoints + a metric-depth/scale term**, no MANO layer.
- **Output:** 21 (or more) 3D joints in camera frame + wrist depth → world placement built in. Optionally a mesh via a CC0 hand template driven by the joints.
- **Training:** supervised on synthetic (perfect GT) with heavy domain randomization; close the sim-to-real gap with (a) photoreal rendering, (b) randomization, (c) optional self-supervised adaptation on real *unlabelled* video (legal, no labels used).
- **Compute:** AWS GPU (g5/A10 or bigger), spun on-demand, OFF between runs. Our AWS MCPs (cost-ops, pricing, serverless) manage the infra/cost.

### 5.3 Evaluation
- Score our model on **egobench** vs WiLoR-NC and the off-the-shelf commercial-safe pipeline. Same W-MPJPE protocol.
- The headline: "our commercial-safe model closes the tax from Y mm to Z mm." That's the product proof.

---

## 6. Resource inventory (what we actually have)

| resource | status | use |
|---|---|---|
| egobench harness + metrics + confidence | ✅ built | eval + QC for every milestone |
| EgoDex / Stera GT | ✅ on disk | **eval only** (NC, never training) |
| MediaPipe, DA-V2-Small, ZoeDepth, MapAnything(Apache) | available | commercial-safe pipeline parts |
| local RTX 3060 (6GB) | ✅ | Milestone A + B inference |
| AWS GPU (on-demand) | available | rendering (C) + training (C) + heavy eval |
| AWS MCPs (cost-ops, serverless, CDK, pricing) | ✅ connected | training infra + cost control |
| Blender (headless) | installable | synthetic data gen (option 1) |
| Isaac Sim / Unreal MCP | ⚠️ **not visible here**, confirm your access | synthetic data gen (options 2/3) |
| Firecrawl | ✅ | license checks, method research |

---

## 7. Risks & honesty flags

1. **Training a hand model to WiLoR-level accuracy is hard**, weeks to months, and the sim-to-real gap is real. Milestone C is a genuine research effort, not an afternoon. Gate A must justify it first.
2. **Gate A might make C unnecessary** (tax small → ship the off-the-shelf pipeline). That is a *win*, not a failure, cheapest path to a shippable, sellable stack.
3. **Asset + tool licenses must be verified** before scaling synthetic gen (hand mesh, MetaHuman EULA). One leaked NC dependency voids the whole "commercial-safe" claim.
4. **Don't let the model build become avoidance** (§13). The tax number is the gate *and* the outreach artifact, it moves the FPV / Macrodata conversations regardless of whether we build C.
5. **DA-V2 metric accuracy varies on unseen scenes**, wrist-only sampling mitigates; report honestly.

---

## 8. This week (concrete)

1. Build Milestone A (steps A1–A5). → the tax number + chart.
2. Read Gate A. Decide B/C.
3. Draft the tweet/thread around the tax number (+ the confidence method as the QC layer).
4. In parallel: verify a commercial-safe hand rig/mesh license (for C) and confirm Isaac/Unreal access.

**Start point: Milestone A1, the Depth-Anything-V2-Small wrapper.**
