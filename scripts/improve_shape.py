"""Push shape toward WiLoR while keeping the placement win.

B. Temporal smoothing (One-Euro) of the per-side world-joint track, kills per-frame
   jitter, should lower PA (shape) and W.
C. Shape ceiling: WiLoR's metric root-relative shape (PA ~8mm) + OUR PnP placement.
   Proves 'great shape + geometric placement' wins both columns. (WiLoR is ND ->
   research/benchmark only; HaMeR MIT is the shippable swap that reproduces it.)

Baselines: commercial-safe PnP = W 56 / PA 17 ; WiLoR-NC = W 197 / PA 10 (mm).
"""
from __future__ import annotations
import numpy as np
from egobench import frames as F
from egobench.data.egodex import discover_clips, load_clip
from egobench.metrics.mpjpe import pa_mpjpe, w_mpjpe
from egobench.models import commercial_safe as CS
from egobench.models import wilor as WL

MATCH_PX = 200.0


def one_euro(series, fps=30.0, min_cutoff=1.0, beta=0.4):
    """One-Euro filter on (T,J,3) with NaN rows for missing frames. Causal."""
    T = len(series)
    out = [None] * T
    x_prev = dx_prev = t_prev = None
    def alpha(cut, dt):
        tau = 1.0 / (2 * np.pi * cut)
        return 1.0 / (1.0 + tau / dt)
    for i in range(T):
        x = series[i]
        if x is None:
            continue
        if x_prev is None:
            out[i] = x.copy(); x_prev = x.copy(); dx_prev = np.zeros_like(x); t_prev = i
            continue
        dt = (i - t_prev) / fps
        dx = (x - x_prev) / max(dt, 1e-6)
        a_d = alpha(1.0, dt)
        dx_hat = a_d * dx + (1 - a_d) * dx_prev
        cutoff = min_cutoff + beta * np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        a = alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * x_prev
        out[i] = x_hat
        x_prev, dx_prev, t_prev = x_hat, dx_hat, i
    return out


def _series_for(clip, pf, use):
    """Per-side time series of (matched world joints) using GT match. use=key in det."""
    T = clip.n_frames
    ser = {"left": [None]*T, "right": [None]*T}
    gtj = {"left": [None]*T, "right": [None]*T}
    for i, (cf, dets) in enumerate(zip(clip.frames(with_rgb=False), pf)):
        K = cf.camera.K; Tw = F.se3(cf.camera.R_wc, cf.camera.t_wc)
        gts = {h.side: h for h in cf.hands}
        gpx = {s: F.project(F.world_to_camera(h.joints[:1], Tw), K)[0] for s, h in gts.items()}
        for d in dets:
            wpx = d["kpts_2d"][0]
            s = min(gpx, key=lambda k: np.hypot(*(wpx-gpx[k])), default=None)
            if s is None or np.hypot(*(wpx-gpx[s])) > MATCH_PX:
                continue
            ser[s][i] = d[use]; gtj[s][i] = gts[s].joints
    return ser, gtj


def _score(ser, gtj):
    w, pa = [], []
    for s in ser:
        for pj, gj in zip(ser[s], gtj[s]):
            if pj is None or gj is None:
                continue
            w.append(w_mpjpe(pj, gj)); pa.append(pa_mpjpe(pj, gj))
    return np.array(w)*1000, np.array(pa)*1000


def wilor_pnp_world(clip):
    """C: PnP WiLoR's metric shape to its 2D -> world placement."""
    import cv2
    pf = WL.run(clip).meta["per_frame"]
    out = []
    for cf, dets in zip(clip.frames(with_rgb=False), pf):
        Tw = F.se3(cf.camera.R_wc, cf.camera.t_wc); K = cf.camera.K
        nd = []
        for d in dets:
            obj = np.ascontiguousarray(d["joints_rel"], np.float64)
            img = np.ascontiguousarray(d["kpts_2d"], np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj, img, K.astype(np.float64), None, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                continue
            R, _ = cv2.Rodrigues(rvec)
            cam = (R @ obj.T).T + tvec.reshape(1, 3)
            nd.append({"joints_world": F.transform_points(Tw, cam), "kpts_2d": d["kpts_2d"]})
        out.append(nd)
    return out


def main():
    A_w=A_pa=B_w=B_pa=C_w=C_pa=None
    aw=apa=bw=bpa=cw=cpa=[]
    aw,apa,bw,bpa,cw,cpa = [np.array([]) for _ in range(6)]
    for hp in discover_clips("data/egodex/test"):
        clip = load_clip(hp)
        pf = CS.run(clip, scale="pnp").meta["per_frame"]
        ser, gtj = _series_for(clip, pf, "joints_world")
        w, pa = _score(ser, gtj); aw=np.r_[aw,w]; apa=np.r_[apa,pa]
        sm = {s: one_euro(ser[s]) for s in ser}
        w, pa = _score(sm, gtj); bw=np.r_[bw,w]; bpa=np.r_[bpa,pa]
        cpf = wilor_pnp_world(clip)
        cser, cgt = _series_for(clip, cpf, "joints_world")
        w, pa = _score(cser, cgt); cw=np.r_[cw,w]; cpa=np.r_[cpa,pa]
    def row(n, w, pa): print(f"  {n:38s} W={np.median(w):6.1f}mm  PA={np.median(pa):6.1f}mm  (n={w.size})")
    print("\n=== SHAPE / SMOOTHING (median) ===")
    row("A. commercial-safe PnP (baseline)", aw, apa)
    row("B. + temporal smoothing (One-Euro)", bw, bpa)
    row("C. WiLoR-shape + our PnP placement", cw, cpa)
    print("  --- references ---")
    print("  WiLoR-NC (MANO)                        W= 196.7mm  PA=  10.0mm")
    np.savez("outputs/improve_shape.npz", aw=aw,apa=apa,bw=bw,bpa=bpa,cw=cw,cpa=cpa)


if __name__ == "__main__":
    main()
