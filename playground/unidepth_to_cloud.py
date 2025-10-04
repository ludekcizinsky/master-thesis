#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import cv2
import torch
from training.helpers.dataset import TraceDataset


# ---------------------- helpers ----------------------

def load_masks_for_frame(mask_root: Path, stem: str, target_hw):
    Ht, Wt = target_hw
    union = np.zeros((Ht, Wt), dtype=np.uint8)
    for pid in [0, 1]:
        mpath = None
        for ext in (".jpg", ".png"):
            p = mask_root / str(pid) / f"{stem}{ext}"
            if p.exists():
                mpath = p
                break
        if mpath is None:
            continue
        m = imageio.imread(mpath)
        if m.ndim == 3:
            m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
        if (m.shape[0], m.shape[1]) != (Ht, Wt):
            m = cv2.resize(m, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
        union = np.maximum(union, (m > 0).astype(np.uint8))
    return (union == 0)  # True = keep background


def w2c_to_c2w(R: np.ndarray, t: np.ndarray):
    Rcw = R.T
    tcw = -R.T @ t
    return Rcw, tcw


def backproject_depth_to_world(depth, K, w2c_4x4, rgb, keep_mask, every_k=4, depth_scale=1.0):
    H, W = depth.shape
    yy, xx = np.mgrid[0:H:every_k, 0:W:every_k]
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)

    valid = keep_mask[yy, xx] & np.isfinite(depth[yy, xx]) & (depth[yy, xx] > 0)
    if not np.any(valid):
        return np.empty((0,3), np.float32), np.empty((0,3), np.uint8), (np.empty(0), np.empty(0))

    u = xx[valid].astype(np.float32)
    v = yy[valid].astype(np.float32)
    z = depth[v.astype(int), u.astype(int)].astype(np.float32) * float(depth_scale)

    Kinv = np.linalg.inv(K)
    pix = np.stack([u, v, np.ones_like(u)], axis=0)  # 3xN
    rays = Kinv @ pix
    Xc = rays * z[None, :]                            # 3xN

    R = w2c_4x4[:3, :3]
    t = w2c_4x4[:3, 3]
    Rcw, tcw = w2c_to_c2w(R, t)
    Xw = (Rcw @ Xc) + tcw[:, None]                    # 3xN

    cols = rgb[v.astype(int), u.astype(int), :]
    return Xw.T.astype(np.float32), cols.astype(np.uint8), (u, v)  # return (u,v) for potential debugging


def project_world_to_cam_pixels(Xw, K, w2c_4x4):
    """
    Xw: Nx3 world points
    returns: (u,v,z_cam) with z_cam > 0 = in front of camera
    """
    if Xw.shape[0] == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    R = w2c_4x4[:3, :3]
    t = w2c_4x4[:3, 3]
    Xc = (R @ Xw.T + t[:, None])  # 3xN
    z = Xc[2, :]
    uvw = K @ Xc
    u = uvw[0, :] / z
    v = uvw[1, :] / z
    return u, v, z


def sample_depth_at(depth, u, v):
    """Nearest-neighbor sampling at floating (u,v); returns sampled depth and a valid mask."""
    H, W = depth.shape
    ui = np.rint(u).astype(int)
    vi = np.rint(v).astype(int)
    valid = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    if not np.any(valid):
        return np.empty(0), valid
    return depth[vi[valid], ui[valid]], valid


def camera_centers_from_w2c_all(w2c_all):
    R = w2c_all[..., :3, :3]
    t = w2c_all[..., :3, 3]
    Rt = np.transpose(R, (0,2,1))
    C = -(Rt @ t[..., None])[..., 0]  # [N,3]
    return C


def select_pairs_with_baseline(w2c_all, max_pairs=20, min_baseline=1e-3, stride=1):
    """
    Pick (i, j=i+Δ) pairs whose camera-center distance exceeds min_baseline.
    Returns list of (i, j).
    """
    C = camera_centers_from_w2c_all(w2c_all)
    N = C.shape[0]
    pairs = []
    for i in range(0, N-1, stride):
        # try a few forward offsets
        for delta in (1, 2, 3, 5, 10):
            j = i + delta
            if j >= N:
                break
            d = np.linalg.norm(C[j] - C[i])
            if d >= min_baseline:
                pairs.append((i, j, d))
                break
    # sort by baseline descending and keep up to max_pairs
    pairs.sort(key=lambda x: -x[2])
    pairs = [(i,j) for (i,j,_) in pairs[:max_pairs]]
    return pairs


# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess_dir", type=str, required=True)
    ap.add_argument("--unidepth_dir", type=str, required=True)
    ap.add_argument("--mask_dir", type=str, required=True)
    ap.add_argument("--every_k", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--output_npz", type=str, required=True)

    # Depth-scale search config
    ap.add_argument("--depth_scale", type=float, default=None,
                    help="If set, skip search and use this scale for UniDepth depths.")
    ap.add_argument("--alpha_min", type=float, default=0.5)
    ap.add_argument("--alpha_max", type=float, default=2.0)
    ap.add_argument("--alpha_steps", type=int, default=13)
    ap.add_argument("--eval_max_pairs", type=int, default=16)
    ap.add_argument("--eval_stride", type=int, default=1,
                    help="Stride over frames when forming candidate pairs (before baseline filter).")
    ap.add_argument("--min_baseline", type=float, default=1e-3,
                    help="Min baseline in Trace units for a pair to be used in the search.")
    ap.add_argument("--per_pair_samples", type=int, default=8000,
                    help="Max number of static pixels per pair for evaluation.")

    args = ap.parse_args()

    preprocess_dir = Path(args.preprocess_dir)
    image_dir = preprocess_dir / "image"
    unidepth_dir = Path(args.unidepth_dir)
    mask_dir = Path(args.mask_dir)

    # --- Trace (poses & intrinsics)
    ds = TraceDataset(preprocess_dir, tid=0, downscale=1)
    tr_w2c = torch.stack(ds.pose_all, dim=0).to(torch.float32).cpu().numpy()   # [N,4,4]
    tr_Ks  = torch.stack(ds.intrinsics_all, dim=0).to(torch.float32).cpu().numpy()
    K = tr_Ks[0, :3, :3]

    # --- Images & depth listing
    img_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    assert len(img_paths) == tr_w2c.shape[0], f"#images ({len(img_paths)}) != #poses ({tr_w2c.shape[0]})."
    stem_to_idx = {p.stem: i for i, p in enumerate(img_paths)}

    depth_files = sorted(list(unidepth_dir.glob("*.npz")))
    if args.max_frames > 0:
        depth_files = depth_files[:args.max_frames]

    # quick accessors
    def load_rgb_depth_keep(stem):
        p = image_dir / f"{stem}.jpg"
        if not p.exists():
            p = image_dir / f"{stem}.png"
        if not p.exists():
            return None
        rgb = imageio.imread(p)
        if rgb.ndim == 2:
            rgb = np.stack([rgb]*3, axis=-1)
        if rgb.shape[2] == 4:
            rgb = rgb[..., :3]
        H, W, _ = rgb.shape
        dfile = unidepth_dir / f"{stem}.npz"
        if not dfile.exists():
            return None
        with np.load(dfile) as npz_d:
            depth = np.asarray(npz_d["depth"]).astype(np.float32)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
        keep = load_masks_for_frame(mask_dir, stem, (H, W))
        return rgb, depth, keep, (H, W)

    # ------------- SCALE SEARCH (depth reprojection consistency) -------------
    if args.depth_scale is None:
        # select baseline-rich pairs
        pairs = select_pairs_with_baseline(tr_w2c, max_pairs=args.eval_max_pairs,
                                           min_baseline=args.min_baseline, stride=args.eval_stride)
        if len(pairs) == 0:
            print("[WARN] No frame pairs with sufficient baseline. "
                  "Scale is not identifiable from geometry. Falling back to depth_scale=1.0")
            alpha_best = 1.0
        else:
            alphas = np.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
            scores = []
            for a in alphas:
                per_pair_errors = []
                for (i, j) in pairs:
                    stem_i = img_paths[i].stem
                    stem_j = img_paths[j].stem
                    data_i = load_rgb_depth_keep(stem_i)
                    data_j = load_rgb_depth_keep(stem_j)
                    if data_i is None or data_j is None:
                        continue
                    rgb_i, depth_i, keep_i, (Hi, Wi) = data_i
                    rgb_j, depth_j, keep_j, (Hj, Wj) = data_j

                    # Build a sparse cloud in world from frame i with scale a
                    # (use a denser stride here; then we’ll subsample)
                    pts_i, _, _ = backproject_depth_to_world(
                        depth_i, K, tr_w2c[i], rgb_i, keep_i,
                        every_k=max(1, args.every_k), depth_scale=a
                    )
                    if pts_i.shape[0] == 0:
                        continue

                    # Subsample to limit compute
                    if pts_i.shape[0] > args.per_pair_samples:
                        idx_sub = np.random.choice(pts_i.shape[0], args.per_pair_samples, replace=False)
                        pts_i = pts_i[idx_sub]

                    # Reproject these points into frame j
                    u_j, v_j, z_j_pred = project_world_to_cam_pixels(pts_i, K, tr_w2c[j])

                    # Keep only points in front of camera j and inside image
                    valid = (z_j_pred > 0)
                    if not np.any(valid):
                        continue
                    u_j = u_j[valid]; v_j = v_j[valid]; z_j_pred = z_j_pred[valid]

                    # Compare with depth_j (scaled by a) at nearest pixels
                    depth_j_scaled = depth_j * a
                    z_j_obs, inb = sample_depth_at(depth_j_scaled, u_j, v_j)
                    if z_j_obs.size == 0:
                        continue
                    z_j_pred_inb = z_j_pred[inb]

                    # Robust L1 median reprojection depth error
                    err = np.abs(z_j_pred_inb - z_j_obs)
                    if err.size > 0:
                        per_pair_errors.append(np.median(err))

                score_a = np.median(per_pair_errors) if len(per_pair_errors) else np.inf
                scores.append(score_a)
                print(f"[alpha={a:.4f}] reprojection median-L1: {score_a:.6f}")

            best_idx = int(np.argmin(scores))
            alpha_best = float(alphas[best_idx])
            print(f"[RESULT] depth_scale (alpha) = {alpha_best:.6f}")
    else:
        alpha_best = float(args.depth_scale)
        print(f"[INFO] Using provided depth_scale = {alpha_best:.6f}")

    # ----------------- BUILD FINAL FUSED CLOUD WITH BEST ALPHA -----------------
    all_pts, all_cols = [], []

    for dpath in depth_files:
        stem = dpath.stem
        p = image_dir / f"{stem}.jpg"
        if not p.exists():
            p = image_dir / f"{stem}.png"
        if not p.exists():
            print(f"[WARN] Missing RGB for {stem}, skipping.")
            continue
        if stem not in stem_to_idx:
            print(f"[WARN] {stem} not in Trace mapping, skipping.")
            continue
        idx = stem_to_idx[stem]

        rgb = imageio.imread(p)
        if rgb.ndim == 2:
            rgb = np.stack([rgb]*3, axis=-1)
        if rgb.shape[2] == 4:
            rgb = rgb[..., :3]
        H, W, _ = rgb.shape

        with np.load(dpath) as npz_d:
            depth = np.asarray(npz_d["depth"]).astype(np.float32)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

        keep = load_masks_for_frame(mask_dir, stem, (H, W))

        pts, cols, _ = backproject_depth_to_world(
            depth, K, tr_w2c[idx], rgb, keep,
            every_k=args.every_k, depth_scale=alpha_best
        )
        if pts.shape[0] == 0:
            continue
        all_pts.append(pts)
        all_cols.append(cols)

    if len(all_pts) == 0:
        print("No points collected. Check inputs.")
        return

    points = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0)

    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, points=points, colors=colors, depth_scale=alpha_best)
    print(f"Done. depth_scale={alpha_best:.6f}. Saved {points.shape[0]:,} points to {out_path}.")


if __name__ == "__main__":
    main()
