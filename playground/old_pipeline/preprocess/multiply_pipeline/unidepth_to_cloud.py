import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
import cv2
import torch
from training.helpers.dataset import FullSceneDataset


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


# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocess_dir", type=str, required=True)
    ap.add_argument("--unidepth_dir", type=str, required=True)
    ap.add_argument("--mask_dir", type=str, required=True)
    ap.add_argument("--every_k", type=int, default=4)
    ap.add_argument("--max_frames", type=int, default=-1)
    ap.add_argument("--output_npz", type=str, required=True)
    ap.add_argument("--depth_scale", type=float, default=None,
                    help="If set, skip search and use this scale for UniDepth depths.")

    args = ap.parse_args()

    preprocess_dir = Path(args.preprocess_dir)
    image_dir = preprocess_dir / "image"
    unidepth_dir = Path(args.unidepth_dir)
    mask_dir = Path(args.mask_dir)

    # --- Camera information via FullSceneDataset
    dataset = FullSceneDataset(
        preprocess_dir=preprocess_dir,
        tids=[],
        mask_path=mask_dir,
        cloud_downsample=1,
        train_bg=False,
    )
    tr_w2c = torch.stack(dataset.pose_all, dim=0).to(torch.float32).cpu().numpy()   # [N,4,4]
    tr_Ks  = torch.stack(dataset.intrinsics_all, dim=0).to(torch.float32).cpu().numpy()
    K = tr_Ks[0, :3, :3]

    # --- Images & depth listing
    img_paths = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    assert len(img_paths) == tr_w2c.shape[0], f"#images ({len(img_paths)}) != #poses ({tr_w2c.shape[0]})."
    stem_to_idx = {p.stem: i for i, p in enumerate(img_paths)}

    depth_files = sorted(list(unidepth_dir.glob("*.npz")))
    if args.max_frames > 0:
        depth_files = depth_files[:args.max_frames]

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

        sample = dataset[idx]
        human_masks = sample["human_mask"].to(dtype=torch.bool)
        if human_masks.numel() > 0:
            union = human_masks.any(dim=0).cpu().numpy().astype(bool)
            keep = np.logical_not(union)
        else:
            keep = np.ones((H, W), dtype=bool)

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
