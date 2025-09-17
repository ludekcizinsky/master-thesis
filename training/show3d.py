import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import viser
import time

_C0 = 0.28209479177387814  # Y_00 = 1/(2*sqrt(pi))

def _sh_dc_to_rgb(sh_coeffs: np.ndarray) -> np.ndarray:
    """
    sh_coeffs: [N, K, 3] real SH coeffs (GraphDECO/gsplat layout).
    Uses DC-only (k=0): rgb = 0.5 + C0 * sh0.
    Returns [N,3] in [0,1].
    """
    sh0 = sh_coeffs[:, 0, :]            # [N,3]
    rgb = 0.5 + _C0 * sh0
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


# ---------- math utils (pure NumPy) ----------
def _quat_to_matrix(quats: np.ndarray, order="wxyz") -> np.ndarray:
    """quats: [N,4]; returns [N,3,3]."""
    q = quats.astype(np.float32)
    if order == "xyzw":
        q = q[:, [3,0,1,2]]  # -> wxyz
    # normalize
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.stack([
        1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),
        2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),
        2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy)
    ], axis=1).reshape(-1,3,3)
    return R.astype(np.float32)

def _covs_for_viewer(log_scales, quats, order="wxyz",
                     sigma_clip=(1e-4, 5e-3),   # 0.1–5 mm for viz
                     max_anisotropy=20.0,
                     use_precision=False):
    # 1) build Σ = R diag(exp(2*log_scales)) R^T
    s = np.exp(log_scales).astype(np.float32)             # [N,3]
    R = _quat_to_matrix(quats.astype(np.float32), order)  # [N,3,3]
    S2 = s**2
    cov = (R * S2[:, None, :]) @ np.transpose(R, (0,2,1)) # [N,3,3]

    # 2) clamp eigenvalues for visualization
    w, V = np.linalg.eigh(cov)
    w = np.clip(w, sigma_clip[0]**2, sigma_clip[1]**2)    # hard size clamp

    # anisotropy cap
    ratio = w.max(axis=1) / np.maximum(w.min(axis=1), 1e-12)
    bad = ratio > max_anisotropy
    if bad.any():
        hi = w[bad].max(axis=1, keepdims=True)
        lo = hi / max_anisotropy
        w[bad] = np.clip(w[bad], lo, hi)

    cov = (V * w[:, None, :]) @ np.transpose(V, (0,2,1))
    cov = 0.5 * (cov + np.transpose(cov, (0,2,1)))

    # 3) some viewers want precision instead
    if use_precision:
        eye = np.eye(3, dtype=np.float32)
        cov = np.linalg.inv(cov + 1e-9 * eye)

    return cov.astype(np.float32)

def _Rx(theta):  # rotate about +X by theta
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], np.float32)

def _apply_global_rotation(centers, covs, R):
    # rotate around the cloud centroid so position stays put
    mean = centers.mean(0, keepdims=True).astype(np.float32)
    C = (centers.astype(np.float32) - mean) @ R.T + mean
    Cov = (R[None] @ covs.astype(np.float32) @ R.T)
    return C, Cov

# ---------- viewer ----------
def view_npz(npz_path: str, quat_order="wxyz", scale_multiplier=1.0):
    data = np.load(npz_path)
    # means
    centers = data["means_c"]
    # rgbs
    colors = data["colors"]
    sh0, shN = colors[:,:1,:], colors[:,1:,:]
    sh_coeffs = np.concatenate([sh0, shN], axis=1).astype(np.float32)  # [N,K,3]
    rgbs = _sh_dc_to_rgb(sh_coeffs)
    # opacities
    # opacities = torch.logit(torch.from_numpy(data["opacity"])).numpy().astype(np.float32) # [N,]
    # opacities = opacities[:, None]  # [N, 1]
    opacities = data["opacity"].astype(np.float32)[:, None]  
    # covariances
    covariances = _covs_for_viewer(
        data["log_scales"], data["quats"], quat_order,
        sigma_clip=(1e-4, 5e-3),   # feel free to tweak
        max_anisotropy=20.0,
        use_precision=False
    )

    server = viser.ViserServer(host="127.0.0.1", port=8080)
    centers[:, 2] *= -1.0
    R = _Rx(np.pi/2)  # Y -> Z
    centers, covariances = _apply_global_rotation(centers, covariances, R)
    server.scene.add_gaussian_splats(
        name="/gaussians",
        centers=centers,
        rgbs=rgbs,
        opacities=opacities,
        covariances=covariances
    )
    print("Viser at http://localhost:8080 (tunnel/forward this port)")
    try:
        while True: time.sleep(1.0)
    except KeyboardInterrupt: pass

if __name__ == "__main__":
    path = "/scratch/izar/cizinsky/thesis/output/modric_vs_ribberi/training/tid_1/dummy-orkus23s_orkus23s/checkpoints/model_canonical_final.npz"
    view_npz(path, quat_order="wxyz", scale_multiplier=1.0)