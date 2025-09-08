import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import viser
import time

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

def _covariances_from_scales_quats(scales: np.ndarray, quats: np.ndarray, order="wxyz") -> np.ndarray:
    """cov = R diag(s^2) R^T, batched."""
    R = _quat_to_matrix(quats, order=order)              # [N,3,3]
    S2 = (scales.astype(np.float32) ** 2)                # [N,3]
    cov = (R * S2[:, None, :]) @ np.transpose(R, (0,2,1))
    # symmetrize + tiny clamp
    cov = 0.5 * (cov + np.transpose(cov, (0,2,1)))
    w, V = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-9)
    cov = (V * w[:, None, :]) @ np.transpose(V, (0,2,1))
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
def view_npz(npz_path: str, quat_order="wxyz", align_to="+z", scale_multiplier=1.0):
    data = np.load(npz_path)
    centers = data["means"] if "means" in data else data["means_c"]
    rgbs = data["colors"].astype(np.float32)
    if rgbs.max() > 1.0: rgbs /= 255.0
    opacities = (data["opacity"].astype(np.float32).reshape(-1,1)
                 if "opacity" in data else np.ones((centers.shape[0],1), np.float32))

    if "scales" in data: scales = data["scales"].astype(np.float32)
    else:                scales = np.exp(data["log_scales"].astype(np.float32))
    scales *= float(scale_multiplier)

    quats = data["quats"].astype(np.float32) if "quats" in data else None
    assert quats is not None, "quats are required to build anisotropic covariances."

    covs = _covariances_from_scales_quats(scales, quats, order=quat_order)

    server = viser.ViserServer(host="127.0.0.1", port=8080)
    centers[:, 2] *= -1.0
    R = _Rx(np.pi/2)  # Y -> Z
    centers, covs = _apply_global_rotation(centers, covs, R)
    server.scene.add_gaussian_splats(
        name="/gaussians",
        centers=centers.astype(np.float32),
        rgbs=np.clip(rgbs,0,1).astype(np.float32),
        opacities=np.clip(opacities,0,1).astype(np.float32),
        covariances=covs.astype(np.float32),
    )
    print("Viser at http://localhost:8080 (tunnel/forward this port)")
    try:
        while True: time.sleep(1.0)
    except KeyboardInterrupt: pass

if __name__ == "__main__":
    path = "/scratch/izar/cizinsky/thesis/output/modric_vs_ribberi/training/woven-energy-17_2owmwlvm/model_canonical.npz"
    view_npz(path, quat_order="wxyz", align_to=None, scale_multiplier=1.0)