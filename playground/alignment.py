import torch
from pytorch3d.ops import corresponding_points_alignment

def _cam_centers_from_w2c(w2c: torch.Tensor) -> torch.Tensor:
    """
    w2c: [N, 4, 4] world->camera extrinsics (OpenCV convention)
    returns: [N, 3] camera centers in world coords
    Uses closed-form: C = -R^T t
    """
    R = w2c[..., :3, :3]
    t = w2c[..., :3, 3]
    C = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)
    return C

@torch.no_grad()
def _sim3_from_corr(X_src: torch.Tensor, X_tgt: torch.Tensor, estimate_scale: bool = True):
    """
    X_src, X_tgt: [N, 3] matched points
    returns: s (scalar tensor), R [3,3], T [3]
    """
    assert X_src.shape == X_tgt.shape and X_src.shape[-1] == 3
    assert X_src.shape[0] >= 3, "Need at least 3 correspondences."
    out = corresponding_points_alignment(
        X_src.unsqueeze(0),  # [1,N,3]
        X_tgt.unsqueeze(0),  # [1,N,3]
        weights=None,
        estimate_scale=estimate_scale,
        allow_reflection=False,
    )
    # out.R: [1,3,3], out.T: [1,3], out.s: [1]  (capital T in recent PyTorch3D)
    return out.s[0], out.R[0], out.T[0]

@torch.no_grad()
def align_megasam_to_trace(ms_w2c: torch.Tensor, tr_w2c: torch.Tensor, pts: torch.Tensor):
    """
    ms_w2c, tr_w2c: [N,4,4] (same frames/order, on same device/dtype)
    pts: [M,3] points in MegaSAM/world coords to transform into Trace/world coords

    returns:
      pts_aligned [M,3], s [], R [3,3], T [3]
    """
    assert ms_w2c.shape == tr_w2c.shape and ms_w2c.shape[-2:] == (4, 4)
    device = ms_w2c.device
    dtype = ms_w2c.dtype
    pts = torch.from_numpy(pts).to(device=device, dtype=dtype)

    M = torch.diag(torch.tensor([1., -1., -1.], device=pts.device, dtype=pts.dtype)) 
    C_ms = _cam_centers_from_w2c(ms_w2c)  # [N,3]
    C_ms = (C_ms @ M.T)
    C_tr = _cam_centers_from_w2c(tr_w2c)  # [N,3]

    # sanity check, compute variance of camera centers
    var_ms = torch.var(C_ms, dim=0).sum().item()
    var_tr = torch.var(C_tr, dim=0).sum().item()
    print(f"Camera center variance: MegaSAM {var_ms:.6f}, Trace {var_tr:.6f}")

    s, R, T = _sim3_from_corr(C_ms, C_tr, estimate_scale=True)  # maps MS -> TR
    print("det(R) =", torch.det(R).item(), "scale s =", s.item())


    # Apply Sim(3): X' = s * R * X + T
    # Prefer pts @ R^T for row-major point arrays
    pts_ms_ref = (pts @ M.T)
    pts_aligned = s * (pts_ms_ref @ R.transpose(0, 1)) + T

    return pts_aligned, s, R, T
