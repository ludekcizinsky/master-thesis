import torch
from pytorch3d.ops import corresponding_points_alignment

def _cam_centers_from_w2c(w2c):           # [N,4,4] -> [N,3]
    return torch.inverse(w2c)[..., :3, 3]

@torch.no_grad()
def _sim3_from_corr(X_src, X_tgt, estimate_scale=True):
    out = corresponding_points_alignment(
        X_src.unsqueeze(0), X_tgt.unsqueeze(0),
        weights=None, estimate_scale=estimate_scale, allow_reflection=False
    )
    return out.s[0], out.R[0], out.T[0]

@torch.no_grad()
def align_megasam_to_trace(ms_w2c, tr_w2c, pts):

    C_ms = _cam_centers_from_w2c(ms_w2c)  # [N,3]  
    C_tr = _cam_centers_from_w2c(tr_w2c)  # [N,3]

    s, R, T = _sim3_from_corr(C_ms, C_tr, estimate_scale=True)

    pts_aligned = s * (R @ pts.T).T + T

    return pts_aligned, s, R, T
