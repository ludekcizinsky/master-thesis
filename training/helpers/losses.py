import torch
import torch.nn.functional as F

def _chunked_cdist_min(X, Y, chunk=20000):
    M = X.shape[0]
    mins = torch.empty(M, device=X.device, dtype=X.dtype)
    for i in range(0, M, chunk):
        d = torch.cdist(X[i:i+chunk], Y)        # [m,V]
        mins[i:i+chunk] = d.min(dim=1).values   # [m]
    return mins

def anchor_to_smpl_surface(
    means_c: torch.Tensor,        # [M,3]  (ALL splats, not just first m0)
    smpl_verts_c: torch.Tensor,   # [V,3]
    free_radius: float = 0.001,    # ~1 mm
) -> torch.Tensor:
    d = _chunked_cdist_min(means_c, smpl_verts_c)   # [M]
    excess = (d - free_radius).clamp_min(0.0)
    per = excess**2
    return per.mean()


def opacity_distance_penalty(
    opa_logits: torch.Tensor,       # [M] raw opacity params; alpha = sigmoid(opa_logits)
    means_c: torch.Tensor,          # [M,3] canonical means
    smpl_verts_c: torch.Tensor,     # [V,3]
    free_radius: float = 0.01,      # meters (e.g., 5–10 mm)
    smooth: float = 0.01,           # meters: how soft the ramp grows
    stop_means_grad: bool = True,
):
    # alpha in (0,1)
    alpha = torch.sigmoid(opa_logits)        # [M]

    # distance to SMPL
    d = _chunked_cdist_min(means_c, smpl_verts_c)     # [M]
    if stop_means_grad:
        d = d.detach()  # regulate only opacity; use a separate anchor for means

    # soft hinge: weight ~ softplus((d - r)/s)
    w = F.softplus((d - free_radius) / smooth)         # [M], ~0 inside band, grows outside

    # penalize opacity outside the body band
    loss = (w * alpha).mean()

    return loss