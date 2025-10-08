import torch
from torch.nn import functional as F
from training.helpers.model_init import SceneSplats



def update_skinning_weights(all_gs: SceneSplats, k: int = 4, eps: float = 1e-6, p: float = 1.0, device="cuda", dtype=torch.float32) -> torch.Tensor:

    dynamic_gs = all_gs.dynamic
    smpl_c_info = all_gs.smpl_c_info

    smpl_V = smpl_c_info["verts_c"].to(device=device, dtype=dtype)     # [N, 3]
    smpl_W = smpl_c_info["weights_c"].to(device=device, dtype=dtype)   # [N, 24]

    new_W_list = list()
    for splats in dynamic_gs:
        means_all = splats["means"].to(device=device, dtype=dtype) # [M, 3]

        M = means_all.shape[0]
        N = smpl_V.shape[0]
        k_eff = min(max(int(k), 1), N)

        # Distances: [M, N]
        dists = torch.cdist(means_all, smpl_V)  # O(M*N)

        # k-NN: [M, k]
        topk_d, topk_idx = torch.topk(dists, k=k_eff, dim=1, largest=False)

        # Gather neighbor weights: [M, k, 24]
        smpl_W_knn = smpl_W[topk_idx]

        # Handle zero-distance rows robustly
        zero_mask = topk_d <= eps                         # [M, k]
        any_zero  = zero_mask.any(dim=1, keepdim=True)    # [M, 1]

        # Inverse-distance weights for non-zero case: w_j ∝ 1 / d_j^p
        inv = 1.0 / torch.clamp(topk_d, min=eps).pow(p)   # [M, k]
        w_geo = inv / (inv.sum(dim=1, keepdim=True) + eps)

        # If any zero distances: distribute uniformly across zero-distance neighbors
        zcount = zero_mask.sum(dim=1, keepdim=True).clamp(min=1)   # [M,1]
        w_zero = zero_mask.float() / zcount                        # [M, k]

        # Select per-row: use zero-based uniform weights if any zero exists, else inverse-distance
        w_final = torch.where(any_zero, w_zero, w_geo)             # [M, k]

        # Blend neighbor SMPL weights: [M, 24]
        new_W = (w_final.unsqueeze(-1) * smpl_W_knn).sum(dim=1)

        # Safety renorm
        new_W = torch.clamp(new_W, min=0)
        new_W = new_W / (new_W.sum(dim=1, keepdim=True) + eps)

        new_W_list.append(new_W)

    return new_W_list


def canon_to_posed(smpl_server, smpl_params, verts_c, lbs_weights, device="cuda"):
    """Transform vertices from canonical to posed space using LBS.

    Args:
        smpl_server: SMPLServer instance.
        smpl_params: SMPL parameters. Shape: [1, 86].
        verts_c: Canonical vertices. Shape: [M, 3].
        lbs_weights: Skinning weights for canonical vertices. Shape: [M, 24].
    """

    tsf = smpl_server(smpl_params.to(device), absolute=False)["smpl_tfs"]
    x_c = verts_c
    x_c_h = F.pad(x_c, (0, 1), value=1.0)
    x_p_h = torch.einsum("pn,bnij,pj->bpi", lbs_weights, tsf, x_c_h)
    verts_p = x_p_h[:, :, :3] / x_p_h[:, :, 3:4]

    return verts_p
