import torch
from pytorch3d.ops import corresponding_points_alignment

def _cam_centers_from_w2c(w2c):           # [N,4,4] -> [N,3]
    return torch.inverse(w2c)[..., :3, 3]

def _cam_centers_from_c2w(c2w):           # [N,4,4] -> [N,3]
    return c2w[..., :3, 3]

@torch.no_grad()
def _sim3_from_corr(X_src, X_tgt, estimate_scale=True):
    out = corresponding_points_alignment(
        X_src.unsqueeze(0), X_tgt.unsqueeze(0),
        weights=None, estimate_scale=estimate_scale, allow_reflection=False
    )
    return out.s[0], out.R[0], out.T[0]

def _apply_sim3(points, s, R, t):
    return (points @ R.T) * s + t

@torch.no_grad()
def _visibility_score(pts_w, K, w2c, W, H):
    ones = torch.ones(len(pts_w), 1, device=pts_w.device)
    Xw_h = torch.cat([pts_w, ones], 1)
    Xc   = (Xw_h @ w2c.T)[:, :3]
    zpos = Xc[:,2] > 0
    if zpos.sum() == 0:
        return 0.0
    uv = (Xc[zpos, :2] / (Xc[zpos, 2:3] + 1e-8)) @ K[:2,:2].T + K[:2,2]
    u, v = uv[:,0], uv[:,1]
    in_img = (u>=0) & (u<W) & (v>=0) & (v<H)
    return (in_img.float().mean().item()) * (zpos.float().mean().item())

@torch.no_grad()
def align_megasam_to_trace_auto(
    static_splats,            # ParameterDict (MegaSaM world)
    ms_c2w_all,               # [Nm,4,4] MegaSaM c2w
    trace_poses_all,          # [Nt,4,4] UNKNOWN: could be w2c or c2w
    K_render, w2c_render, W, H,   # use the actual TRACE camera you render with to score
    max_pairs=50, use_se3_then_sim3=True,
):
    dev = static_splats["means"].device
    ms_c2w_all = ms_c2w_all.to(dev).float()
    trace_poses_all = trace_poses_all.to(dev).float()
    Nm, Nt = ms_c2w_all.shape[0], trace_poses_all.shape[0]
    k = min(max_pairs, Nm, Nt)
    idx = torch.linspace(0, min(Nm,Nt)-1, steps=k).round().long().to(dev)

    C_ms = _cam_centers_from_c2w(ms_c2w_all[idx])     # MegaSaM centers

    # Two hypotheses for TRACE
    C_tr_w2c = _cam_centers_from_w2c(trace_poses_all[idx])    # if TRACE input is w2c
    C_tr_c2w = _cam_centers_from_c2w(trace_poses_all[idx])    # if TRACE input is c2w

    def solve_and_score(C_tr):
        # optionally solve SE(3) first then Sim(3)
        if use_se3_then_sim3:
            s1, R1, t1 = _sim3_from_corr(C_ms, C_tr, estimate_scale=False)  # s=1
        s2, R2, t2 = _sim3_from_corr(C_ms, C_tr, estimate_scale=True)
        s, R, t = (s2, R2, t2)

        # apply to a subset of points and score
        pts0 = static_splats["means"].data
        sample = pts0[: min(20000, pts0.shape[0])]
        ptsA  = _apply_sim3(sample, s, R, t)
        score = _visibility_score(ptsA, K_render[0], w2c_render[0], W, H)
        return s, R, t, score

    s_w2c, R_w2c, t_w2c, score_w2c = solve_and_score(C_tr_w2c)
    s_c2w, R_c2w, t_c2w, score_c2w = solve_and_score(C_tr_c2w)

    # pick the better interpretation
    if score_w2c >= score_c2w:
        s, R, t = s_w2c, R_w2c, t_w2c
        picked = "TRACE poses interpreted as w2c"
    else:
        s, R, t = s_c2w, R_c2w, t_c2w
        picked = "TRACE poses interpreted as c2w"

    # apply to ALL points (in-place)
    pts_all = static_splats["means"].data
    pts_aligned = _apply_sim3(pts_all, s, R, t)
    static_splats["means"].data.copy_(pts_aligned)

    # final report
    final_score = _visibility_score(pts_aligned[:min(20000, pts_aligned.shape[0])],
                                    K_render[0], w2c_render[0], W, H)
    print(f"[ALIGN] picked: {picked}")
    print(f"[ALIGN] s={float(s):.6f}, ||t||={t.norm().item():.3f}, score={final_score:.3f}")
    return {"s": s, "R": R, "t": t, "score": final_score, "picked": picked}
