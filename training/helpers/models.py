from pathlib import Path

import math
import torch
import torch.nn as nn
import numpy as np

from utils.smpl_deformer.smpl_server import SMPLServer
from training.helpers.utils import lbs_apply


class CanonicalGaussians(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()

        self.device = device
        self.smpl_server = SMPLServer().to(self.device).eval()
        with torch.no_grad():
            verts_c = self.smpl_server.verts_c[0].to(self.device)      # [6890,3]
            weights_c = self.smpl_server.weights_c[0].to(self.device)  # [6890,24]

        device = verts_c.device
        M = verts_c.shape[0]

        # buffers
        self.register_buffer("weights_c", weights_c, persistent=False)   # [M,24]

        # ---- trainables ----
        # means
        self.means_c = nn.Parameter(verts_c.clone())                     # [M,3]

        # anisotropic per-axis log-scales
        init_sigma = 0.02
        self.log_scales = nn.Parameter(
            torch.full((M, 3), math.log(init_sigma), device=device)
        )                                                                # [M,3]

        # rotations as quaternions [w,x,y,z], init to identity
        quats_init = torch.zeros(M, 4, device=device)
        quats_init[:, 0] = 1.0
        self.quats = nn.Parameter(quats_init)                            # [M,4]

        # colors and opacities
        self.colors = nn.Parameter(torch.rand(M, 3, device=device))      # [M,3]
        init_opacity = 0.1
        self.opacity_logit = nn.Parameter(
            torch.full((M,), torch.logit(torch.tensor(init_opacity, device=device)))
        )                                                                # [M]

        # numeric guards
        self._min_sigma = 1e-4
        self._max_sigma = 1.0

    # ---- getters ----
    def means_cam(self, smpl_param: torch.Tensor) -> torch.Tensor:

        out = self.smpl_server(smpl_param, absolute=False)
        T_rel = out["smpl_tfs"][0].to(self.device)       # [24,4,4]
        means_cam = lbs_apply(self.means_c, self.weights_c, T_rel)  # [M,3]
        return means_cam

    def opacity(self) -> torch.Tensor:
        """[M] in (0,1)"""
        return torch.sigmoid(self.opacity_logit)
    
    def get_colors(self) -> torch.Tensor:
        """[M,3] in [0,1]"""
        return self.colors.clamp(0, 1)

    def scales(self) -> torch.Tensor:
        """Per-axis std devs [M,3], clamped."""
        s = torch.exp(self.log_scales)
        return s.clamp(self._min_sigma, self._max_sigma)

    def rotations(self) -> torch.Tensor:
        """
        Return unit quaternions [w, x, y, z] with shape [M,4],
        which matches your rasteriser's expected format.
        """
        q = self.quats
        return q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    
    # ---- Saving / loading ----
    @torch.no_grad()
    def export_canonical_npz(self, path: Path):
        data = {
            "means_c": self.means_c.detach().cpu().numpy(),          # [M,3]
            "log_scales": self.log_scales.detach().cpu().numpy(),    # [M,3]
            "quats": self.rotations().detach().cpu().numpy(),        # [M,4] unit quats [w,x,y,z]
            "colors": self.get_colors().detach().cpu().numpy(),      # [M,3], [0,1]
            "opacity": self.opacity().detach().cpu().numpy(),        # [M]
        }
        np.savez(path, **data)