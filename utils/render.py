from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer.mesh import TexturesVertex # modern alias

from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, MeshRasterizer, MeshRenderer, RasterizationSettings, HardPhongShader

import torch
import numpy as np
from tqdm import tqdm


def smpl_to_scene_mesh(smpl_verts_dict, smpl_faces, smpl_colors_dict, device):
    """Return ONE Meshes scene with all people merged (not a batch)."""
    meshes_list = []
    for pid, V in smpl_verts_dict.items():
        V = torch.as_tensor(V, device=device, dtype=torch.float32)
        F = torch.as_tensor(smpl_faces, device=device, dtype=torch.int64)
        C = torch.as_tensor(smpl_colors_dict[pid], device=device, dtype=torch.float32)
        C = C.view(1, 3).expand(V.shape[0], -1)
        textures = TexturesVertex(verts_features=[C])
        meshes_list.append(Meshes(verts=[V], faces=[F], textures=textures))
    if not meshes_list:
        return None
    return join_meshes_as_scene(meshes_list)  # ONE mesh with all people

def camdict_to_torch3d(camdict, device, zoom_scale=1.):
    img_size = [int(camdict['H']), int(camdict['W'])]

    if 'f' in camdict:
        fx = camdict['f'] * zoom_scale
        fy = camdict['f'] * zoom_scale
    else:
        fx = camdict['fx'] * zoom_scale
        fy = camdict['fy'] * zoom_scale

    cx = camdict['cx']
    cy = camdict['cy']


    focal_length = torch.tensor([fx, fy]).unsqueeze(0).to(device).float()
    principal_point = torch.tensor([cx, cy]).unsqueeze(0).to(device).float()


    cam_R = torch.diag(torch.tensor([1, 1, 1]))[None].float()
    cam_T = torch.zeros(3)[None].float() 
    cam_R[:, :2, :] *= -1.0
    cam_T[:, :1] *= -1.0
    cam_T[:, :2] *= -1.0
    
    cameras = PerspectiveCameras(focal_length=focal_length, principal_point=principal_point, R=cam_R, T=cam_T, device=device, in_ndc=False, image_size=[img_size])

    return cameras


@torch.no_grad()
def render_w_pytorch3d(
    render_camdicts,
    people_dict,                 
    smpl_server,                 
    smpl_faces,
    zoom_scale=1.0,
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # faces once
    if isinstance(smpl_faces, np.ndarray):
        smpl_faces = torch.tensor(smpl_faces.astype(np.int64), device=device)
    else:
        smpl_faces = smpl_faces.to(device).long()

    results = {}
    for fid in tqdm(sorted(render_camdicts.keys()), desc="Render (PyTorch3D)"):
        camdict = render_camdicts[fid]
        H, W = int(camdict["H"]), int(camdict["W"])
        cameras = camdict_to_torch3d(camdict, device, zoom_scale)

        mesh_rast = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1)
        mesh_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=mesh_rast),
            shader=HardPhongShader(device=device, cameras=cameras)
        )

        # gather verts for this frame
        smpl_verts_dict = {}
        frame_people = people_dict.get(fid, {})
        for pid, payload in frame_people.items():
            params = payload.get("smpl_params", payload.get("smpl_param"))
            if params is None:
                print(f"--- FYIL fid={fid} pid={pid} has no 'smpl_params'/'smpl_param' keys: {list(payload.keys())}")
                continue
            smpl_param = torch.as_tensor(params, device=device, dtype=torch.float32)
            if smpl_param.ndim == 1: smpl_param = smpl_param.unsqueeze(0)
            V = smpl_server(smpl_param)["smpl_verts"][0]
            smpl_verts_dict[pid] = V

        # deterministic colors
        smpl_colors_dict = {}
        for pid in smpl_verts_dict.keys():
            h = (hash(pid) if not isinstance(pid, int) else pid) % 997
            color = torch.tensor([0.4 + 0.3*((h % 7)/6.0),
                                  0.5 + 0.3*(((h//7) % 7)/6.0),
                                  0.5 + 0.3*(((h//49) % 7)/6.0)],
                                 device=device)
            smpl_colors_dict[pid] = color.clamp(0, 1)

        # render
        if not smpl_verts_dict:
            rgba = torch.zeros((H, W, 4), device=device)
        else:
            scene_mesh = smpl_to_scene_mesh(smpl_verts_dict, smpl_faces, smpl_colors_dict, device)
            rgba = mesh_renderer(meshes_world=scene_mesh)[0]  # single scene -> one image

#        out_rgb = rgba[..., :3] * rgba[..., 3:] + (1 - rgba[..., 3:]) * torch.ones((H, W, 3), device=device)
        #img = (out_rgb.clamp(0, 1) * 255).byte().cpu().numpy()[..., ::-1]
        #results[fid] = img

        img = (rgba.clamp(0, 1) * 255).byte().cpu().numpy()  # (H,W,4), RGBA
        results[fid] = img

    return results
