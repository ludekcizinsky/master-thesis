from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import time
import viser
import viser.transforms as tf

from training.smpl_deformer.smpl_server import SMPLServer
from evaluation.helpers.pose_estimation import load_gt_smpl_params, load_latest_smpl_checkpoint


def compute_vertices(
    smpl_server: SMPLServer,
    params: np.ndarray,
    device: torch.device,
) -> Tuple[List[List[np.ndarray]], np.ndarray]:
    faces = smpl_server.smpl.faces
    result: List[List[np.ndarray]] = []
    for frame in params:
        frame_vertices: List[np.ndarray] = []
        for smpl_param in frame:
            tensor = torch.from_numpy(smpl_param).float().unsqueeze(0).to(device)
            output = smpl_server(tensor, absolute=False)
            frame_vertices.append(output["smpl_verts"][0].cpu().numpy())
        result.append(frame_vertices)
    return result, faces


def main(
    prediction_checkpoint: Union[str, Path] = "/scratch/izar/cizinsky/thesis/output/hi4d_pair00_dance00_cam76/checkpoints/v7_hi4d_pair00_dance00_cam76/smpl/iter_005000.pt",
    gt_raw_dir: Union[str, Path] = "/scratch/izar/cizinsky/ait_datasets/full/hi4d/pair00_1/dance00/smpl",
    preprocess_dir: Union[str, Path] = "/scratch/izar/cizinsky/multiply-output/preprocessing/data/hi4d_pair00_dance00_cam76",
) -> None:
    pred_params = load_latest_smpl_checkpoint(prediction_checkpoint)
    gt_params = load_gt_smpl_params(gt_raw_dir, preprocess_dir)

    if pred_params.shape != gt_params.shape:
        raise ValueError(f"Prediction and GT shapes differ: {pred_params.shape} vs {gt_params.shape}")

    smpl_server = SMPLServer().eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smpl_server = smpl_server.to(device)

    pred_vertices, faces = compute_vertices(smpl_server, pred_params, device)
    gt_vertices, _ = compute_vertices(smpl_server, gt_params, device)

    server = viser.ViserServer()
    num_frames = pred_params.shape[0]

    frame_nodes = []
    for frame_idx in range(num_frames):
        frame_node = server.scene.add_frame(
            f"/scene/frame_{frame_idx}",
            wxyz=tf.SO3.exp(np.array([np.pi / 2.0, 0.0, 0.0])).wxyz,
            position=(0, 0, 0),
            show_axes=False,
        )

        for person_id, verts in enumerate(pred_vertices[frame_idx]):
            server.scene.add_mesh_simple(
                f"/scene/frame_{frame_idx}/pred_{person_id}",
                vertices=verts,
                faces=faces,
                color=(0, 0, 255),
            )
        for person_id, verts in enumerate(gt_vertices[frame_idx]):
            server.scene.add_mesh_simple(
                f"/scene/frame_{frame_idx}/gt_{person_id}",
                vertices=verts,
                faces=faces,
                color=(255, 0, 0),
            )

        frame_node.visible = frame_idx == 0
        frame_nodes.append(frame_node)

    with server.gui.add_folder("Playback"):
        gui_frame = server.gui.add_slider(
            "Frame",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
        )

    @gui_frame.on_update
    def _(_) -> None:
        current_frame = gui_frame.value
        with server.atomic():
            for idx, node in enumerate(frame_nodes):
                node.visible = idx == current_frame

    print("Viser server is running. Use the slider to switch frames (Ctrl+C to exit).")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down viewer...")


if __name__ == "__main__":
    main()
