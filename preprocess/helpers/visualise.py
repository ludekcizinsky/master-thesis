import os
import cv2

import numpy as np
from tqdm import tqdm

import supervision as sv

from preprocess.helpers.pose import OP19_SKELETON_1B, xywh_to_xyxy
from preprocess.helpers.video_utils import load_images, frames_to_video

from utils.io import load_frame_map_jsonl_restore
from utils.render import render_w_pytorch3d
from utils.smpl_deformer.smpl_server import SMPLServer
from preprocess.helpers.cameras import load_camdicts_json


def visualise_smpl(cfg, h4d_results, frames):

    cam_dicts_path = f"{cfg.output_dir}/preprocess/cam_dicts.json"
    default_cam_dicts = load_camdicts_json(cam_dicts_path)

    smpl_server = SMPLServer()
    rendered_smpl = render_w_pytorch3d(
        default_cam_dicts,
        h4d_results,
        smpl_server,
        smpl_server.smpl.faces,
        zoom_scale=1
    )

    fids = sorted(rendered_smpl.keys())

    final_imgs = []
    for fid in fids:

        rend = rendered_smpl[fid]  # (H,W,4), RGBA, uint8
        smpl_rgb = rend[..., :3].astype(np.float32) / 255.0
        a = rend[..., 3:4].astype(np.float32) / 255.0   # (H,W,1)

        frame = frames[fid].astype(np.float32) / 255.0
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        final_img = smpl_rgb * a + frame * (0.8 - a)
        final_imgs.append(final_img)

    # Save annotated images
    frames_dir = os.path.join(cfg.output_dir, "visualizations", "smpl", "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for fid, img in enumerate(final_imgs):
        img_bgr = (img * 255).clip(0,255).astype(np.uint8)[..., ::-1]
        cv2.imwrite(os.path.join(frames_dir, f"frame_{fid:05d}.png"), img_bgr)

    # Create video from frames
    output_file = os.path.join(cfg.output_dir, "visualizations", "smpl", "video.mp4")
    frames_to_video(frames_dir, output_file, framerate=12)

def visualise_tracking_results(cfg, h4d_results, frames):
    edge_annotator   = sv.EdgeAnnotator(thickness=2, edges=OP19_SKELETON_1B, color=sv.Color.BLUE)
    vertex_annotator = sv.VertexAnnotator(radius=5, color=sv.Color.GREEN)
    box_annotator    = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
    mask_annotator = sv.MaskAnnotator()

    for fid, frame_results in tqdm(h4d_results.items()):
        bboxes = []
        pids = []
        joints = []
        masks = []
        for pid, frame_res in frame_results.items():
            bboxes.append(frame_res['bbox'])
            pids.append(pid)
            joints.append(frame_res['phalp_j2ds'])
            masks.append(frame_res['phalp_mask'])
        img = frames[fid]
        bboxes = np.stack(bboxes)
        joints_arr = np.stack(joints)  # (P, 19, 2)
        xyxy_bboxes = xywh_to_xyxy(bboxes)
        masks_arr = np.stack(masks).astype(bool)  # (P, H, W)
        pids = np.array(pids)

        if len(bboxes) > 0:

            det = sv.Detections(xyxy=xyxy_bboxes, class_id=pids, mask=masks_arr)

            img = box_annotator.annotate(
                scene=img,
                detections=det,
            )
            img = label_annotator.annotate(
                scene=img,
                detections=det,
            )
            img = mask_annotator.annotate(
                scene=img,
                detections=det,
            )

        if len(joints_arr) > 0:
            kps = sv.KeyPoints(xy=joints_arr, class_id=pids)

            img = edge_annotator.annotate(
                scene=img,
                key_points=kps,
            )
            img = vertex_annotator.annotate(
                scene=img,
                key_points=kps,
            )

        frames[fid] = img

    # Save annotated images
    frames_dir = os.path.join(cfg.output_dir, "visualizations", "tracking", "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for fid, img in frames.items():
        cv2.imwrite(os.path.join(frames_dir, f"frame_{fid:05d}.png"), img)

    # Create video from frames
    output_file = os.path.join(cfg.output_dir, "visualizations", "tracking", "video.mp4")
    frames_to_video(frames_dir, output_file, framerate=12)


def visualise_human4d(cfg):

    frame_results_path = f"{cfg.output_dir}/preprocess/frame_map.jsonl"
    scene_root = f"{cfg.output_dir}/preprocess/"
    h4d_results = load_frame_map_jsonl_restore(frame_results_path, scene_root=scene_root)
    print("--- FYI: Loaded Human4D results")

    img_dir = os.path.join(cfg.output_dir, "preprocess", "images")

    frames = load_images(img_dir)
    visualise_tracking_results(cfg, h4d_results, frames)

    frames = load_images(img_dir) # reload frames since the old ones had track annotations
    visualise_smpl(cfg, h4d_results, frames)