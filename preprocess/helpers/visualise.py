import os
import cv2

import numpy as np
from tqdm import tqdm

import supervision as sv

from preprocess.helpers.pose import OP19_SKELETON_1B, xywh_to_xyxy, op25_to_op19
from preprocess.helpers.video_utils import load_images, frames_to_video

from utils.io import load_frame_map_jsonl_restore


def visualise_human4d(cfg):

    frame_path_path = f"{cfg.output_dir}/preprocess/frame_map.jsonl"
    scene_root = f"{cfg.output_dir}/preprocess/"
    h4d_results = load_frame_map_jsonl_restore(frame_path_path, scene_root=scene_root)
    print("--- FYI: Loaded Human4D results")

    img_dir = os.path.join(cfg.output_dir, "preprocess", "images")
    img_dict = load_images(img_dir)
    print("--- FYI: Loaded images")

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
        img = img_dict[fid]
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

        img_dict[fid] = img

    # Save annotated images
    frames_dir = os.path.join(cfg.output_dir, "visualizations", "tracking", "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for fid, img in img_dict.items():
        cv2.imwrite(os.path.join(frames_dir, f"frame_{fid:05d}.png"), img)

    output_file = os.path.join(cfg.output_dir, "visualizations", "tracking", "video.mp4")
    frames_to_video(frames_dir, output_file, framerate=12)