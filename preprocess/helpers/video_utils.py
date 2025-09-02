import os
import cv2
import subprocess
from pathlib import Path


def extract_frames(cfg):

    output_image_folder = f"{cfg.output_dir}/preprocess/images"
    os.makedirs(output_image_folder, exist_ok=True)

    # if exists then clear the folder from images
    if os.path.exists(output_image_folder):
        print("--- FYI: Clearing images since there were already some in the output folder")
        for file in os.listdir(output_image_folder):
            file_path = os.path.join(output_image_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    cap = cv2.VideoCapture(cfg.frame_extraction.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sample_every = int(fps // cfg.frame_extraction.sample_fps) if cfg.frame_extraction.sample_fps else 1
    print(f"--- FYI: Sampling every {sample_every} frames")
    frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            out_path = os.path.join(output_image_folder, f"frame_{saved_idx:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved_idx += 1
        frame_idx += 1
    cap.release()

    print(f"--- FYI: Saved {saved_idx} frames to {output_image_folder}")

def extract_frame_id(name: str) -> int:
    return int(name.split("_")[1].split(".")[0])

def load_images(img_dir: str):
    img_dir = Path(img_dir)
    img_dict = dict()

    paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

    for img_fname in paths:
        img = cv2.imread(str(img_fname))
        fid = extract_frame_id(img_fname.name)
        img_dict[fid] = img

    return img_dict


def frames_to_video(frames_dir, output_file, framerate=24):
    command = [
        "ffmpeg",
        "-framerate", str(framerate),
        "-i", f"{frames_dir}/frame_%05d.png",  # matches frame_00000.png, frame_00001.png...
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_file
    ]
    subprocess.run(command, check=True)