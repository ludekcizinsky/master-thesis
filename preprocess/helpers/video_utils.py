import os
import cv2


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