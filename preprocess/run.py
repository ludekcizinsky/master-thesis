import hydra
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../submodules/humans4d")))
os.environ["TORCH_HOME"] = "/scratch/izar/cizinsky/.cache"
os.environ["HF_HOME"] = "/scratch/izar/cizinsky/.cache"

from omegaconf import DictConfig

from preprocess.helpers.video_utils import extract_frames
from preprocess.helpers.human4d_connector import run_human4d, load_human4d_results
from preprocess.helpers.cameras import load_default_camdicts, save_camdicts_json
from preprocess.helpers.visualise import visualise_human4d
from utils.io import save_frame_map_jsonl_with_masks


@hydra.main(config_path="../configs", config_name="preprocess.yaml", version_base=None)
def main(cfg: DictConfig):

    os.makedirs(f"{cfg.output_dir}/preprocess", exist_ok=True)
    # Step 1 / Frame Extraction
    print("ℹ️  Start of frame extraction")
    extract_frames(cfg)
    print("✅ Frame extraction completed.\n")

    # Step 2 / Human Tracking
    print("ℹ️  Start of human tracking")
    run_human4d(cfg)
    print("✅ Human tracking completed.\n")

    # Step 3 / Save Frame Map
    print("ℹ️  Start of saving frame map")
    human4d_resfile = os.path.join(cfg.output_dir, "phalp_v2", "results", "demo_images.pkl")
    humand4d_results = load_human4d_results(human4d_resfile)
    save_frame_map_jsonl_with_masks(
        humand4d_results,
        f"{cfg.output_dir}/preprocess/frame_map.jsonl",
        f"{cfg.output_dir}/preprocess/masks"
    )
    default_cam_dicts = load_default_camdicts(human4d_resfile)
    save_camdicts_json(default_cam_dicts, f"{cfg.output_dir}/preprocess/cam_dicts.json")
    print("✅ Frame map and camera dicts saving completed.\n")

    # Step 4 / Visualization
    print("ℹ️  Start of visualization")
    visualise_human4d(cfg)
    print("✅ Visualization completed.")

if __name__ == "__main__":
    main()
