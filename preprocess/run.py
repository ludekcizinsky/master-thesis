import hydra
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from omegaconf import DictConfig

from preprocess.helpers.video_utils import extract_frames




@hydra.main(config_path="../configs", config_name="preprocess.yaml", version_base=None)
def main(cfg: DictConfig):

    os.makedirs(f"{cfg.output_dir}/preprocess", exist_ok=True)

    # Step 1 / Frame Extraction
    print("ℹ️  Start of frame extraction")
    extract_frames(cfg)
    print("✅ Frame extraction completed.")




if __name__ == "__main__":
    main()
