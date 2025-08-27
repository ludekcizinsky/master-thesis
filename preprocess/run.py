import hydra
from omegaconf import DictConfig

@hydra.main(config_path="../configs", config_name="preprocess.yaml", version_base=None)
def main(cfg: DictConfig):
    print("Video path:", cfg.video_path)

if __name__ == "__main__":
    main()
