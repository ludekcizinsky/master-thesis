from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose

def run_human4d(cfg):
    # Import registers the submodule's ConfigStore entries
    import submodules.humans4d.track as track  # exposes HMR2_4dhuman or similar

    # We are already inside a Hydra app => must clear before re-initializing
    GlobalHydra.instance().clear()

    # Re-initialize Hydra for the submodule's config (uses its ConfigStore)
    with initialize(version_base="1.2", config_path=None, job_name="human4d"):
        composed = compose(
            config_name="config",   # the name registered in the submodule
            overrides=[
                f"video.source={cfg.output_dir}/preprocess/images",
                f"video.output_dir={cfg.output_dir}/phalp_v2",
                "render.enable=True",
                "render.type=HUMAN_MESH",
            ],
        )

    # Run tracker programmatically
    tracker = track.HMR2_4dhuman(composed)
    tracker.track()
