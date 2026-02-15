from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import tyro
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.path_config import ensure_runtime_dirs, load_runtime_paths


def _default_difix_data_root_dir() -> Path:
    return Path(os.getenv("THESIS_DIFIX_DATA_ROOT_DIR", "/scratch/izar/cizinsky/thesis/difix_tuning_data"))


@dataclass
class SlurmConfig:
    job_name: str = "difix_tune"
    slurm_script: Path = Path("training/difix_tune/submit.slurm")
    time: str = "12:00:00"
    cpus_per_task: int = 8
    mem: str = "96G"
    gpus: int = 1
    account: str = "master"
    partition: Optional[str] = None
    qos: Optional[str] = None


@dataclass
class Config:
    repo_dir: Path = REPO_ROOT
    paths_config: Path = Path("configs/paths.yaml")
    difix_train_script: Path = Path("submodules/difix3d/src/train_difix.py")

    # Dataset + run naming
    exp_name: str = "v0_difix_data_full"
    run_name: str = "run_000"
    data_root_dir: Path = field(default_factory=_default_difix_data_root_dir)
    dataset_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    resume: Optional[Path] = None

    # Accelerate launch config (single-node only)
    mixed_precision: str = "fp16"
    num_processes: int = 1
    main_process_port: int = 29501
    conda_env: str = "thesis"

    # train_difix.py args
    max_train_steps: int = 10000
    resolution: int = 512
    base_model_id: str = "stabilityai/sd-turbo"
    learning_rate: float = 2e-5
    train_batch_size: int = 1
    dataloader_num_workers: int = 8
    enable_xformers_memory_efficient_attention: bool = True
    checkpointing_steps: int = 1000
    eval_freq: int = 1000
    viz_freq: int = 100
    max_grad_norm: float = 1.0
    lambda_lpips: float = 1.0
    lambda_l2: float = 1.0
    lambda_gram: float = 1.0
    gram_loss_warmup_steps: int = 2000
    report_to: str = "wandb"
    tracker_project_name: str = "difix"
    tracker_run_name: Optional[str] = None
    timestep: int = 199
    extra_train_args: List[str] = field(default_factory=list)

    submit: bool = False
    dry_run: bool = False
    slurm: SlurmConfig = field(default_factory=SlurmConfig)


_MANIFEST_REL_PATH = Path("meta/run_config.yaml")
_CONFIG_SKIP_FIELDS = {"resume", "submit", "dry_run", "slurm"}
_PATH_FIELDS = {
    "repo_dir",
    "paths_config",
    "difix_train_script",
    "data_root_dir",
    "dataset_path",
    "output_dir",
}


def _resolve_repo_path(repo_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_dir / path


def _resolve_dataset_path(cfg: Config) -> Path:
    if cfg.dataset_path is not None:
        return _resolve_repo_path(cfg.repo_dir, cfg.dataset_path)
    return cfg.data_root_dir / cfg.exp_name / "difix" / "data.json"


def _resolve_output_dir(cfg: Config) -> Path:
    if cfg.output_dir is not None:
        return _resolve_repo_path(cfg.repo_dir, cfg.output_dir)
    return cfg.data_root_dir / cfg.exp_name / "tuning_runs" / cfg.run_name


def _to_plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return value


def _config_for_manifest(cfg: Config) -> dict:
    raw = asdict(cfg)
    for field_name in _CONFIG_SKIP_FIELDS:
        raw.pop(field_name, None)
    return _to_plain(raw)


def _manifest_path_for_output_dir(output_dir: Path) -> Path:
    return output_dir / _MANIFEST_REL_PATH


def _resolve_resume_run_dir(resume_path: Path) -> Path:
    if resume_path.is_file():
        if resume_path.suffix != ".pkl":
            raise ValueError(f"Unsupported --resume file path: {resume_path}")
        if resume_path.parent.name == "checkpoints":
            return resume_path.parent.parent
        return resume_path.parent
    if resume_path.is_dir():
        if resume_path.name == "checkpoints":
            return resume_path.parent
        if (resume_path / "checkpoints").exists():
            return resume_path
        raise ValueError(
            f"Unsupported --resume directory path: {resume_path}. "
            "Pass either a checkpoints dir, run dir, or model_*.pkl file."
        )
    raise FileNotFoundError(f"--resume path not found: {resume_path}")


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Run config manifest not found: {manifest_path}. "
            "This repo requires a saved run config for resume."
        )
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict) or "config" not in payload:
        raise ValueError(f"Invalid run config manifest format: {manifest_path}")
    return payload


def _save_manifest(cfg: Config, dataset_path: Path, output_dir: Path) -> Path:
    manifest_path = _manifest_path_for_output_dir(output_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": _config_for_manifest(cfg),
        "resolved": {
            "dataset_path": str(dataset_path.resolve()),
            "output_dir": str(output_dir.resolve()),
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return manifest_path


def _load_resume_locked_config(cfg: Config) -> Config:
    if cfg.resume is None:
        return cfg
    resume_path = _resolve_repo_path(cfg.repo_dir, cfg.resume)
    run_dir = _resolve_resume_run_dir(resume_path)
    manifest_path = _manifest_path_for_output_dir(run_dir)
    payload = _load_manifest(manifest_path)

    saved_cfg = payload["config"]
    if not isinstance(saved_cfg, dict):
        raise ValueError(f"Invalid 'config' payload in {manifest_path}")

    for key, value in saved_cfg.items():
        if key in _CONFIG_SKIP_FIELDS:
            continue
        if not hasattr(cfg, key):
            continue
        if key in _PATH_FIELDS and value is not None:
            setattr(cfg, key, Path(value))
        else:
            setattr(cfg, key, value)

    resolved = payload.get("resolved", {})
    dataset_path = resolved.get("dataset_path")
    output_dir = resolved.get("output_dir")
    if dataset_path:
        cfg.dataset_path = Path(dataset_path)
    if output_dir:
        cfg.output_dir = Path(output_dir)

    print(f"Loaded run config manifest for resume: {manifest_path}")
    return cfg


def _build_accelerate_cmd(cfg: Config, dataset_path: Path, output_dir: Path) -> List[str]:
    if cfg.num_processes < 1:
        raise ValueError(f"--num-processes must be >= 1, got {cfg.num_processes}.")

    script_path = _resolve_repo_path(cfg.repo_dir, cfg.difix_train_script)
    if not script_path.exists():
        raise FileNotFoundError(f"DiFix train script not found: {script_path}")

    cmd = [
        "accelerate",
        "launch",
        "--mixed_precision",
        cfg.mixed_precision,
    ]
    if cfg.num_processes > 1:
        cmd.extend(
            [
                "--multi_gpu",
                "--num_machines",
                "1",
                "--num_processes",
                str(cfg.num_processes),
                "--main_process_port",
                str(cfg.main_process_port),
            ]
        )

    cmd.extend(
        [
            str(script_path),
            "--output_dir",
            str(output_dir),
            "--dataset_path",
            str(dataset_path),
            "--max_train_steps",
            str(cfg.max_train_steps),
            "--resolution",
            str(cfg.resolution),
            "--base_model_id",
            cfg.base_model_id,
            "--learning_rate",
            str(cfg.learning_rate),
            "--train_batch_size",
            str(cfg.train_batch_size),
            "--dataloader_num_workers",
            str(cfg.dataloader_num_workers),
            "--checkpointing_steps",
            str(cfg.checkpointing_steps),
            "--eval_freq",
            str(cfg.eval_freq),
            "--viz_freq",
            str(cfg.viz_freq),
            "--max_grad_norm",
            str(cfg.max_grad_norm),
            "--lambda_lpips",
            str(cfg.lambda_lpips),
            "--lambda_l2",
            str(cfg.lambda_l2),
            "--lambda_gram",
            str(cfg.lambda_gram),
            "--gram_loss_warmup_steps",
            str(cfg.gram_loss_warmup_steps),
            "--report_to",
            cfg.report_to,
            "--tracker_project_name",
            cfg.tracker_project_name,
            "--tracker_run_name",
            cfg.tracker_run_name or f"{cfg.exp_name}_{cfg.run_name}",
            "--timestep",
            str(cfg.timestep),
        ]
    )
    if cfg.resume is not None:
        resume_path = _resolve_repo_path(cfg.repo_dir, cfg.resume)
        cmd.extend(["--resume", str(resume_path)])
    if cfg.enable_xformers_memory_efficient_attention:
        cmd.append("--enable_xformers_memory_efficient_attention")
    cmd.extend(cfg.extra_train_args)
    return cmd


def _print_plan(cfg: Config, dataset_path: Path, output_dir: Path, launch_cmd: Sequence[str]) -> None:
    print("DiFix Tuning Plan:")
    print(f"  repo_dir: {cfg.repo_dir}")
    print(f"  exp_name: {cfg.exp_name}")
    print(f"  run_name: {cfg.run_name}")
    print(f"  dataset_path: {dataset_path}")
    print(f"  output_dir: {output_dir}")
    print(f"  base_model_id: {cfg.base_model_id}")
    if cfg.resume is not None:
        print(f"  resume: {_resolve_repo_path(cfg.repo_dir, cfg.resume)}")
    print(f"  num_processes: {cfg.num_processes}")
    print(f"  mixed_precision: {cfg.mixed_precision}")
    if cfg.submit:
        print("  mode: submit")
        print(f"  slurm.job_name: {cfg.slurm.job_name}")
        print(f"  slurm.time: {cfg.slurm.time}")
        print(f"  slurm.gpus: {cfg.slurm.gpus}")
        print(f"  slurm.cpus_per_task: {cfg.slurm.cpus_per_task}")
        print(f"  slurm.mem: {cfg.slurm.mem}")
        print(f"  slurm.account: {cfg.slurm.account}")
        if cfg.slurm.partition:
            print(f"  slurm.partition: {cfg.slurm.partition}")
        if cfg.slurm.qos:
            print(f"  slurm.qos: {cfg.slurm.qos}")
    else:
        print("  mode: local")
    print("  launch_cmd:")
    print("    " + " ".join(shlex.quote(part) for part in launch_cmd))


def _confirm_submit() -> bool:
    try:
        response = input("Press Enter to submit, or type anything to cancel: ")
    except EOFError:
        return False
    return response.strip() == ""


def _build_export_env(runtime_paths, cfg: Config) -> str:
    hf_cache_root = runtime_paths.hf_cache_dir
    return ",".join(
        [
            "ALL",
            f"THESIS_WANDB_ROOT_DIR={runtime_paths.wandb_root_dir}",
            f"THESIS_HYDRA_ROOT_DIR={runtime_paths.hydra_root_dir}",
            f"THESIS_RESULTS_ROOT_DIR={runtime_paths.results_root_dir}",
            f"THESIS_PREPROCESSING_ROOT_DIR={runtime_paths.preprocessing_root_dir}",
            f"THESIS_CANONICAL_GT_ROOT_DIR={runtime_paths.canonical_gt_root_dir}",
            f"THESIS_DIFIX_DATA_ROOT_DIR={cfg.data_root_dir}",
            f"THESIS_CONDA_ENV={cfg.conda_env}",
            f"THESIS_HF_CACHE_DIR={hf_cache_root}",
            f"HF_HOME={hf_cache_root}",
            f"HUGGINGFACE_HUB_CACHE={hf_cache_root / 'hub'}",
            f"TRANSFORMERS_CACHE={hf_cache_root / 'transformers'}",
            f"HF_DATASETS_CACHE={hf_cache_root / 'datasets'}",
        ]
    )


def _forwarded_args(argv: Sequence[str]) -> List[str]:
    forwarded: List[str] = []
    blocked_exact = {"--submit", "--no-submit", "--dry-run", "--no-dry-run"}
    blocked_prefix = ("--submit=", "--dry-run=")
    for arg in argv:
        if arg in blocked_exact:
            continue
        if arg.startswith(blocked_prefix):
            continue
        forwarded.append(arg)
    return forwarded


def _run_local(cfg: Config, launch_cmd: Sequence[str]) -> None:
    runtime_paths = load_runtime_paths(_resolve_repo_path(cfg.repo_dir, cfg.paths_config))
    ensure_runtime_dirs(runtime_paths)
    hf_cache_root = runtime_paths.hf_cache_dir
    hf_hub_cache = hf_cache_root / "hub"
    hf_transformers_cache = hf_cache_root / "transformers"
    hf_datasets_cache = hf_cache_root / "datasets"
    for p in (hf_cache_root, hf_hub_cache, hf_transformers_cache, hf_datasets_cache):
        p.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("THESIS_WANDB_ROOT_DIR", str(runtime_paths.wandb_root_dir))
    env.setdefault("THESIS_HYDRA_ROOT_DIR", str(runtime_paths.hydra_root_dir))
    env.setdefault("THESIS_RESULTS_ROOT_DIR", str(runtime_paths.results_root_dir))
    env.setdefault("THESIS_PREPROCESSING_ROOT_DIR", str(runtime_paths.preprocessing_root_dir))
    env.setdefault("THESIS_CANONICAL_GT_ROOT_DIR", str(runtime_paths.canonical_gt_root_dir))
    env.setdefault("THESIS_DIFIX_DATA_ROOT_DIR", str(cfg.data_root_dir))
    env.setdefault("THESIS_HF_CACHE_DIR", str(hf_cache_root))
    env.setdefault("HF_HOME", str(hf_cache_root))
    env.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_hub_cache))
    env.setdefault("TRANSFORMERS_CACHE", str(hf_transformers_cache))
    env.setdefault("HF_DATASETS_CACHE", str(hf_datasets_cache))
    subprocess.run(list(launch_cmd), check=True, cwd=str(cfg.repo_dir), env=env)


def _submit(cfg: Config) -> None:
    slurm_script = _resolve_repo_path(cfg.repo_dir, cfg.slurm.slurm_script)
    if not slurm_script.exists():
        raise FileNotFoundError(f"Slurm script not found: {slurm_script}")

    runtime_paths = load_runtime_paths(_resolve_repo_path(cfg.repo_dir, cfg.paths_config))
    ensure_runtime_dirs(runtime_paths)

    cmd: List[str] = [
        "sbatch",
        "--job-name",
        cfg.slurm.job_name,
        "--time",
        cfg.slurm.time,
        "--output",
        str(runtime_paths.slurm_dir / "%x.%j.out"),
        "--error",
        str(runtime_paths.slurm_dir / "%x.%j.err"),
        "--ntasks",
        "1",
        "--cpus-per-task",
        str(cfg.slurm.cpus_per_task),
        "--mem",
        cfg.slurm.mem,
        "--gres",
        f"gpu:{cfg.slurm.gpus}",
        "--account",
        cfg.slurm.account,
        "--export",
        _build_export_env(runtime_paths, cfg),
        str(slurm_script),
    ]
    if cfg.slurm.partition:
        cmd.extend(["--partition", cfg.slurm.partition])
    if cfg.slurm.qos:
        cmd.extend(["--qos", cfg.slurm.qos])

    cmd.extend(_forwarded_args(sys.argv[1:]))

    print("About to submit DiFix tuning job with:")
    print(f"  slurm_script: {slurm_script}")
    print(f"  job_name: {cfg.slurm.job_name}")
    print(f"  time: {cfg.slurm.time}")
    print(f"  gpus: {cfg.slurm.gpus}")
    print(f"  cpus_per_task: {cfg.slurm.cpus_per_task}")
    print(f"  mem: {cfg.slurm.mem}")
    if cfg.slurm.partition:
        print(f"  partition: {cfg.slurm.partition}")
    if cfg.slurm.qos:
        print(f"  qos: {cfg.slurm.qos}")

    if not _confirm_submit():
        print("Submission cancelled.")
        return

    if cfg.dry_run:
        print(" ".join(shlex.quote(part) for part in cmd))
        return

    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if res.stdout.strip():
        print(res.stdout.strip())
    if res.stderr.strip():
        print(res.stderr.strip())


def main() -> None:
    cfg = tyro.cli(Config)
    cfg = _load_resume_locked_config(cfg)
    dataset_path = _resolve_dataset_path(cfg)
    output_dir = _resolve_output_dir(cfg)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset JSON not found: {dataset_path}. "
            "Run DiFix data generation + aggregate first, or pass --dataset-path."
        )

    if cfg.submit:
        if cfg.num_processes != cfg.slurm.gpus:
            raise ValueError(
                "In submit mode, --num-processes must match --slurm.gpus "
                f"(got {cfg.num_processes} vs {cfg.slurm.gpus})."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _save_manifest(cfg, dataset_path, output_dir)
    launch_cmd = _build_accelerate_cmd(cfg, dataset_path, output_dir)
    _print_plan(cfg, dataset_path, output_dir, launch_cmd)
    print(f"  run_config_manifest: {manifest_path}")

    if cfg.submit:
        _submit(cfg)
        return

    if cfg.dry_run:
        print(" ".join(shlex.quote(part) for part in launch_cmd))
        return

    _run_local(cfg, launch_cmd)


if __name__ == "__main__":
    main()
