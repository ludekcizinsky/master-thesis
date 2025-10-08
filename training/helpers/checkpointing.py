from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from training.helpers.model_init import SceneSplats


class GaussianCheckpointManager:
    """Handles persistence of static and human 3DGS parameter sets."""

    def __init__(self, scene_output_dir: Path, group_name: str, tids: Iterable[int]):
        self.scene_output_dir = Path(scene_output_dir)
        self.group_name = group_name
        self.root = self.scene_output_dir / "checkpoints" / group_name
        self.static_dir = self.root / "static"
        self.tids = list(tids)
        self.human_dirs = {
            tid: self.root / f"human_{tid}"
            for tid in self.tids
        }

        self.static_dir.mkdir(parents=True, exist_ok=True)
        for directory in self.human_dirs.values():
            directory.mkdir(parents=True, exist_ok=True)

        self.base_iterations: dict = {"static": 0}
        for tid in self.tids:
            self.base_iterations[("human", tid)] = 0

        print(f"--- FYI: checkpoint root dir: {self.root}")

    def save(self, scene_splats: SceneSplats, iteration: int) -> None:
        """Persist current gaussian parameters to disk."""
        if scene_splats.static is not None:
            base_iter = int(self.base_iterations.get("static", 0))
            total_iter = int(iteration) + base_iter
            self._save_param_dict(
                target_dir=self.static_dir,
                params=scene_splats.static,
                iteration=total_iter,
                payload_type="static",
                tid=None,
                base_iteration=base_iter,
                session_iteration=iteration,
            )

        if len(scene_splats.dynamic) != len(self.tids):
            raise ValueError(
                f"Mismatch between configured tids ({self.tids}) and "
                f"dynamic gaussian sets ({len(scene_splats.dynamic)})."
            )

        for tid, splats in zip(self.tids, scene_splats.dynamic):
            base_iter = int(self.base_iterations.get(("human", tid), 0))
            total_iter = int(iteration) + base_iter
            self._save_param_dict(
                target_dir=self.human_dirs[tid],
                params=splats,
                iteration=total_iter,
                payload_type="human",
                tid=tid,
                base_iteration=base_iter,
                session_iteration=iteration,
            )

    @staticmethod
    def _save_param_dict(
        *,
        target_dir: Path,
        params: torch.nn.ParameterDict,
        iteration: int,
        payload_type: str,
        tid: Optional[int],
        base_iteration: int,
        session_iteration: int,
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"iter_{iteration:06d}.pt"
        file_path = target_dir / file_name

        payload = {
            "iteration": iteration,
            "type": payload_type,
            "params": {k: v.detach().cpu() for k, v in params.items()},
            "base_iteration": base_iteration,
            "session_iteration": session_iteration,
        }
        if tid is not None:
            payload["tid"] = tid

        torch.save(payload, file_path)

        latest_path = target_dir / "latest.txt"
        latest_path.write_text(file_name)

    def load_static(
        self, device: torch.device | str
    ) -> Tuple[Optional[nn.ParameterDict], Optional[int]]:
        params, iteration = self._load_param_dict_from_dir(self.static_dir, device)
        self.base_iterations["static"] = int(iteration) if iteration is not None else 0
        return params, iteration

    def load_human(
        self, tid: int, device: torch.device | str
    ) -> Tuple[Optional[nn.ParameterDict], Optional[int]]:
        directory = self.human_dirs.get(tid, self.root / f"human_{tid}")
        directory.mkdir(parents=True, exist_ok=True)
        self.human_dirs[tid] = directory
        params, iteration = self._load_param_dict_from_dir(directory, device)
        self.base_iterations[("human", tid)] = int(iteration) if iteration is not None else 0
        return params, iteration

    def _load_param_dict_from_dir(
        self, target_dir: Path, device: torch.device | str
    ) -> Tuple[Optional[nn.ParameterDict], Optional[int]]:
        ckpt_path = self._latest_checkpoint_path(target_dir)
        if ckpt_path is None:
            return None, None

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        params = payload.get("params", None)
        if params is None:
            raise ValueError(f"Checkpoint {ckpt_path} missing 'params' entry.")

        param_dict = nn.ParameterDict({
            name: nn.Parameter(tensor.to(device=device).contiguous())
            for name, tensor in params.items()
        })

        iteration = payload.get("iteration")
        if iteration is not None:
            iteration = int(iteration)

        return param_dict, iteration

    @staticmethod
    def _latest_checkpoint_path(target_dir: Path) -> Optional[Path]:
        if not target_dir.exists():
            return None

        latest_file = target_dir / "latest.txt"
        if latest_file.exists():
            name = latest_file.read_text().strip()
            ckpt_path = target_dir / name
            if ckpt_path.exists():
                return ckpt_path

        candidates = sorted(target_dir.glob("iter_*.pt"))
        if not candidates:
            return None
        return candidates[-1]
