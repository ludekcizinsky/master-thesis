from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch

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

        print(f"--- FYI: checkpoint root dir: {self.root}")

    def save(self, scene_splats: SceneSplats, iteration: int) -> None:
        """Persist current gaussian parameters to disk."""
        if scene_splats.static is not None:
            self._save_param_dict(
                target_dir=self.static_dir,
                params=scene_splats.static,
                iteration=iteration,
                payload_type="static",
                tid=None,
            )

        if len(scene_splats.dynamic) != len(self.tids):
            raise ValueError(
                f"Mismatch between configured tids ({self.tids}) and "
                f"dynamic gaussian sets ({len(scene_splats.dynamic)})."
            )

        for tid, splats in zip(self.tids, scene_splats.dynamic):
            self._save_param_dict(
                target_dir=self.human_dirs[tid],
                params=splats,
                iteration=iteration,
                payload_type="human",
                tid=tid,
            )

    @staticmethod
    def _save_param_dict(
        *,
        target_dir: Path,
        params: torch.nn.ParameterDict,
        iteration: int,
        payload_type: str,
        tid: Optional[int],
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"iter_{iteration:06d}.pt"
        file_path = target_dir / file_name

        payload = {
            "iteration": iteration,
            "type": payload_type,
            "params": {k: v.detach().cpu() for k, v in params.items()},
        }
        if tid is not None:
            payload["tid"] = tid

        torch.save(payload, file_path)

        latest_path = target_dir / "latest.txt"
        latest_path.write_text(file_name)
