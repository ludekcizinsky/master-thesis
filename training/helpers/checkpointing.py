from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn

from training.helpers.model_init import SceneSplats


class ModelCheckpointManager:
    """Handles persistence of static and human 3DGS parameter sets as well as SMPL parameters."""

    def __init__(
        self,
        scene_output_dir: Path,
        group_name: str,
        tids: Iterable[int],
    ):
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
        self.smpl_dir = self.root / "smpl"
        self.smpl_dir.mkdir(parents=True, exist_ok=True)

        self.base_epochs: dict = {"static": 0}
        for tid in self.tids:
            self.base_epochs[("human", tid)] = 0
        self.smpl_base_epoch: int = 0

        print(f"--- FYI: checkpoint root dir: {self.root}")

    def save(
        self,
        scene_splats: SceneSplats,
        epoch: int,
        *,
        smpl_params: Optional[Dict[int, nn.Parameter]] = None,
    ) -> None:
        """Persist current gaussian parameters to disk."""
        if scene_splats.static is not None:
            base_epoch = int(self.base_epochs.get("static", 0))
            total_epoch = int(epoch) + base_epoch
            self._save_param_dict(
                target_dir=self.static_dir,
                params=scene_splats.static,
                epoch=total_epoch,
                payload_type="static",
                tid=None,
                base_epoch=base_epoch,
                session_epoch=epoch,
            )

        if len(smpl_params) > 0:
            base_epoch = int(self.smpl_base_epoch)
            total_epoch = int(epoch) + base_epoch
            self._save_smpl_params(
                params=smpl_params,
                epoch=total_epoch,
            )

        if len(scene_splats.dynamic) != len(self.tids):
            raise ValueError(
                f"Mismatch between configured tids ({self.tids}) and "
                f"dynamic gaussian sets ({len(scene_splats.dynamic)})."
            )

        for tid, splats in zip(self.tids, scene_splats.dynamic):
            base_epoch = int(self.base_epochs.get(("human", tid), 0))
            total_epoch = int(epoch) + base_epoch
            self._save_param_dict(
                target_dir=self.human_dirs[tid],
                params=splats,
                epoch=total_epoch,
                payload_type="human",
                tid=tid,
                base_epoch=base_epoch,
                session_epoch=epoch,
            )

    @staticmethod
    def _save_param_dict(
        *,
        target_dir: Path,
        params: torch.nn.ParameterDict,
        epoch: int,
        payload_type: str,
        tid: Optional[int],
        base_epoch: int,
        session_epoch: int,
    ) -> None:
        target_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"epoch_{epoch:06d}.pt"
        file_path = target_dir / file_name

        payload = {
            "epoch": epoch,
            "type": payload_type,
            "params": {k: v.detach().cpu() for k, v in params.items()},
            "base_epoch": base_epoch,
            "session_epoch": session_epoch,
        }
        if tid is not None:
            payload["tid"] = tid

        torch.save(payload, file_path)

        latest_path = target_dir / "latest.txt"
        latest_path.write_text(file_name)

    def load_static(
        self, device: torch.device | str
    ) -> Tuple[Optional[nn.ParameterDict], Optional[int]]:
        params, epoch = self._load_param_dict_from_dir(self.static_dir, device)
        self.base_epochs["static"] = int(epoch) if epoch is not None else 0
        return params, epoch

    def load_human(
        self, tid: int, device: torch.device | str
    ) -> Tuple[Optional[nn.ParameterDict], Optional[int]]:
        directory = self.human_dirs.get(tid, self.root / f"human_{tid}")
        directory.mkdir(parents=True, exist_ok=True)
        self.human_dirs[tid] = directory
        params, epoch = self._load_param_dict_from_dir(directory, device)
        self.base_epochs[("human", tid)] = int(epoch) if epoch is not None else 0
        return params, epoch

    def load_smpl(
        self, device: torch.device | str
    ) -> Tuple[Optional[Dict[int, nn.Parameter]], Optional[int]]:
        params, epoch = self._load_smpl_params(device=device)
        self.smpl_base_epoch = int(epoch) if epoch is not None else 0
        return params, epoch

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

        epoch = payload.get("epoch")
        if epoch is not None:
            epoch = int(epoch)
        return param_dict, epoch

    def _load_smpl_params(
        self, device: torch.device | str
    ) -> Tuple[Optional[Dict[int, nn.Parameter]], Optional[int]]:
        ckpt_path = self._latest_checkpoint_path(self.smpl_dir)
        if ckpt_path is None:
            return None, None

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        params = payload.get("params")
        if params is None:
            raise ValueError(f"Checkpoint {ckpt_path} missing 'params' entry.")

        smpl_params: Dict[int, nn.Parameter] = {}
        for fid, tensor in params.items():
            restored = nn.Parameter(tensor.to(device=device).contiguous())
            restored.requires_grad_(True)
            smpl_params[int(fid)] = restored

        epoch = payload.get("epoch")
        if epoch is not None:
            epoch = int(epoch)
        return smpl_params, epoch

    def _save_smpl_params(
        self,
        *,
        params: Dict[int, nn.Parameter],
        epoch: int,
    ) -> None:
        self.smpl_dir.mkdir(parents=True, exist_ok=True)
        file_name = f"epoch_{epoch:06d}.pt"
        file_path = self.smpl_dir / file_name

        payload = {
            "epoch": epoch,
            "params": {int(fid): tensor.detach().cpu() for fid, tensor in params.items()},
        }

        torch.save(payload, file_path)

        latest_path = self.smpl_dir / "latest.txt"
        latest_path.write_text(file_name)

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

        epoch_candidates = sorted(target_dir.glob("epoch_*.pt"))
        if epoch_candidates:
            return epoch_candidates[-1]
        return None

    @staticmethod
    def _clear_directory(target_dir: Path) -> None:
        if not target_dir.exists():
            return
        for item in target_dir.glob("*"):
            if item.is_file():
                item.unlink()

    def reset(self, reset_static: bool, reset_tids: Iterable[int]) -> None:
        if reset_static:
            self._clear_directory(self.static_dir)
            self.base_epochs["static"] = 0

        for tid in reset_tids:
            directory = self.human_dirs.get(tid, self.root / f"human_{tid}")
            directory.mkdir(parents=True, exist_ok=True)
            self._clear_directory(directory)
            self.base_epochs[("human", tid)] = 0

        self._clear_directory(self.smpl_dir)
        self.smpl_base_epoch = 0
