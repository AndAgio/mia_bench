
# utils/checkpoint_manager_ddp.py
from __future__ import annotations
import os
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import torch
import numpy as np
from src.trainer.stats_tracker import TrainStats
from src.trainer.distributed import get_ddp_info
import copy



@dataclass
class BestRecord:
    epoch: int
    value: float
    stage: str
    metric: str
    mode: str  # "min" or "max"


class CheckpointManager:
    """
    DDP-safe Checkpoint manager with epoch-indexed files, TrainStats integration, RNG persistence.

    --- differences vs your original ---
    - Atomic writes: save to tmp file and os.replace(...) to avoid partial files.
    - DDP barriers: synchronize ranks before/after save and during resume.
    - Unwrap model under DDP (model.module) for clean state_dict.
    - Key prefix reconciliation when loading ("module." vs non-DDP keys).
    - Broadcast chosen resume checkpoint path from save_on_rank to all ranks.

    API is unchanged.
    """

    CKPT_REGEX = re.compile(r"^epoch_(\d+)\.pth$")
    MODEL_REGEX = re.compile(r"^model_epoch_(\d+)\.pth$")

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_dir: Path,
        scheduler: Optional[Any] = None,
        logger: Optional[Any] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.map_location = 'cpu'

        # DDP info
        self._ddp_active, self._rank, self._world = get_ddp_info()

    # ---------------------- helpers: DDP & RNG ----------------------
    def _barrier(self) -> None:
        if self._ddp_active:
            import torch.distributed as dist
            dist.barrier()

    def _is_master(self) -> bool:
        """Return True if this process may write checkpoints (DDP rank guard)."""
        if not self._ddp_active:
            return True
        return self._rank == 0

    def _unwrap_model(self) -> torch.nn.Module:
        """Return the actual nn.Module under DDP/FSDP wrappers."""
        return getattr(self.model, "module", self.model)

    @staticmethod
    def _capture_rng_state() -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),  # CPU default generator
        }
        if np is not None:
            state["numpy"] = np.random.get_state()
        if torch.cuda.is_available():
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
            state["torch_cuda_active_device"] = torch.cuda.current_device()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                state["torch_mps"] = torch.mps.get_rng_state()
            except Exception:
                state["torch_mps"] = None
        return state

    @staticmethod
    def _restore_rng_state(state: Dict[str, Any]) -> None:
        if "python" in state and state["python"] is not None:
            random.setstate(state["python"])
        if np is not None and "numpy" in state and state["numpy"] is not None:
            np.random.set_state(state["numpy"])
        if "torch_cpu" in state and state["torch_cpu"] is not None:
            torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available():
            if "torch_cuda_all" in state and state["torch_cuda_all"] is not None:
                try:
                    torch.cuda.set_rng_state_all(state["torch_cuda_all"])
                except Exception:
                    if "torch_cuda_active_device" in state and state["torch_cuda_active_device"] is not None:
                        dev_idx = int(state["torch_cuda_active_device"])
                        torch.cuda.set_rng_state(state["torch_cuda_all"][dev_idx], device=dev_idx)
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            if "torch_mps" in state and state["torch_mps"] is not None:
                try:
                    torch.mps.set_rng_state(state["torch_mps"])
                except Exception:
                    pass

    # ---------------------- atomic save ----------------------
    def _atomic_save(self, obj: Any, path: Path) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, tmp)
        os.replace(tmp, path)  # atomic on POSIX

    # ---------------------- state_dict key reconciliation ----------------------
    def _load_model_state(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        """
        Load state_dict into the (unwrapped) model.
        If keys use 'module.' prefix (or not) and current model differs, reconcile keys.
        """
        module = self._unwrap_model()
        try:
            module.load_state_dict(state_dict, strict=strict)
            return
        except RuntimeError as e:
            # Try to reconcile 'module.' prefix differences
            cur_keys = list(module.state_dict().keys())
            has_module_prefix_now = cur_keys and cur_keys[0].startswith("module.")
            loaded_has_module_prefix = next(iter(state_dict)).startswith("module.")

            if loaded_has_module_prefix and not has_module_prefix_now:
                fixed = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
            elif not loaded_has_module_prefix and has_module_prefix_now:
                fixed = {("module." + k): v for k, v in state_dict.items()}
            else:
                raise e  # Different mismatch; let the original error bubble up

            module.load_state_dict(fixed, strict=False)

    # ---------------------- checkpoint payload ----------------------
    def build_checkpoint(
        self,
        epoch: int,
        train_stats: TrainStats,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        ckpt: Dict[str, Any] = {
            "epoch": int(epoch),
            "model_state": self._unwrap_model().state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "train_stats": train_stats.to_checkpoint_dict(),
            "extra": extra or {},
        }
        if self.scheduler is not None:
            ckpt["scheduler_state"] = self.scheduler.state_dict()
        ckpt["rng"] = self._capture_rng_state()
        return ckpt

    # ---------------------- save APIs ----------------------
    def save(self,
            name: str,
            ckpt: Dict[str, Any],
            ):
        self._barrier()
        ckpt_path = self.checkpoint_dir/name
        if self._is_master():
            self._atomic_save(ckpt, ckpt_path)
            self.logger.print_it(f"[CheckpointManager] Saved checkpoint at {ckpt_path}")
        self._barrier()

    # ---------------------- load/apply ----------------------
    def load(self, path: Path) -> Dict[str, Any]:
        ckpt = torch.load(path, weights_only=False, map_location=self.map_location)
        if self.logger:
            self.logger.print_it(f"[CheckpointManager] Loaded checkpoint: {path}")
        return ckpt

    def apply_checkpoint(self, ckpt: Dict[str, Any]) -> None:
        self._load_model_state(ckpt["model_state"], strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if self.scheduler is not None and "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])

    def restore_rng(self, ckpt: Dict[str, Any]) -> None:
        rng = ckpt.get("rng")
        if rng is not None:
            self._restore_rng_state(rng)
            if self.logger:
                self.logger.print_it("[CheckpointManager] RNG state restored.")

    # ---------------------- latest-epoch discovery ----------------------
    def latest_epoch_checkpoint(self) -> Optional[Tuple[int, Path]]:
        items: List[Tuple[int, Path]] = []
        if not self.checkpoint_dir.exists():
            return None
        latest_ckpt = self.checkpoint_dir/'last.pth'
        if latest_ckpt.exists():
            epoch = torch.load(latest_ckpt, weights_only=False)["epoch"]
            self.logger.print_it(f'[CheckpointManager] Found "last.pth" checkpoint for epoch {epoch}, loading it...')
            return (epoch, latest_ckpt)
        self.logger.print_it('[CheckpointManager] No "last.pth" checkpoint found, trying to look for epochs_N.pth files...')
        for p in self.checkpoint_dir.iterdir():
            if p.is_file():
                m = self.CKPT_REGEX.match(p.name)
                if m:
                    try:
                        ep = int(m.group(1))
                        items.append((ep, p))
                    except Exception:
                        continue
        items.sort(key=lambda t: t[0])  # ascending by epoch
        if len(items) > 0:
            self.logger.print_it(f'[CheckpointManager] Found checkpoint at epoch {items[-1][0]} with file "{items[-1][1]}", loading it...')
            items[-1]
        else:
            self.logger.print_it('[CheckpointManager] No epochs_N.pth checkpoint files found. Loading nothing. Make sure that a checkpoint is available...')
            return None
        

    # ---------------------- broadcast helpers ----------------------
    def _broadcast_path_from_master(self, chosen_path: Optional[Path]) -> Optional[Path]:
        """
        In DDP, broadcast the string path from save_on_rank to all ranks.
        For non-DDP, return chosen_path unchanged.
        """
        if not self._ddp_active:
            return chosen_path

        import torch.distributed as dist
        obj_list = [str(chosen_path) if chosen_path is not None else ""]
        dist.broadcast_object_list(obj_list, src=self.save_on_rank if self.save_on_rank >= 0 else 0)
        s = obj_list[0]
        return Path(s) if s else None

    # ---------------------- resume ----------------------
    def resume(
        self,
        path: Optional[Path] = None,
        latest: bool = True,
    ) -> Dict[str, Any]:
        """
        Resume training consistently across all ranks:
        - master rank decides checkpoint (explicit path or auto-latest),
        - path is broadcast to all ranks,
        - all ranks load/apply, restore RNG,
        - TrainStats rehydrated and total timer resumed.
        Returns: {'start_epoch', 'best', 'path'}
        """
        chosen_path: Optional[Path] = None

        # Master decides the path
        if self._is_master():
            if path is not None:
                chosen_path = Path(path)
            elif latest:
                latest_info = self.latest_epoch_checkpoint()
                if latest_info is not None:
                    _, chosen_path = latest_info

        # Broadcast the decision to all ranks
        chosen_path = self._broadcast_path_from_master(chosen_path)
        self._barrier()

        if chosen_path is None or not Path(chosen_path).exists():
            if self.logger and self._is_master():
                self.logger.print_it("[CheckpointManager] No checkpoint found; starting fresh.")
            return None

        ckpt = self.load(chosen_path)
        self.apply_checkpoint(ckpt)
        self.restore_rng(ckpt)

        self._barrier()
        return ckpt

    def load_best_model(self) -> torch.nn.Module:
        ckpt_path = self.checkpoint_dir/'best.pth'
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=self.map_location)
        self._load_model_state(state_dict=ckpt.get("model_state"))
        best_model = copy.deepcopy(self.model)
        latest_ckpt_path = self.checkpoint_dir/'last.pth'
        ckpt = torch.load(latest_ckpt_path, weights_only=False, map_location=self.map_location)
        self._load_model_state(state_dict=ckpt.get("model_state"))
        return best_model
