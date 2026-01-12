import os
import torch
from typing import Tuple
try:
    import torch.distributed as dist
except Exception:
    dist = None


# TODO: Restructure everything into a single class.
# Issue URL: https://github.com/AndAgio/mia_bench/issues/8
# assignees: AndAgio

# ---- DDP setup/teardown ----
def get_ddp_info() -> Tuple[bool, int, int]:
    try:
        if dist.is_available() and dist.is_initialized():
            return True, dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return False, 0, 1


def maybe_init_ddp(use_ddp: bool, device: str = 'cpu', backend: str = "nccl"):
    """
    Initialize DDP only if requested and not already initialized.
    Returns: (rank, world_size, local_rank, device)
    """
    if use_ddp and not ddp_ready() and device != 'cpu':
        dist.init_process_group(backend=backend)
    is_available, rank, world = get_ddp_info()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available() and device != 'cpu':
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    elif hasattr(torch, "mps") and torch.backends.mps.is_available() and device != 'cpu':
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return rank, world, local_rank, device


def maybe_cleanup_ddp():
    if ddp_ready():
        dist.destroy_process_group()


def ddp_ready() -> bool:
    return (dist is not None) and dist.is_available() and dist.is_initialized()


def is_rank0() -> bool:
    return (not ddp_ready()) or dist.get_rank() == 0


def ddp_barrier() -> None:
    if ddp_ready():
        dist.barrier()