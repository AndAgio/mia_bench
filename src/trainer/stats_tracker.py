
# utils/epoch_stats.py
import time
from dataclasses import dataclass
from typing import Dict, Callable, Union, Sequence, Optional, List, Any, Tuple
import torch
import torch.distributed as dist
import numpy as np


Number = Union[float, int]
TensorOrSeq = Union[torch.Tensor, Sequence[torch.Tensor], Sequence[Number]]

def _dist_ready() -> bool:
    return (dist is not None) and dist.is_available() and dist.is_initialized()

def _collective_device() -> torch.device:
    # Use CUDA if available; otherwise CPU tensors for collectives.
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@dataclass
class StageSummary:
    stage: str
    metrics: Dict[str, float]
    stage_time_sec: Optional[float] = None
    batch_time_avg_sec: Optional[float] = None
    samples_per_sec_avg: Optional[float] = None
    num_batches: int = 0

@dataclass
class EpochSummary:
    epoch: int
    stages: Dict[str, StageSummary]
    epoch_time_sec: Optional[float] = None

class AverageMeter:
    """Sample-weighted running average of a scalar."""
    def __init__(self, name: str):
        self.name = name
        self.val: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: Number, n: int = 1):
        v = float(val)
        self.val = v
        self.sum += v * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)

def _to_float_scalar(x: Union[float, torch.Tensor]) -> float:
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return float(x.detach().cpu().mean().item())
    return float(x)

class _StageStats:
    """
    Per-stage meters + timing with optional DDP reduction.

    DDP hooks:
        - ddp_reduce_meters(): globally reduce sum & count across ranks (true global averages).
        - ddp_reduce_timing(op='max'|'mean'): consolidate stage_time_sec & timing meters.
    """
    def __init__(
        self,
        stage: str,
        base_metrics: Dict[str, Callable[[TensorOrSeq, TensorOrSeq], Union[float, torch.Tensor]]],
        sync_cuda: bool = False,
        sync_mps: bool = False,
    ):
        self.stage = stage
        self.base_metrics = dict(base_metrics)
        self.meters: Dict[str, AverageMeter] = {}
        self.meters["batch_time_sec"]   = AverageMeter("batch_time_sec")
        self.meters["samples_per_sec"]  = AverageMeter("samples_per_sec")
        self._stage_t0: Optional[float] = None
        self._stage_time_sec: Optional[float] = None
        self._batch_t0: Optional[float] = None
        self._batch_idx: int = 0
        self._num_batches: int = 0
        self.sync_cuda = bool(sync_cuda)
        self.sync_mps  = bool(sync_mps)

    # ---- timing ----
    def start(self) -> None:
        self._stage_t0 = time.perf_counter()
        self._stage_time_sec = None

    def end(self) -> float:
        if self._stage_t0 is None:
            return 0.0
        self._stage_time_sec = time.perf_counter() - self._stage_t0
        return self._stage_time_sec

    def batch_start(self) -> None:
        self._batch_t0 = time.perf_counter()

    def batch_end(self, batch_size: Optional[int] = None) -> float:
        # Optional device synchronization for accurate timing
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        if self.sync_mps and hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.synchronize()
            except Exception:
                pass

        if self._batch_t0 is None:
            duration = 0.0
        else:
            duration = time.perf_counter() - self._batch_t0

        sps = 0.0
        if batch_size is not None and duration > 0.0:
            sps = float(batch_size) / duration

        self.meters["batch_time_sec"].update(duration, n=1)
        self.meters["samples_per_sec"].update(sps, n=1)
        self._batch_idx += 1
        self._num_batches += 1
        return duration

    def batch_log_dict(self) -> Dict[str, float]:
        bt = self.meters["batch_time_sec"]
        sp = self.meters["samples_per_sec"]
        return {
            "stage": self.stage,
            "batch_index": self._batch_idx,
            "batch_time_sec": round(bt.val, 6),
            "batch_time_avg_sec": round(bt.avg, 6),
            "samples_per_sec": round(sp.val, 2),
            "samples_per_sec_avg": round(sp.avg, 2),
        }

    # ---- metrics ----
    def add_metric(self, name: str, fn: Callable[[TensorOrSeq, TensorOrSeq], Union[float, torch.Tensor]]) -> None:
        self.base_metrics[name] = fn

    def update(self, preds: TensorOrSeq, targets: TensorOrSeq, metrics: Optional[Dict[str, Callable]] = None, extras: Dict[str, Any] = None,) -> None:
        metric_fns = metrics if metrics is not None else self.base_metrics
        if not metric_fns:
            return
        with torch.no_grad():
            for name, fn in metric_fns.items():
                try:
                    if name in ['mace', 'mean_absolute_calibration_error', 'quantile_calibration_error']:
                        out = fn(preds, targets, extras["quantiles"])
                    else:
                        out = fn(preds, targets)
                    val = _to_float_scalar(out)
                except:
                    continue
                if name not in self.meters:
                    self.meters[name] = AverageMeter(name)
                # One unit per batch → batch-wise running average.
                self.meters[name].update(val, n=1)

    def current_avgs(self) -> Dict[str, float]:
        return {name: meter.avg for name, meter in self.meters.items()}

    def current_vals(self) -> Dict[str, float]:
        return {name: meter.val for name, meter in self.meters.items()}

    # ---- DDP reductions ----
    def ddp_reduce_meters(self) -> None:
        """
        All-reduce (SUM) the meters' sums and counts across ranks.
        After this, .avg is a true global average.
        """
        if not _dist_ready():
            return
        dev = _collective_device()
        for m in self.meters.values():
            t = torch.tensor([m.sum, m.count], dtype=torch.float64, device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            m.sum = float(t[0].item())
            m.count = int(t[1].item())

    def ddp_reduce_timing(self, op: str = "max") -> None:
        """
        Consolidate stage_time_sec across ranks via all-reduce:
            - op='max' → reflect the slowest worker (recommended).
            - op='mean' → average times across ranks (optional).
        """
        if not _dist_ready():
            return
        if self._stage_time_sec is None:
            # If not measured yet, force measurement
            self.end()
        dev = _collective_device()
        t = torch.tensor([self._stage_time_sec], dtype=torch.float64, device=dev)
        reduce_op = dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM
        dist.all_reduce(t, op=reduce_op)
        if op == "mean":
            t = t / dist.get_world_size()
        self._stage_time_sec = float(t.item())

    def finalize(self) -> StageSummary:
        if self._stage_time_sec is None:
            self.end()
        metrics_avgs = {
            name: meter.avg
            for name, meter in self.meters.items()
            if name not in ("batch_time_sec", "samples_per_sec")
        }
        bt_avg = self.meters["batch_time_sec"].avg
        sp_avg = self.meters["samples_per_sec"].avg
        return StageSummary(
            stage=self.stage,
            metrics=metrics_avgs,
            stage_time_sec=self._stage_time_sec,
            batch_time_avg_sec=bt_avg,
            samples_per_sec_avg=sp_avg,
            num_batches=self._num_batches,
        )

    def reset(self) -> None:
        for m in self.meters.values():
            m.val = 0.0
            m.sum = 0.0
            m.count = 0
        self._stage_t0 = None
        self._stage_time_sec = None
        self._batch_t0 = None
        self._batch_idx = 0
        self._num_batches = 0


class EpochStats:
    """
    Single object to track an epoch with multiple stages ("train", "val", "test"),
    now with **DDP hooks** for metric and timing consolidation.

    DDP hooks:
        - ddp_reduce_current_stage(): reduce active stage meters
        - ddp_reduce_all_stages(): reduce meters for all started stages
        - ddp_consolidate_epoch_time(op='max'|'mean'): consolidate epoch_time_sec across ranks

    Typical DDP usage per stage:
        es.stage_begin("train", metrics=..., sync_cuda=True)
        for batch:
            es.batch_start(); ...; es.update(...); es.batch_end(...)
        es.ddp_reduce_current_stage()       # <-- before stage_end
        es.stage_end()

    Or consolidate all stages at once before finalize:
        es.ddp_reduce_all_stages()
        summary = es.finalize_epoch(epoch)
    """
    def __init__(self):
        self._epoch_t0: Optional[float] = None
        self._epoch_time_sec: Optional[float] = None
        self._stages: Dict[str, _StageStats] = {}
        self._current_stage: Optional[str] = None

    def get_stage(self) -> str:
        return self._current_stage

    # ---- epoch timing ----
    def epoch_start(self) -> None:
        self._epoch_t0 = time.perf_counter()
        self._epoch_time_sec = None

    def epoch_end(self) -> float:
        if self._epoch_t0 is None:
            return 0.0
        self._epoch_time_sec = time.perf_counter() - self._epoch_t0
        return self._epoch_time_sec
    
    def get_epoch_time(self) -> float:
        return self._epoch_time_sec
    
    def get_current_running_time(self, op: str = "max") -> float:
        if not _dist_ready():
            return time.perf_counter() - self._epoch_t0
        dev = _collective_device()
        t = torch.tensor([time.perf_counter() - self._epoch_t0], dtype=torch.float64, device=dev)
        reduce_op = dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM
        dist.all_reduce(t, op=reduce_op)
        if op == "mean":
            t = t / dist.get_world_size()
        return float(t.item())

    def ddp_consolidate_epoch_time(self, op: str = "max") -> None:
        """
        Consolidate epoch_time_sec across ranks.
        - 'max' reflects slowest worker (recommended).
        - 'mean' averages across ranks.
        """
        if not _dist_ready():
            return
        if self._epoch_time_sec is None:
            self.epoch_end()
        dev = _collective_device()
        t = torch.tensor([self._epoch_time_sec], dtype=torch.float64, device=dev)
        reduce_op = dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM
        dist.all_reduce(t, op=reduce_op)
        if op == "mean":
            t = t / dist.get_world_size()
        self._epoch_time_sec = float(t.item())

    # ---- stages ----
    def stage_begin(
        self,
        stage: str,
        metrics: Optional[Dict[str, Callable[[TensorOrSeq, TensorOrSeq], Union[float, torch.Tensor]]]] = None,
        sync_cuda: bool = False,
        sync_mps: bool = False,
    ) -> None:
        if stage not in self._stages:
            self._stages[stage] = _StageStats(stage, base_metrics=metrics or {}, sync_cuda=sync_cuda, sync_mps=sync_mps)
        else:
            if metrics is not None:
                for name, fn in metrics.items():
                    self._stages[stage].add_metric(name, fn)
        self._current_stage = stage
        self._stages[stage].start()

    def stage_end(self) -> StageSummary:
        if self._current_stage is None:
            raise RuntimeError("No active stage to end. Call stage_begin(...) first.")
        st = self._stages[self._current_stage]
        summary = st.finalize()
        self._current_stage = None
        return summary

    def _require_active_stage(self) -> _StageStats:
        if self._current_stage is None:
            raise RuntimeError("No active stage. Call stage_begin(...) before batch/metrics operations.")
        return self._stages[self._current_stage]

    def batch_start(self) -> None:
        self._require_active_stage().batch_start()

    def batch_end(self, batch_size: Optional[int] = None) -> float:
        return self._require_active_stage().batch_end(batch_size=batch_size)

    def update(
        self,
        preds: TensorOrSeq,
        targets: TensorOrSeq,
        metrics: Optional[Dict[str, Callable[[TensorOrSeq, TensorOrSeq], Union[float, torch.Tensor]]]] = None,
        extras: Dict[str, Any] = None,
    ) -> None:
        self._require_active_stage().update(preds, targets, metrics=metrics, extras=extras)

    def batch_log_dict(self) -> Dict[str, float]:
        return self._require_active_stage().batch_log_dict()

    # ---- DDP hooks ----
    def ddp_reduce_current_stage(self) -> None:
        if self._current_stage is None:
            return
        self._stages[self._current_stage].ddp_reduce_meters()
        self._stages[self._current_stage].ddp_reduce_timing(op="max")

    def ddp_reduce_all_stages(self) -> None:
        if not _dist_ready():
            return
        for st in self._stages.values():
            st.ddp_reduce_meters()
            st.ddp_reduce_timing(op="max")

    # ---- finalize ----
    def finalize_epoch(self, epoch: int) -> EpochSummary:
        if self._epoch_time_sec is None:
            self.epoch_end()
        if self._current_stage is not None:
            self.stage_end()
        stage_summaries: Dict[str, StageSummary] = {}
        for name, st in self._stages.items():
            stage_summaries[name] = st.finalize()
        summary = EpochSummary(epoch=epoch, stages=stage_summaries, epoch_time_sec=self._epoch_time_sec)
        self.reset()
        return summary

    def reset(self) -> None:
        self._epoch_t0 = None
        self._epoch_time_sec = None
        self._current_stage = None
        for st in self._stages.values():
            st.reset()
        self._stages.clear()

    def elapsed_epoch_time(self) -> float:
        return time.perf_counter() - self._epoch_t0


class TrainStats:
    """
    Across-epoch tracker with DDP-friendly usage:

    Tracks
    ------
    - current_epoch: int
    - total training time (pause/resume safe across runs)
    - explicit history: Dict[int, EpochSummary]
    - best_records: per-stage/per-metric best

    DDP conveniences
    ----------------
    - is_rank0(), ddp_barrier()
    - Pure-local storage; reduce metrics in EpochStats before add_epoch_summary for global stats.
    """
    def __init__(self, stage_to_track_best: str = 'train', metric_to_track_best: str = 'loss', best_record: float = np.inf, best_epoch: int = 0, mode_to_track_best: str = 'min') -> None:
        self.current_epoch: int = 0
        self.history: Dict[int, EpochSummary] = {}
        assert stage_to_track_best in ("train", "val", "test"), f"stage to track best metric must be 'train', 'val' or 'test'"
        self.stage_to_track_best = stage_to_track_best
        self.metric_to_track_best = metric_to_track_best
        assert mode_to_track_best in ("min", "max"), f"mode to track best metric must be 'min' or 'max'"
        self.mode_to_track_best = mode_to_track_best
        self.best_record = best_record
        self.best_epoch = best_epoch
        # self.best_records: Dict[str, Dict[str, Any]] = {}
        self._timer_sec: float = 0.0
        self._timer_t0: Optional[float] = None

    # ----- total timer -----
    def start_timer(self) -> None:
        self._timer_t0 = time.perf_counter()

    def update_timer(self) -> float:
        self._timer_sec += (time.perf_counter() - self._timer_t0)
        return self._timer_sec

    def restore_timer(self, total_seconds: float) -> None:
        self._timer_sec = float(total_seconds)

    def total_time(self) -> float:
        return self._timer_sec
    
    def get_current_running_time(self, op: str = "max") -> float:
        if not _dist_ready():
            return time.perf_counter() - self._timer_t0
        dev = _collective_device()
        t = torch.tensor([time.perf_counter() - self._timer_t0], dtype=torch.float64, device=dev)
        reduce_op = dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM
        dist.all_reduce(t, op=reduce_op)
        if op == "mean":
            t = t / dist.get_world_size()
        return float(t.item())

    # ----- history & best -----
    def add_epoch_summary(self, summary: EpochSummary) -> None:
        self.history[summary.epoch] = summary
        self.current_epoch = summary.epoch

    def get_epoch_summary(self, epoch: int) -> Optional[EpochSummary]:
        return self.history.get(epoch)

    def last_epoch_summary(self) -> Optional[EpochSummary]:
        if not self.history:
            return None
        last_epoch = max(self.history.keys())
        return self.history[last_epoch]

    def last_stage_summary(self, stage: str) -> Optional[StageSummary]:
        last = self.last_epoch_summary()
        if last is None:
            return None
        return last.stages.get(stage)
    
    def get_best_score(self) -> float:
        return self.best_record
    
    def get_best_epoch(self) -> int:
        return self.best_epoch
    
    def get_best(self) -> Tuple[int, float]:
        return self.best_epoch, self.best_record
    
    def update_best_if_better(self):
        last = self.last_epoch_summary()
        last_epoch = last.epoch
        last_stage = last.stages.get(self.stage_to_track_best)
        last_value = last_stage.metrics.get(self.metric_to_track_best)
        cmp = (lambda a, b: a < b) if self.mode_to_track_best == "min" else (lambda a, b: a > b)
        if cmp(last_value, self.best_record):
            self.best_record = last_value
            self.best_epoch = last_epoch

    def update_history_and_best(self, epoch_summary: EpochSummary) -> Tuple[int, float]:
        self.update_timer()
        self.add_epoch_summary(epoch_summary)
        self.update_best_if_better()
        return self.get_best()

    # ----- serialization helpers -----
    def to_checkpoint_dict(self) -> Dict[str, Any]:
        flat_history: List[Dict[str, Any]] = []
        for ep, es in self.history.items():
            for stage_name, st in es.stages.items():
                row = {
                    "epoch": ep,
                    "stage": stage_name,
                    "epoch_time_sec": es.epoch_time_sec if es.epoch_time_sec is not None else None,
                    "stage_time_sec": st.stage_time_sec if st.stage_time_sec is not None else None,
                    "batch_time_avg_sec": st.batch_time_avg_sec if st.batch_time_avg_sec is not None else None,
                    "samples_per_sec_avg": st.samples_per_sec_avg if st.samples_per_sec_avg is not None else None,
                    "num_batches": st.num_batches,
                }
                for k, v in st.metrics.items():
                    row[f"metric:{k}"] = float(v)
                flat_history.append(row)
        return {
            "current_epoch": int(self.current_epoch),
            "train_total_time_sec": float(self.total_time()),
            # "best_records": dict(self.best_records),
            "stage_to_track_best": self.stage_to_track_best,
            "metric_to_track_best": self.metric_to_track_best, 
            "best_record": self.best_record,
            "best_epoch": self.best_epoch, 
            "mode_to_track_best": self.mode_to_track_best,
            "history": flat_history,
        }

    def load_from_checkpoint_dict(self, payload: Dict[str, Any]) -> None:
        self.history.clear()
        # self.best_records = dict(payload.get("best_records", {}))

        self.stage_to_track_best = str(payload.get("stage_to_track_best", 'train'))
        self.metric_to_track_best = str(payload.get("metric_to_track_best", 'loss'))
        self.best_record = float(payload.get("best_record", np.inf))
        self.best_epoch = int(payload.get("best_epoch", 0))
        self.mode_to_track_best = str(payload.get("mode_to_track_best", 'min'))

        self.current_epoch = int(payload.get("current_epoch", 0))

        by_epoch: Dict[int, Dict[str, StageSummary]] = {}
        epoch_time: Dict[int, Optional[float]] = {}
        for row in payload.get("history", []):
            ep = int(row["epoch"])
            stage_name = str(row["stage"])
            st_metrics: Dict[str, float] = {}
            for k, v in row.items():
                if isinstance(k, str) and k.startswith("metric:"):
                    st_metrics[k.split("metric:", 1)[1]] = float(v)
            st_summary = StageSummary(
                stage=stage_name,
                metrics=st_metrics,
                stage_time_sec=row.get("stage_time_sec"),
                batch_time_avg_sec=row.get("batch_time_avg_sec"),
                samples_per_sec_avg=row.get("samples_per_sec_avg"),
                num_batches=int(row.get("num_batches", 0)),
            )
            by_epoch.setdefault(ep, {})[stage_name] = st_summary
            epoch_time.setdefault(ep, row.get("epoch_time_sec"))

        for ep, stages in by_epoch.items():
            es = EpochSummary(epoch=ep, stages=stages, epoch_time_sec=epoch_time.get(ep))
            self.history[ep] = es

        self.restore_timer(float(payload.get("train_total_time_sec", 0.0)))
