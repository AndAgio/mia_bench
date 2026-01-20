
# utils/metric_fns.py
import torch
from typing import Sequence, Union

TensorOrSeq = Union[torch.Tensor, Sequence[torch.Tensor]]


def multiclass_accuracy_from_logits(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """preds: [B, C] logits; targets: [B] class ids"""
    pred_cls = preds.argmax(dim=1)
    return float((pred_cls == targets.view(-1)).float().mean().item())


def binary_accuracy_from_logits(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """preds: [B] or [B,1] logits; targets: [B] in {0,1}"""
    probs = torch.sigmoid(preds.view(-1).float())
    pred_bin = (probs >= threshold).long()
    return float((pred_bin == targets.view(-1).long()).float().mean().item())


def mse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    diff = preds.view(-1).float() - targets.view(-1).float()
    return float((diff ** 2).mean().item())


def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    diff = preds.view(-1).float() - targets.view(-1).float()
    return float(diff.abs().mean().item())


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    diff = preds.view(-1).float() - targets.view(-1).float()
    return float(torch.sqrt((diff ** 2).mean()).item())


def r2_batch(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Batch R² computed on this batch only (mean-of-batch R² when averaged across batches).
    Note: this is not identical to global R² over the whole epoch unless computed jointly.
    """
    y = targets.view(-1).float()
    yhat = preds.view(-1).float()
    ss_res = torch.sum((y - yhat) ** 2)
    ss_tot = torch.sum((y - y.mean()) ** 2)
    if ss_tot.item() <= 1e-12:
        return 0.0
    return float(1.0 - (ss_res / ss_tot).item())


def quantile_coverage(y_true: torch.Tensor,
                    y_pred: torch.Tensor,):
    """
    y_true: (N,)
    y_pred: (N, T)
    taus: list/tensor of length T (ascending)
    Returns coverage per τ and calibration error per τ (coverage - τ).
    """
    covered = (y_true.unsqueeze(-1) <= y_pred).float()
    coverage = covered.mean(dim=0)
    return coverage.cpu()


def quantile_calibration_error(y_true: torch.Tensor,
                    y_pred: torch.Tensor,
                    quantile: torch.Tensor):
    """
    y_true: (N,)
    y_pred: (N, T)
    taus: list/tensor of length T (ascending)
    Returns coverage per τ and calibration error per τ (coverage - τ).
    """
    covered = (y_true.unsqueeze(-1) <= y_pred).float()
    coverage = covered.mean(dim=0)
    calib_error = coverage - quantile
    return calib_error.cpu().item()


def mean_absolute_calibration_error(y_true: torch.Tensor,
                                y_pred: torch.Tensor,
                                quantiles: torch.Tensor):
    """
    MACE = mean over τ of |coverage(τ) - τ|
    coverage: (T,) tensor on CPU or device
    taus: list/tensor length T
    Returns a Python float.
    """
    coverage = quantile_coverage(y_true=y_true,
                                y_pred=y_pred)
    mace = torch.abs(coverage - quantiles.cpu()).mean()
    return mace.item()



def get_performance_metric_func(metric_name: str):
    if metric_name in ['multiclass_accuracy', 'multi_class_accuracy', 'multiclass_acc', 'multi_acc', 'accuracy', 'acc']:
        return multiclass_accuracy_from_logits
    elif metric_name in ['binary_accuracy', 'binary_acc', 'bin_acc']:
        return binary_accuracy_from_logits
    elif metric_name in ['mse']:
        return mse
    elif metric_name in ['mae']:
        return mae
    elif metric_name in ['rmse']:
        return rmse
    elif metric_name in ['r2']:
        return r2_batch
    elif metric_name in ['quantile_coverage']:
        return quantile_coverage
    elif metric_name in ['quantile_calibration_error']:
        return quantile_calibration_error
    elif metric_name in ['mace', 'mean_absolute_calibration_error']:
        return mean_absolute_calibration_error
    else:
        raise ValueError(f"Performance metric '{metric_name}' not recognized as a valid option")