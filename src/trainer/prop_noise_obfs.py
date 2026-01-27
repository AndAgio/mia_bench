import pdb
import time
from typing import Tuple, Dict
import copy
import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler


import torch
import numpy as np
from typing import List
import torch.nn.functional as F


print_stats2 = 0

"""
Metric Privacy Noise:
model: the model to obfuscate (obfuscation happens in place, the function doenst return something)
d: distance to protect (hyperparameter, try values 1,2,5,20)
b: how much noise to put (hyperparameter, try values, 0.5,1,2)
which: which layer to target to obfuscate: currently I have implemented penultimate
debug: Keep it false, it prints some values when True, I will eventually add more
coord_mask: which coordinates to pick. I am trying to implement to pick the best ones. 
If you want to choose random ones just put a random mask over the parameters of the penultimate layer
coord_mask is like this: coord_mask={"weight": weight_mask, "bias": bias_mask},
so you have to create the weight_mask and bias_mask for the penltumate layer (or use my test function below)
"""


"""
test function for random weights (without their risks)
just pass the layer you want and the distance d (e.g. 1,5,10,20)
"""
@torch.no_grad()
def build_test_coord_mask_from_weights(
    layer: torch.nn.Module,
    d: float,
):
    """
    Create a coord_mask dict for testing:
      - weight mask: based on |weight| L1 budget
      - bias mask: any row with at least one selected weight
    """
    if not hasattr(layer, "weight"):
        raise ValueError("Layer has no weight")

    weight = layer.weight.detach()

    weight_mask = select_coords_until_l1_le_d_(
        tensor=weight,
        d=d,
        require_at_least_one=True,
    )

    if getattr(layer, "bias", None) is not None:
        bias_mask = weight_mask.any(dim=1)
    else:
        bias_mask = None

    return {
        "weight": weight_mask,
        "bias": bias_mask,
    }


def metric_privacy_obfuscation(
    model: nn.Module,
    d: float,
    b: float,
    which: str = "penultimate",
    clip_scope: str = "masked",   # "all" or "masked" seting to all will clip all parameters
    debug: bool = False,
    coord_mask: torch.Tensor = None, 
):
    #If you want to use my test function, just pass None on coord_mask and use this:
    #coord_mask = build_test_coord_mask_from_weights(layer,d)

    global print_stats2
    if (print_stats2 == 0):
        print("****** RUNNING METRIC PRIVACY DEFENCE *****")
        print_stats2 = 1

    if clip_scope not in {"all", "masked"}:
        raise ValueError("clip_scope must be 'all' or 'masked'")

    #select the penltimate layer
    layer = _select_trainable_layer(model, which=which)


    #apply the noise
    _select_clip_noise_step_(
        layer,
        d=d,
        b=b,
        clip_scope=clip_scope,
        debug=debug,
        coord_mask=coord_mask,
    )



@torch.no_grad()
def _snapshot_layer_grads(layer):
    """
    Returns a dict:
      name -> cloned grad tensor
    """
    snaps = {}
    for name, p in layer.named_parameters(recurse=True):
        if p.grad is not None:
            snaps[name] = p.grad.detach().clone()
    return snaps


@torch.no_grad()
def _print_grad_changes(
    layer,
    grads_before,
    k_show: int = 5,
):
    """
    Prints only the coordinates that changed.
    """
    for name, p in layer.named_parameters(recurse=True):
        if p.grad is None or name not in grads_before:
            continue

        g_before = grads_before[name]
        g_after = p.grad
        diff = g_after - g_before

        changed = diff != 0
        n_changed = changed.sum().item()

        if n_changed == 0:
            continue

        print(f"[DEFENSE CHECK] {layer.__class__.__name__}.{name}: "
              f"{n_changed}/{diff.numel()} coords changed")

        flat_idx = changed.view(-1).nonzero(as_tuple=False).view(-1)
        for i in flat_idx[:k_show]:
            idx = torch.unravel_index(i, diff.shape)
            print(
                f"  idx={tuple(int(x) for x in idx)} | "
                f"{g_before[idx].item():.6e} -> {g_after[idx].item():.6e} "
                f"(Δ={diff[idx].item():+.6e})"
            )

def _is_trainable_leaf(module: torch.nn.Module) -> bool:
    """
    A trainable leaf module:
    - has parameters
    - none of its children have parameters
    """
    has_params = any(p.requires_grad for p in module.parameters(recurse=False))
    has_trainable_children = any(
        any(p.requires_grad for p in c.parameters())
        for c in module.children()
    )
    return has_params and not has_trainable_children


def get_trainable_leaf_modules(model: torch.nn.Module) -> List[torch.nn.Module]:
    return [m for m in model.modules() if _is_trainable_leaf(m)]


def select_target_layer(model: torch.nn.Module, which: str = "penultimate"):
    leaves = get_trainable_leaf_modules(model)
    if len(leaves) < 2:
        raise ValueError("Model has fewer than 2 trainable layers")

    if which == "last":
        return leaves[-1]
    elif which == "penultimate":
        return leaves[-2]
    else:
        raise ValueError(f"Unknown layer selector: {which}")



def _unwrap_model(model: nn.Module) -> nn.Module:
    # Handles DistributedDataParallel / DataParallel
    return model.module if hasattr(model, "module") else model

@torch.no_grad()
def clip_masked_l1_(grad: torch.Tensor, mask: torch.Tensor, l1_max: float, eps: float = 1e-12) -> float:
    """
    Clips only grad[mask] so that sum(|grad[mask]|) <= l1_max.
    Returns pre-clip masked L1.
    """
    
    masked_l1 = grad[mask].abs().sum().item()
    if masked_l1 > l1_max:
        scale = l1_max / (masked_l1 + eps)
        grad[mask] = grad[mask] * scale 
    return masked_l1


def _is_trainable_leaf(module: nn.Module) -> bool:
    """
    A trainable leaf module:
      - has at least one direct (recurse=False) trainable parameter
      - none of its children have any trainable parameters
    """
    has_direct_trainable_params = any(
        p.requires_grad for p in module.parameters(recurse=False)
    )
    if not has_direct_trainable_params:
        return False

    # if any child has any trainable params, this isn't a leaf
    for child in module.children():
        if any(p.requires_grad for p in child.parameters()):
            return False

    return True


def _trainable_leaf_modules(model: nn.Module) -> List[nn.Module]:
    model = _unwrap_model(model)
    leaves: List[nn.Module] = []
    for m in model.modules():
        if _is_trainable_leaf(m):
            leaves.append(m)
    return leaves


def _select_trainable_layer(
    model: nn.Module,
    which: str = "penultimate",
    debug: bool = False,
) -> nn.Module:
    """
    Selects a trainable layer in a model in a model-agnostic way.

    which:
      - "last": last trainable leaf module
      - "penultimate": second-to-last trainable leaf module
    """
    leaves = _trainable_leaf_modules(model)

    if debug:
        print(f"[layer-select] found {len(leaves)} trainable leaf modules:")
        for i, m in enumerate(leaves):
            # show shape summary of direct params
            shapes = [tuple(p.shape) for p in m.parameters(recurse=False) if p.requires_grad]
            print(f"  {i:02d}: {m.__class__.__name__} direct_param_shapes={shapes}")

    if len(leaves) == 0:
        raise ValueError("No trainable leaf modules found (are all params frozen?)")

    if which == "last":
        return leaves[-1]

    if which == "penultimate":
        if len(leaves) < 2:
            raise ValueError("Cannot select penultimate layer: model has <2 trainable leaf modules")
        return leaves[-2]

    raise ValueError(f"Unknown which='{which}'. Use 'last' or 'penultimate'.")

@torch.no_grad()
def _select_clip_noise_step_(
    layer,
    d,
    b,
    clip_scope="masked",
    debug=False,
    coord_mask=None,   # bool tensor OR dict {"weight": bool2d, "bias": bool1d}
):
    for name, p in layer.named_parameters(recurse=False):
        if (not p.requires_grad) or (p.grad is None):
            continue

        g = p.grad

        # pick the right mask for this parameter
        if isinstance(coord_mask, dict):
            m = coord_mask.get(name, None)      # name is "weight" or "bias"
        else:
            m = coord_mask                      # single mask (weight-only case)

        # If a mask is provided, validate shape
        if m.shape != g.shape:
            raise ValueError(f"coord_mask[{name}] shape {m.shape} must match grad shape {g.shape}")
        m = m.to(device=g.device, dtype=torch.bool)
        #clip (as Theorem 2.1. suggests)
        clip_l1_all_(g,l1_max = d/2.0)
        
        #clip only the selecetd parameters: just for testing it doenst follow theorem
        #clip_masked_l1_(g, m, l1_max=d / 2.0)
       
        #add laplacian noise
        add_laplace_noise_masked_(g, m, b=b)
     
        if debug:
            print(f"[obfs] {name}: masked={int(m.sum())}/{m.numel()}")

@torch.no_grad()
def add_laplace_noise_masked_(grad: torch.Tensor, mask: torch.Tensor, b: float):
    """
    Adds i.i.d. Laplace(0,b) noise only on grad[mask].
    """
    if b <= 0:
        return

    dist = torch.distributions.Laplace(
        loc=torch.zeros((), device=grad.device, dtype=torch.float32),
        scale=torch.full((), float(b), device=grad.device, dtype=torch.float32),
    )

    noise = dist.sample(grad.shape).to(device=grad.device, dtype=grad.dtype)
    grad[mask] = grad[mask] + noise[mask]  

@torch.no_grad()
def clip_l1_all_(grad: torch.Tensor, l1_max: float, eps: float = 1e-12) -> float:
    l1 = grad.abs().sum().item()
    if l1 > l1_max:
        grad.mul_(l1_max / (l1 + eps))
    return l1

@torch.no_grad()
def select_coords_risk_biased_random_until_weight_l1_le_d_(
    weight: torch.Tensor,
    risk_scores: torch.Tensor,
    d: float,
    *,
    alpha: float = 1.0,              # risk sharpness: 0.5 flatter, 2.0 sharper
    eps: float = 1e-12,              # avoids zero-prob
    max_draw: int = None,            # optional cap on how many candidates to draw
    require_at_least_one: bool = True,
    generator: torch.Generator = None,
):
    """
    Random (without replacement) coordinate selection biased by risk_scores,
    then keep the sampled coords in that random order until sum(|weight[selected]|) <= d.

    Returns: (mask same shape as weight, stats dict)
    """
    if weight.shape != risk_scores.shape:
        raise ValueError(f"weight shape {weight.shape} must match risk_scores shape {risk_scores.shape}")
    if d < 0:
        raise ValueError("d must be >= 0")

    w = weight.reshape(-1)
    r = risk_scores.reshape(-1)

    n = w.numel()
    if n == 0:
        mask = torch.zeros_like(w, dtype=torch.bool).view_as(weight)
        return mask, {"k_selected": 0, "n_total": 0, "used_weight_l1": 0.0}

    absw = w.detach().float().abs()

    # Build sampling probabilities from risk (must be nonnegative)
    rr = r.detach().float()
    rr = torch.clamp(rr, min=0.0)

    # if all risks are zero, fall back to uniform random
    if float(rr.max().item()) == 0.0:
        probs = torch.full((n,), 1.0 / n, device=rr.device, dtype=torch.float32)
    else:
        probs = (rr + eps).pow(alpha)
        probs = probs / probs.sum()

    # How many to draw as candidates
    if max_draw is None:
        # draw all candidates (exact but heavier)
        k_draw = n
    else:
        k_draw = int(min(max_draw, n))

    # Weighted sample without replacement
    order = torch.multinomial(probs, num_samples=k_draw, replacement=False, generator=generator)

    # Now take prefix until weight-L1 budget is exhausted
    cumsum = torch.cumsum(absw[order], dim=0)
    # count of entries where cumulative <= d
    k = int(torch.searchsorted(
        cumsum,
        torch.tensor(float(d), device=cumsum.device, dtype=cumsum.dtype),
        right=True
    ).item())

    forced = False
    if require_at_least_one and k == 0:
        k = 1
        forced = True

    chosen = order[:k]

    mask_flat = torch.zeros(n, dtype=torch.bool, device=weight.device)
    mask_flat[chosen] = True
    mask = mask_flat.view_as(weight)

    used_l1 = absw[chosen].sum().item() if k > 0 else 0.0
    stats = {
        "k_selected": int(k),
        "n_total": int(n),
        "frac_selected": float(k) / float(n),
        "used_weight_l1": float(used_l1),
        "budget_d": float(d),
    }
    #print("K is",k, "<<<")
    #print("USED L1",stats["used_weight_l1"])
    return mask, stats

@torch.no_grad()
def select_coords_toprisk_until_weight_l1_le_d_(
    weight: torch.Tensor,
    risk_scores: torch.Tensor,
    d: float,
    *,
    require_at_least_one: bool = True,
) -> torch.Tensor:
    """
    Select coords in descending risk_scores order until sum(|weight[selected]|) <= d.
    Returns boolean mask with same shape as weight.

    Note: If d is very small (smaller than the smallest |weight|), the mask will be empty
    unless require_at_least_one=True.
    """
    if weight.shape != risk_scores.shape:
        raise ValueError(f"weight shape {weight.shape} must match risk_scores shape {risk_scores.shape}")
    if d < 0:
        raise ValueError("d must be >= 0")

    w = weight.reshape(-1)
    r = risk_scores.reshape(-1)
    #print("RISKS:", r)
    #print("WEIGHTS",w)

    # order by risk descending
    order = torch.argsort(r, descending=True)

    # cumulative L1 mass of weights along that order
    absw = w.abs()
    cumsum = torch.cumsum(absw[order], dim=0)

    absw = w.detach().float().abs()
    total_l1 = absw.sum().item()
    min_w = absw.min().item()
    max_w = absw.max().item()

    #print(f"d={d}  total_weight_L1={total_l1:.6f}  min|w|={min_w:.6f}  max|w|={max_w:.6f}")
    #print("first 5 |w| in risk order:", absw[order[:5]].cpu().tolist())
    #print("first 5 risks:", r[order[:5]].detach().cpu().tolist())

    # choose largest k such that cumsum[k-1] <= d
    # searchsorted returns insertion index; using right=True gives count of <= d
    k = int(torch.searchsorted(
        cumsum,
        torch.tensor(d, device=cumsum.device, dtype=cumsum.dtype),
        right=True
    ).item())

    if require_at_least_one and w.numel() > 0:
        k = max(k, 1)

    k = min(k, w.numel())

    chosen = order[:k]
    mask = torch.zeros_like(w, dtype=torch.bool)
    mask[chosen] = True
    stats = {
    "k_selected": k,
    "n_total": w.numel(),
    "frac_selected": (k / max(1, w.numel())),
    }
    #print("K Is",k)
    return mask.view_as(weight),stats


@torch.no_grad()
def select_coords_toprisk_until_l1_ge_d_(
    grad: torch.Tensor,
    risk_scores: torch.Tensor,
    d: float,
) -> torch.Tensor:
    """
    Deterministically select coordinates with largest risk_scores until
    sum(|grad[selected]|) >= d (or select all available if not enough L1 mass).

    Returns a boolean mask with same shape as grad.
    """
    if risk_scores.shape != grad.shape:
        raise ValueError(f"risk_scores shape {risk_scores.shape} must match grad shape {grad.shape}")

    flat_g = grad.view(-1)
    flat_r = risk_scores.view(-1)

    # Sort coords by risk descending (highest risk first)
    order = torch.argsort(flat_r, descending=True)

    abs_g = flat_g.abs()
    cumsum = torch.cumsum(abs_g[order], dim=0)

    # first index where cumsum >= d
    idx = torch.searchsorted(
        cumsum,
        torch.tensor(d, device=grad.device, dtype=cumsum.dtype),
        right=False,
    )

    n = flat_g.numel()
    k = int(min(max(idx.item() + 1, 1), n))

    chosen = order[:k]
    mask_flat = torch.zeros(n, dtype=torch.bool, device=grad.device)
    mask_flat[chosen] = True
    return mask_flat.view_as(grad)

@torch.no_grad()
@torch.no_grad()
def select_coords_until_l1_le_d_(
    tensor: torch.Tensor,
    d: float,
    *,
    require_at_least_one: bool = True,
) -> torch.Tensor:
    """
    Deterministically select coordinates with largest |tensor| values until
    sum(|tensor[selected]|) <= d.

    Returns a boolean mask with same shape as tensor.
    """
    if d < 0:
        raise ValueError("d must be >= 0")

    flat = tensor.view(-1)
    abs_flat = flat.abs()

    # sort by magnitude descending
    order = torch.argsort(abs_flat, descending=True)

    cumsum = torch.cumsum(abs_flat[order], dim=0)

    # number of coords whose cumulative L1 <= d
    k = int(torch.searchsorted(
        cumsum,
        torch.tensor(float(d), device=cumsum.device, dtype=cumsum.dtype),
        right=True
    ).item())

    if require_at_least_one and flat.numel() > 0:
        k = max(k, 1)

    k = min(k, flat.numel())

    chosen = order[:k]
    mask = torch.zeros_like(flat, dtype=torch.bool)
    mask[chosen] = True
    return mask.view_as(tensor)

"""
def compute_laplace_b(d, epsilon, q, T, lam):

    Computes b = d / log( (epsilon * d * (1 - q)) / (T * (lam + 1) * q^2) )

    All inputs are assumed to be positive scalars.

    numerator = epsilon * d * (1.0 - q)
    denominator = T * (lam + 1.0) * (q ** 2)

    ratio = numerator / denominator
    if ratio <= 1.0:
        raise ValueError(
            f"log argument must be > 1, got {ratio}. "
            "Check epsilon, q, T, lambda."
        )

    return d / math.log(ratio)


def lambda_upper_bound(q: float, d: float, b: float, epsilon: float, T: float) -> float:

    Computes:
      min{
        (1-q) / (q * (exp(d/b) - 1)),
        (epsilon * d * (1-q)) / (T * q^2) - 1
      }

    if not (0 < q < 1):
        raise ValueError("q must be in (0,1)")
    if d <= 0 or b <= 0 or epsilon <= 0 or T <= 0:
        raise ValueError("d, b, epsilon, T must be > 0")

    term1 = (1.0 - q) / (q * (math.exp(d / b) - 1.0))
    term2 = (epsilon * d * (1.0 - q)) / (T * (q ** 2)) - 1.0

    return min(term1, term2)

"""