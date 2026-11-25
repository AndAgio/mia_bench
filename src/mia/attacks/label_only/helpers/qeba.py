"""QEBA, a HopSkipJump whose gradient estimates draw their random directions from a lower-dimensional subspace.

Paper: "QEBA: Query-Efficient Boundary-Based Blackbox Attack" (Li et al., CVPR 2020), https://arxiv.org/abs/2005.14137.
Authors' code: https://github.com/AI-secure/QEBA.
"""
import math
from typing import Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from .hop_skip_jump import HopSkipJump


def block_basis(shape: Tuple[int, ...], reduction_factor: int) -> torch.Tensor:
    """Return the basis of the custom mode: one indicator per channel and square block of reduction_factor pixels.

    Not in the paper: a piecewise-constant version of the spatial subspace. Returns a (C*H*W, n_sub) tensor.
    """
    if len(shape) != 3:
        raise ValueError(f"QEBA's custom basis needs images with shape (C, H, W), got shape {tuple(shape)}.")
    channels, height, width = shape
    reduction_factor = max(1, int(reduction_factor))
    # One indicator per channel and non-overlapping spatial block. Qeba normalizes the columns before use.
    columns = []
    for channel in range(channels):
        for row in range(0, height, reduction_factor):
            for col in range(0, width, reduction_factor):
                direction = torch.zeros(shape, dtype=torch.float32)
                direction[
                    channel,
                    row:min(row + reduction_factor, height),
                    col:min(col + reduction_factor, width),
                ] = 1.0
                columns.append(direction.reshape(-1))
    return torch.stack(columns, dim=1)


def input_gradients(model: torch.nn.Module, images: Sequence, device: torch.device, batch_size: int = 256):
    """Return the gradients of a reference model's loss with respect to each image, flattened, on the CPU.

    The loss is the cross-entropy at the model's own predicted label, as in the QEBA authors' code
    (gradient_generate.py). `images` holds tensors, or tuples whose first item is the image, as the model takes them.
    """
    model = model.to(device).eval()
    gradients = []
    for batch in DataLoader(images, batch_size=batch_size, shuffle=False):
        inputs = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device).requires_grad_(True)
        with torch.enable_grad():
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, logits.argmax(dim=1), reduction='sum')
            gradient, = torch.autograd.grad(loss, inputs)
        gradients.append(gradient.detach().flatten(start_dim=1).cpu())
    return torch.cat(gradients)


def gradient_pca_basis(gradients: torch.Tensor, reduction_factor: int) -> torch.Tensor:
    """Return QEBA-I's subspace: the top principal components of reference-model input gradients (paper, Sec. 3.4).

    Keeps about 1/reduction_factor^2 of the input dimensions, as the spatial and frequency subspaces do, and at most
    one fewer than the number of gradients. Returns a (dims, n_sub) tensor.
    """
    gradients = gradients.to(dtype=torch.float32)
    if gradients.shape[0] < 2:
        raise ValueError("QEBA-PCA needs the gradients of at least two samples.")
    gradients = gradients - gradients.mean(dim=0, keepdim=True)
    requested_rank = max(1, gradients.shape[1] // (max(1, int(reduction_factor)) ** 2))
    rank = min(requested_rank, gradients.shape[0] - 1, gradients.shape[1])
    _, _, components = torch.pca_lowrank(gradients, q=rank, center=False)
    return components


class Qeba(HopSkipJump):
    """HopSkipJump with QEBA's gradient estimate, whose random directions come from a lower-dimensional subspace.

    The variant sets the subspace:

    - 'spatial' (QEBA-S): low-resolution Gaussian noise, bilinearly upsampled to the image size;
    - 'dct' (QEBA-F): random low-frequency DCT coefficients, transformed back to an image;
    - 'pca' (QEBA-I): a given (dims, n_sub) basis, the principal components of reference-model input gradients
      (gradient_pca_basis);
    - 'custom': any other given basis, such as block_basis.

    The reduction factor r keeps about 1/r^2 of the input dimensions in the spatial and frequency subspaces. Each
    gradient estimate spends HopSkipJump's growing number of queries, init_num_evals * sqrt(t), where the QEBA
    authors' code spends a fixed number. Inputs that are not images, such as tabular rows, get HopSkipJump's estimate
    in the full input space.
    """

    def __init__(
        self,
        norm: str,
        max_queries: int,
        gamma: float = 1.0,
        stepsize_search: str = "geometric_progression",
        max_num_evals: int = 10000,
        init_num_evals: int = 100,
        clamping: bool = False,
        logger: callable = None,
        # QEBA-specific:
        variant: str = "spatial",            # 'spatial' | 'dct' | 'pca' | 'custom'
        reduction_factor: int = 2,           # r: the subspace keeps about 1/r^2 of the input dimensions
        basis: torch.Tensor = None,          # (m, n_sub) for 'pca' or 'custom'
        normalize_basis_cols: bool = True,   # normalize columns of basis to unit L2
    ):
        """Set HopSkipJump's search parameters, plus the subspace: its variant, reduction factor and basis."""
        super().__init__(
            norm=norm,
            max_queries=max_queries,
            gamma=gamma,
            stepsize_search=stepsize_search,
            max_num_evals=max_num_evals,
            init_num_evals=init_num_evals,
            clamping=clamping,
            logger=logger,
        )
        self.name = f"QEBA-{variant.upper()}"
        assert variant in ("spatial", "dct", "pca", "custom"), "invalid QEBA variant"
        self.variant = variant
        self.r = int(max(1, reduction_factor))
        self.basis = basis  # will be moved to device in run()
        self.normalize_basis_cols = bool(normalize_basis_cols)

        # cache for DCT matrices keyed by (H,W,device)
        self._dct_cache = {}

    # -------------------------- helpers: shapes & subspaces -------------------

    @staticmethod
    def _is_image(x: torch.Tensor) -> bool:
        """Return whether a single sample is an image (C, H, W), whose subspaces QEBA can use."""
        # image-like shape (C,H,W)
        return x.dim() == 3 and x.shape[-1] >= 2 and x.shape[-2] >= 2

    def _prepare_basis(self, sample: torch.Tensor):
        """Ensure PCA/custom basis is on the right device/shape."""
        if self.basis is None:
            raise ValueError(f"{self.name} '{self.variant}' requires a 'basis' tensor of shape (m, n_sub).")
        m = int(torch.numel(sample))
        if self.basis.dim() != 2 or self.basis.shape[0] != m:
            raise ValueError(f"{self.name} basis shape must be (m, n_sub) with m={m}, got {tuple(self.basis.shape)}")
        if self.basis.device != self.device:
            self.basis = self.basis.to(self.device)
        if self.normalize_basis_cols:
            # normalize columns
            B = self.basis
            self.basis = B / (torch.norm(B, p=2, dim=0, keepdim=True) + 1e-12)

    # ---------- DCT (2D) utilities using matrix-form DCT-II / IDCT-II --------

    def _dct_1d_matrix(self, N: int, device: torch.device) -> torch.Tensor:
        """Return the orthonormal N x N DCT-II matrix, cached."""
        # DCT-II orthonormal matrix C (N x N)
        # C[k,n] = alpha(k) * cos( pi*(n+0.5)*k / N ), alpha(0)=sqrt(1/N), else sqrt(2/N)
        key = ("dct1d", N, device)
        if key in self._dct_cache:
            return self._dct_cache[key]
        n = torch.arange(N, dtype=torch.float32, device=device).view(1, N)
        k = torch.arange(N, dtype=torch.float32, device=device).view(N, 1)
        C = torch.cos(math.pi * (n + 0.5) * k / float(N))
        C[0, :] *= math.sqrt(1.0 / N)
        C[1:, :] *= math.sqrt(2.0 / N)
        self._dct_cache[key] = C
        return C

    def _get_dct_mats(self, H: int, W: int, device: torch.device):
        """Return the DCT-II matrices of the image's height and width, cached."""
        key = ("dct2d", H, W, device)
        if key in self._dct_cache:
            return self._dct_cache[key]
        Ch = self._dct_1d_matrix(H, device)
        Cw = self._dct_1d_matrix(W, device)
        self._dct_cache[key] = (Ch, Cw)
        return Ch, Cw

    def _idct2_lowfreq_batch(self, B: int, C: int, H: int, W: int) -> torch.Tensor:
        """
        Sample low-frequency DCT coefficients and reconstruct via IDCT:
            X = C_h^T * Q * C_w, with Q only having top-left (H//r, W//r) coefficients.
        Returns: rv of shape (B, C, H, W)
        """
        hf = max(1, H // self.r)
        wf = max(1, W // self.r)
        Ch, Cw = self._get_dct_mats(H, W, self.device)  # (H,H), (W,W)

        # allocate zero coeffs then fill low-freq block
        Q = torch.zeros((B, C, H, W), device=self.device, dtype=torch.float32)
        Q[:, :, :hf, :wf] = torch.randn((B, C, hf, wf), device=self.device)

        # IDCT per batch/channel: X = Ch^T @ Q @ Cw
        # We'll do: X_bc = Ch.t() @ Q_bc @ Cw
        # vectorize via reshapes:
        # left multiply: (H,H) x (H,W) -> (H,W)
        X = torch.einsum("ij,bcjk->bcik", Ch.t(), Q)
        X = torch.einsum("bcij,jk->bcik", X, Cw)
        return X

    # -------------------- QEBA direction samplers (subspaces) -----------------

    def _sample_directions_spatial(self, sample: torch.Tensor, B: int) -> torch.Tensor:
        """
        QEBA-S: sample low-res Gaussian noise and bilinear-upsample to input size.
        """
        C, H, W = sample.shape
        hr, wr = max(1, H // self.r), max(1, W // self.r)
        low = torch.randn((B, C, hr, wr), device=self.device)
        rv = torch.nn.functional.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)
        return rv

    def _sample_directions_dct(self, sample: torch.Tensor, B: int) -> torch.Tensor:
        """
        QEBA-F: sample low-frequency DCT coefficients and IDCT to image space.
        """
        C, H, W = sample.shape
        rv = self._idct2_lowfreq_batch(B, C, H, W)
        return rv

    def _sample_directions_basis(self, sample: torch.Tensor, B: int) -> torch.Tensor:
        """
        QEBA-I / custom: sample v in R^{n_sub}, then rv = W v in input space.
        """
        self._prepare_basis(sample)
        m, n_sub = self.basis.shape
        v = torch.randn((B, n_sub), device=self.device)
        v = v / (torch.norm(v, p=2, dim=1, keepdim=True) + 1e-12)  # unit sphere in low-dim
        rv_flat = v @ self.basis.t()                                # (B, m)
        return rv_flat.view((B,) + tuple(sample.shape))

    # ---------------------- QEBA gradient estimator override ------------------

    @torch.no_grad()
    def approximate_gradient(
        self,
        sample: torch.Tensor,  # (...), single sample
        label_or_target: int,
        num_evals: int,
        delta: float,
    ) -> torch.Tensor:
        """
        QEBA: same estimator as HSJA but the random directions are drawn
        from a *low-dimensional subspace* (spatial / DCT / PCA).

        If `sample` is not image-shaped (C,H,W), fallback to HSJA's full-space
        estimator (useful for tabular).
        """
        # Fallback for non-image inputs (tabular or 1D): use HSJA estimator.
        if not self._is_image(sample):
            return super().approximate_gradient(sample, label_or_target, num_evals, delta)

        shape = sample.shape    # (C,H,W)
        B = int(num_evals)

        # 1) sample subspace directions
        if self.variant == "spatial":
            rv = self._sample_directions_spatial(sample, B)    # (B,C,H,W)
        elif self.variant == "dct":
            rv = self._sample_directions_dct(sample, B)        # (B,C,H,W)
        elif self.variant in ("pca", "custom"):
            rv = self._sample_directions_basis(sample, B)      # (B,C,H,W)
        else:
            raise ValueError(f"Unknown {self.name} variant: {self.variant}")

        # 2) Normalize each direction to unit L2 (as in HSJA)
        rv = rv / (self._flatten(rv).norm(p=2, dim=1).view(B, *([1] * len(shape))) + 1e-12)

        # 3) Query perturbed points (batch) and build estimator (HSJA logic)
        perturbed = sample.unsqueeze(0) + float(delta) * rv
        perturbed = self.clip_image(perturbed)
        # Align rv to the exact displacement on the (possibly) clamped domain
        rv = (perturbed - sample.unsqueeze(0)) / float(delta)

        decisions = self.decision_function(perturbed, label_or_target)   # (B,)
        fval = (2.0 * torch.tensor(decisions, device=self.device, dtype=torch.float32) - 1.0)  # {-1,+1}

        m = float(fval.mean().item())
        if abs(m - 1.0) < 1e-12:
            gradf = rv.mean(dim=0)
        elif abs(m + 1.0) < 1e-12:
            gradf = -rv.mean(dim=0)
        else:
            fval = fval - fval.mean()
            gradf = (fval.view(B, *([1] * len(shape))) * rv).mean(dim=0)

        # 4) Normalize gradient direction
        gradf = gradf / (gradf.reshape(-1).norm(p=2) + 1e-12)
        return gradf

    def run(self, inputs: torch.Tensor, labels: torch.Tensor, model: torch.nn.Module, targeted: bool, device: torch.device = None):
        """Run HopSkipJump's search on each input, with the basis moved to the device."""
        if self.basis is not None and device is not None and self.basis.device != device:
            self.basis = self.basis.to(device)
        return super().run(inputs, labels, model, targeted, device)

    def run_single(self, inputs: torch.Tensor, labels: torch.Tensor, model: torch.nn.Module, targeted: bool, device: torch.device = None):
        """Run HopSkipJump's search on one input, with the basis moved to the device."""
        if self.basis is not None and device is not None and self.basis.device != device:
            self.basis = self.basis.to(device)
        return super().run_single(inputs, labels, model, targeted, device)
