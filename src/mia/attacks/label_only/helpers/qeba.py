# --- QEBA: Query-Efficient Boundary-based Black-box Attack -------------------
# Andrea: this class reuses your HSJA implementation and overrides only the
# gradient estimation step to sample from low-dimensional subspaces as in QEBA.
# Paper: Huichen Li et al., "QEBA: Query-Efficient Boundary-Based Blackbox Attack", CVPR 2020.
#   - Subspaces: Spatial (QEBA-S), Frequency (QEBA-F via DCT), Intrinsic (QEBA-I via PCA).
#   - This implementation keeps your API and logic intact.
# References: CVPR paper (openaccess + arXiv) and the authors' reference code (Foolbox-based).
#   (for readers) https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_QEBA_Query-Efficient_Boundary-Based_Blackbox_Attack_CVPR_2020_paper.pdf
#   (arXiv)      https://arxiv.org/pdf/2005.14137
#   (code)       https://github.com/AI-secure/QEBA
# [1](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_QEBA_Query-Efficient_Boundary-Based_Blackbox_Attack_CVPR_2020_paper.pdf)[2](https://arxiv.org/pdf/2005.14137)[3](https://github.com/AI-secure/QEBA)

import math
from typing import Sequence

import torch
from .hop_skip_jump import HopSkipJump


def build_qeba_basis(
    samples: Sequence,
    variant: str,
    reduction_factor: int,
    max_pca_samples: int = 256,
) -> torch.Tensor:
    """Build the basis used by the configured QEBA MIA CLI variant.

    PCA learns an intrinsic subspace from data available to the attacker.  The
    custom CLI variant uses a deterministic block-indicator basis; callers that
    use :class:`Qeba` directly can still pass any custom basis they want.
    """
    if variant not in ("pca", "custom"):
        raise ValueError(f"A basis is not used by QEBA variant '{variant}'.")
    if len(samples) == 0:
        raise ValueError(f"Cannot build a QEBA-{variant.upper()} basis from an empty dataset.")

    first = samples[0][0] if isinstance(samples[0], (tuple, list)) else samples[0]
    if not isinstance(first, torch.Tensor) or first.dim() != 3:
        raise ValueError(
            f"QEBA-{variant.upper()} basis construction requires image tensors "
            f"with shape (C, H, W), got {type(first).__name__} "
            f"with shape {getattr(first, 'shape', None)}."
        )
    channels, height, width = first.shape
    reduction_factor = max(1, int(reduction_factor))

    if variant == "custom":
        # One indicator per channel and non-overlapping spatial block. Qeba
        # normalizes the columns before use.
        columns = []
        for channel in range(channels):
            for row in range(0, height, reduction_factor):
                for col in range(0, width, reduction_factor):
                    direction = torch.zeros_like(first, dtype=torch.float32)
                    direction[
                        channel,
                        row:min(row + reduction_factor, height),
                        col:min(col + reduction_factor, width),
                    ] = 1.0
                    columns.append(direction.reshape(-1))
        return torch.stack(columns, dim=1)

    count = min(len(samples), max(2, int(max_pca_samples)))
    flattened = []
    for index in range(count):
        item = samples[index]
        sample = item[0] if isinstance(item, (tuple, list)) else item
        if not isinstance(sample, torch.Tensor) or tuple(sample.shape) != tuple(first.shape):
            raise ValueError("All samples used for a QEBA-PCA basis must have the same image shape.")
        flattened.append(sample.detach().to(device="cpu", dtype=torch.float32).reshape(-1))
    data = torch.stack(flattened)
    data = data - data.mean(dim=0, keepdim=True)
    # Match the spatial/frequency variants' approximate dimensional reduction.
    requested_rank = max(1, data.shape[1] // (reduction_factor ** 2))
    rank = min(requested_rank, data.shape[0] - 1, data.shape[1])
    if rank < 1:
        raise ValueError("QEBA-PCA needs at least two reference samples.")
    _, _, components = torch.pca_lowrank(data, q=rank, center=False)
    return components


class Qeba(HopSkipJump):
    """
    QEBA: decision-only black-box attack with query-efficient boundary search.

    Key idea:
        Use a *low-dimensional subspace* to sample query directions for
        boundary-based gradient estimation.

    Variants (QEBA-S / QEBA-F / QEBA-I):
        - 'spatial': sample low-res noise and bilinear-upsample to input size.
        - 'dct'    : sample low-frequency DCT coefficients and IDCT to image.
        - 'pca'    : sample in a precomputed (m x n_sub) basis and project.
        - 'custom' : user-supplied basis (same as 'pca').

    Notes:
        * If inputs are not images (e.g., 1D tabular), we fallback to HSJA’s approximate_gradient (full space).
        * This implementation targets L2/L∞ norms, exactly like HSJA.
        * For 'pca'/'custom', pass `basis` as a torch.Tensor of shape (m, n_sub).

    Args (adds to HopSkipJump):
        variant: str in {'spatial','dct','pca','custom'}
        reduction_factor: int r >= 1 (used for 'spatial' and 'dct')
        basis: Optional[torch.Tensor] (#dims, n_sub), for 'pca' or 'custom'
        normalize_basis_cols: bool, if True L2-normalize each column of basis
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
        reduction_factor: int = 8,           # r: e.g., 4/8/16 (must divide H,W reasonably)
        basis: torch.Tensor = None,          # (m, n_sub) for 'pca' or 'custom'
        normalize_basis_cols: bool = True,   # normalize columns of basis to unit L2
    ):
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

    # ---------------------------- run() override note -------------------------
    # We only need to ensure basis tensors are moved to the correct device.
    def run(self, inputs: torch.Tensor, labels: torch.Tensor, model: torch.nn.Module, targeted: bool, device: torch.device = None):
        if self.basis is not None and device is not None and self.basis.device != device:
            self.basis = self.basis.to(device)
        return super().run(inputs, labels, model, targeted, device)

    def run_single(self, inputs: torch.Tensor, labels: torch.Tensor, model: torch.nn.Module, targeted: bool, device: torch.device = None):
        if self.basis is not None and device is not None and self.basis.device != device:
            self.basis = self.basis.to(device)
        return super().run_single(inputs, labels, model, targeted, device)
