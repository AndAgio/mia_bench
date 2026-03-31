import time
import math
from typing import Callable, Tuple
import numpy as np
import torch

from src.utils.log import Loggable


class HopSkipJump(Loggable):
    """
    HopSkipJumpAttack (HSJA), decision-based black-box adversarial attack.

    This implementation mirrors the structure of the BlackboxBench HSJA you provided:
        - initialize() to get a first adversarial point
        - binary_search_batch() to project to the decision boundary
        - approximate_gradient() via random directions and decision queries
        - stepsize search via geometric progression (or optional grid search)
        - repeat until query budget is exhausted or an epsilon distance is reached

    Key design choices requested:
        - One boolean option: clamping
            * clamping=True  -> clamp inputs to [0,1] whenever needed (image-like bounded domain)
            * clamping=False -> do not clamp at all (useful for standardized/unbounded features)
        - No "channels_last" support: ALWAYS PyTorch format
            * Images:  x has shape (B, C, H, W)
            * Tabular: x has shape (B, D)

    Threat model:
        - Decision-based: the attack only uses the model's predicted labels (argmax),
            not probabilities or gradients.

    Attributes (main hyperparameters):
        p (str):
            - Norm used for distance and constraint, either 'l2' (L2) or 'linf' (L∞).
        max_queries (int):
            - Maximum number of model evaluations (queries) allowed per attacked sample.
        gamma (float):
            - Controls binary search termination threshold theta (smaller theta -> tighter boundary search).
        stepsize_search (str):
            - Either 'geometric_progression' (default) or 'grid_search'.
        max_num_evals (int):
            - Maximum number of random directions evaluated per iteration for gradient estimation.
        init_num_evals (int):
            - Initial number of random directions (grows with sqrt(iter)).
        match_bb_delta (bool):
            - If True, match BB’s off-by-one delta special case (special delta at j==1 instead of j==0).
    """

    def __init__(
        self,
        norm: str, # 'l2' or 'linf'
        max_queries: int,
        gamma: float = 1.0,
        stepsize_search: str = "geometric_progression",  # 'geometric_progression' or 'grid_search'
        max_num_evals: int = 10_000,
        init_num_evals: int = 100,
        clamping: bool = False,
        logger: Callable = None,
    ):
        super().__init__(logger=logger)
        self.name = 'HopSkipJump'
        # Validate norm type and stepsize search mode.
        assert norm in ("l2", "linf"), "norm must be 'l2' or 'linf'"
        assert stepsize_search in ("geometric_progression", "grid_search")
        # Norm type: 'l2' or 'linf'. Determines distance computation, theta formula, and update direction.
        self.norm = norm
        # Query budget per sample.
        self.max_queries = int(max_queries)
        # Gamma parameter to set boundary-search threshold theta.
        self.gamma = float(gamma)
        # Step size search strategy.
        self.stepsize_search = stepsize_search
        # Parameters controlling gradient estimation sampling budget.
        self.max_num_evals = int(max_num_evals)
        self.init_num_evals = int(init_num_evals)
        # The ONLY domain option:
        # - If True, clamp intermediate points to [0,1].
        # - If False, never clamp (unbounded domain).
        self.clamping = bool(clamping)
        # These are set by run()
        self.device = None
        self.model = None
        self.targeted = False
        # Query counter per attacked point (BB uses self.query).
        # It counts total number of evaluated samples in all decision_function calls.
        self.query = 0

    # ---------------------------------------------------------------------
    # Basic tensor utilities
    # ---------------------------------------------------------------------
    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        """
        Flatten all non-batch dimensions.

        Args:
            x: Tensor with shape (B, ...) where ... could be (C,H,W) or (D,)

        Returns:
            Tensor with shape (B, Dflat)
        """
        return x.view(x.shape[0], -1)

    # ---------------------------------------------------------------------
    # Domain clamping (single toggle)
    # ---------------------------------------------------------------------
    def clip_image(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply domain constraint if clamping is enabled.

        In BB HSJA, many operations clamp to [lb, ub]. With our single-flag design:
            - clamping=True  -> [0,1]
            - clamping=False -> no-op

        Args:
            x: Tensor with shape (...), (B, ...)

        Returns:
            Tensor of same shape, possibly clamped to [0,1].
        """
        if not self.clamping:
            return x
        return torch.clamp(x, 0.0, 1.0)

    # ---------------------------------------------------------------------
    # Model prediction + decision oracle (hard-label queries)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def predict_label(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Query the model and return predicted labels.

        IMPORTANT: This method assumes input is already in PyTorch convention:
            - images: (B, C, H, W)
            - tabular: (B, D)

        BlackboxBench parent class does NHWC->NCHW and clamps to [0,1].
        We keep only the clamp controlled by `clamping` and never permute.

        Args:
            xs: Tensor of shape (B, C, H, W) or (B, D)

        Returns:
            labels: Tensor of shape (B,) on CPU
        """
        x_eval = xs.to(self.device)

        # Optional domain clamp to [0,1] (requested single switch).
        x_eval = self.clip_image(x_eval)

        # Forward pass. Must return logits (B, num_classes) or a tuple whose last element is logits.
        out = self.model(x_eval)
        if isinstance(out, (tuple, list)):
            out = out[-1]

        # Hard labels
        return out.argmax(dim=1).detach().cpu()

    @torch.no_grad()
    def decision_function(self, images: torch.Tensor, label_or_target: int):
        """
        Decision oracle. Returns True on the "adversarial" side of the boundary.

        Untargeted:
            decision(x) = (pred(x) != true_label)
        Targeted:
            decision(x) = (pred(x) == target_label)

        Query counting:
            This increments self.query by the number of evaluated samples (batch size),
            matching BlackboxBench’s HSJA.

        Args:
            images:
                - shape (B, C, H, W) or (B, D)
                - or shape (C,H,W) / (D,) for single example (will be unsqueezed)
            label_or_target: int
                - if targeted=False: the original/true label
                - if targeted=True: the target label

        Returns:
            decisions: numpy bool array of shape (B,)
        """
        # If a single sample is provided without batch dim, add one.
        if images.dim() in (3, 1):  # (C,H,W) or (D,)
            images = images.unsqueeze(0)

        # Count queries as number of evaluated samples.
        self.query += images.shape[0]

        preds = self.predict_label(images).numpy()

        if self.targeted:
            return (preds == label_or_target)
        else:
            return (preds != label_or_target)

    # ---------------------------------------------------------------------
    # Distance computation (scalar, like BB)
    # ---------------------------------------------------------------------
    def compute_distance(self, x_ori: torch.Tensor, x_pert: torch.Tensor) -> float:
        """
        Compute distance between two single samples under the chosen norm.

        Args:
            x_ori: Tensor of shape (...) single sample
            x_pert: Tensor of shape (...) single sample

        Returns:
            dist: float
        """
        diff = (x_ori - x_pert).abs()

        if self.norm == "l2":
            return float(diff.reshape(-1).norm(p=2).item())
        else:
            return float(diff.max().item())

    # ---------------------------------------------------------------------
    # HSJA: gradient estimation (batched) - key query bottleneck
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def approximate_gradient(
        self,
        sample: torch.Tensor,          # (...), single sample
        label_or_target: int,
        num_evals: int,
        delta: float
    ) -> torch.Tensor:
        """
        Approximate the gradient direction at the current boundary point.

        HSJA’s key trick: sample random directions v_i, query decisions for x + delta*v_i,
        then use decision outcomes to estimate a normal vector to the boundary.

        Steps (mirrors BB HSJA logic):
            1) sample random directions rv (num_evals, ...)
                - L2: Gaussian
                - Linf: uniform[-1,1]
            2) L2-normalize each rv
            3) perturbed = sample + delta*rv
            4) decision_function(perturbed) -> boolean vector decisions
            5) fval = 2*decisions - 1 in {-1,+1}
            6) baseline-subtraction:
                - if all fval are +1 -> grad = mean(rv)
                - if all fval are -1 -> grad = -mean(rv)
                - else centered fval -> mean(fval*rv)
            7) normalize grad to unit L2 norm

        Args:
            sample: Tensor of shape (...) single point on/near the boundary
            label_or_target: int (true label for untargeted, target label for targeted)
            num_evals: number of random directions to query
            delta: magnitude of exploration around sample

        Returns:
            gradf: Tensor of shape (...) with unit L2 norm
        """
        shape = sample.shape
        B = int(num_evals)

        # (B, ...) random vectors
        if self.norm == "l2":
            rv = torch.randn((B,) + shape, device=self.device)
        else:
            rv = 2.0 * torch.rand((B,) + shape, device=self.device) - 1.0

        # Normalize each rv by its L2 norm so directions are on the unit sphere.
        rv = rv / (self._flatten(rv).norm(p=2, dim=1).view(B, *([1] * len(shape))) + 1e-12)

        # Perturb in batch (B, ...)
        perturbed = sample.unsqueeze(0) + float(delta) * rv

        # Domain clamp (only if clamping=True)
        perturbed = self.clip_image(perturbed)

        # Recompute rv exactly as BB: (perturbed - sample) / delta
        rv = (perturbed - sample.unsqueeze(0)) / float(delta)

        # Query once for the full batch -> vector of decisions
        decisions = self.decision_function(perturbed, label_or_target)  # (B,) bool

        # Convert {False,True} -> {-1,+1}
        fval = (2.0 * torch.tensor(decisions, device=self.device, dtype=torch.float32) - 1.0)  # (B,)

        # Baseline subtraction to reduce estimator bias.
        m = float(fval.mean().item())
        if abs(m - 1.0) < 1e-12:
            gradf = rv.mean(dim=0)
        elif abs(m + 1.0) < 1e-12:
            gradf = -rv.mean(dim=0)
        else:
            fval = fval - fval.mean()
            gradf = (fval.view(B, *([1] * len(shape))) * rv).mean(dim=0)

        # Normalize gradient direction (unit L2 norm)
        gradf = gradf / (gradf.reshape(-1).norm(p=2) + 1e-12)
        return gradf

    # ---------------------------------------------------------------------
    # Projection operator used by binary search
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def project(
        self,
        original: torch.Tensor,        # (...), single sample
        perturbed: torch.Tensor,       # (B, ...)
        alphas: torch.Tensor           # (B,)
    ) -> torch.Tensor:
        """
        Project points toward the original to search for the boundary.

        For L2:
            out = (1-alpha)*original + alpha*perturbed
            where alpha in [0,1].

        For L∞:
            out = clip(perturbed, original-alpha, original+alpha)
            where alpha is a per-sample radius.

        Args:
            original: Tensor of shape (...) original sample
            perturbed: Tensor of shape (B, ...) adversarial candidates
            alphas: Tensor of shape (B,) interpolation factors or radii

        Returns:
            projected: Tensor of shape (B, ...)
        """
        B = perturbed.shape[0]
        a = alphas.view(B, *([1] * original.dim()))

        if self.norm == "l2":
            return (1.0 - a) * original.unsqueeze(0) + a * perturbed
        else:
            lo = original.unsqueeze(0) - a
            hi = original.unsqueeze(0) + a
            return torch.max(torch.min(perturbed, hi), lo)

    # ---------------------------------------------------------------------
    # Binary search to the boundary (batch version)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def binary_search_batch(
        self,
        original: torch.Tensor,        # (...), single sample
        perturbed: torch.Tensor,       # (B, ...) or (...), adversarial
        label_or_target: int,
        theta: float
    ):
        """
        Binary search to approach the boundary from adversarial points.

        This mirrors BlackboxBench HSJA logic:
            - define highs/lows depending on norm
            - repeatedly project to midpoints
            - update highs/lows depending on which side of boundary midpoints are
            - stop when (high-low)/threshold <= 1

        Variables:
            dists_post_update:
                distance(original, perturbed_i) for each initial candidate (used for Linf highs, and returned)

            highs, lows:
                interval bounds for the binary search parameter alpha (L2) or radius (Linf)

            thresholds:
                stopping thresholds:
                    - L2: constant theta
                    - Linf: min(dist*theta, theta) per candidate

        Args:
            original: (...) original input
            perturbed: (B, ...) adversarial candidates (or (...), treated as (1,...))
            label_or_target: label used for decision oracle
            theta: boundary-search threshold parameter

        Returns:
            out_image: (...) best candidate after search
            dist_post_update: float distance(original, selected_initial_candidate)
        """
        if perturbed.dim() == original.dim():
            perturbed = perturbed.unsqueeze(0)
        B = perturbed.shape[0]

        # Distance between original and each candidate
        dists_post = torch.tensor(
            [self.compute_distance(original, perturbed[i]) for i in range(B)],
            device=self.device,
            dtype=torch.float32
        )

        # Initialize highs and thresholds depending on norm
        if self.norm == "inf":
            highs = dists_post.clone()
            thresholds = torch.minimum(dists_post * float(theta), torch.tensor(float(theta), device=self.device))
        else:
            highs = torch.ones(B, device=self.device)
            thresholds = torch.tensor(float(theta), device=self.device)

        lows = torch.zeros(B, device=self.device)

        def ratio_max():
            return torch.max((highs - lows) / (thresholds + 1e-12)).item()

        # Repeat until boundary interval is sufficiently small
        while ratio_max() > 1.0:
            mids = 0.5 * (highs + lows)
            mid_images = self.project(original, perturbed, mids)

            # Decisions: 1 if adversarial side, 0 otherwise
            decisions = self.decision_function(mid_images, label_or_target)
            decisions = torch.tensor(decisions, device=self.device)

            # Update intervals
            lows = torch.where(decisions == 0, mids, lows)
            highs = torch.where(decisions == 1, mids, highs)

            if self.query > self.max_queries:
                break

        # Final projected images at highs
        out_images = self.project(original, perturbed, highs)

        # Choose the candidate with minimum distance to original
        dists = torch.tensor(
            [self.compute_distance(original, out_images[i]) for i in range(B)],
            device=self.device,
            dtype=torch.float32
        )
        idx = int(torch.argmin(dists).item())

        out_image = out_images[idx]
        dist_post_update = float(dists_post[idx].item())
        return out_image, dist_post_update

    # ---------------------------------------------------------------------
    # Initialization step
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def initialize(self, x: torch.Tensor, label_or_target: int) -> torch.Tensor:
        """
        Find an initial adversarial point.

        BlackboxBench strategy:
            - sample random noise until it is misclassified (untargeted) or hits target class (targeted)
            - blend (binary search on alpha) between x and random_noise to approach boundary

        We adapt only the noise distribution based on clamping:
            - clamping=True:  random_noise ~ Uniform([0,1]) (bounded domain)
            - clamping=False: random_noise ~ Normal(0,1)   (unbounded domain)

        Args:
            x: (...) original sample
            label_or_target: label used by decision oracle

        Returns:
            initialization: (...) adversarial initialization
        """
        # 1) Find any adversarial noise sample.
        while True:
            if self.clamping:
                random_noise = torch.rand_like(x).to(self.device)   # [0,1]
            else:
                random_noise = torch.randn_like(x).to(self.device)  # unbounded

            success = self.decision_function(random_noise.unsqueeze(0), label_or_target)[0]
            if success:
                break
            if self.query > self.max_queries:
                break

        # 2) Blend-search to get closer to original while staying adversarial.
        low, high = 0.0, 1.0
        while (high - low) > 1e-3:
            mid = 0.5 * (high + low)
            blended = (1.0 - mid) * x + mid * random_noise
            blended = self.clip_image(blended)

            success = self.decision_function(blended.unsqueeze(0), label_or_target)[0]
            if success:
                high = mid
            else:
                low = mid

            if self.query > self.max_queries:
                break

        initialization = (1.0 - high) * x + high * random_noise
        initialization = self.clip_image(initialization)
        return initialization

    # ---------------------------------------------------------------------
    # Step size search: geometric progression
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def geometric_progression_for_stepsize(
        self,
        x: torch.Tensor,                 # (...), current boundary point
        label_or_target: int,
        update: torch.Tensor,            # (...), direction to step
        dist: float,                     # current distance(original, x)
        j: int                           # iteration index starting at 1
    ) -> float:
        """
        Geometric progression to find a step size epsilon that stays adversarial.

        Algorithm:
            - initialize epsilon = dist / sqrt(j)
            - while x + epsilon*update is NOT adversarial, halve epsilon
            - stop when it becomes adversarial or queries exhausted

        Args:
            x: current boundary point (single sample)
            update: update direction
            dist: current distance to original
            j: iteration number (1-indexed)

        Returns:
            epsilon step size (float)
        """
        epsilon = float(dist) / math.sqrt(j)

        while True:
            new = x + epsilon * update
            new = self.clip_image(new)
            success = self.decision_function(new.unsqueeze(0), label_or_target)[0]
            if success:
                return epsilon
            epsilon /= 2.0
            if self.query > self.max_queries:
                return epsilon

    # ---------------------------------------------------------------------
    # Main HSJA loop
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def find_adversarial_single(self, input_x: torch.Tensor, label_or_target: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
        """
        Perform HSJA on a SINGLE sample.

        Variables:
            d:
                number of scalar dimensions in the input (C*H*W or D)
            theta:
                boundary-search tolerance parameter:
                - L2:  gamma / (sqrt(d)*d)
                - Linf: gamma / d^2
            perturbed:
                current adversarial point (kept near boundary by binary_search_batch)
            dist_post_update:
                a distance term returned by binary_search_batch (used to scale delta)
            dist:
                current distance(original, perturbed) under p-norm
            delta:
                exploration radius for gradient estimation (changes per iteration)
            num_evals:
                number of random directions sampled for approximate_gradient
            gradf:
                estimated boundary normal direction
            update:
                for L2: gradf
                for Linf: sign(gradf)
            eps_step:
                step size found by stepsize search

        Stopping criteria (same spirit as BB HSJA):
            - stop if query budget exceeded
            - stop if dist < epsilon

        Args:
            input_x: (...) original sample (1,C,H,W) or (1,D)
            label_or_target: int

        Returns:
            adv: (1, ...) adversarial sample
        """

        assert input_x.shape[0] == 1, f"This {self.name} mirrors BB behavior: outer batch size must be 1."
        input_x = input_x[0].to(self.device) # (...), single sample
        label_or_target = int(label_or_target.item()) # scalar label or target

        d = int(np.prod(input_x.shape))

        # Binary search threshold theta
        if self.norm == "l2":
            theta = self.gamma / (math.sqrt(d) * d)
        else:
            theta = self.gamma / (d ** 2)

        # Reset per-sample query counter
        self.query = 0

        # 1) Initialization
        perturbed = self.initialize(input_x, label_or_target)

        # 2) Project initialization to boundary
        perturbed, dist_post_update = self.binary_search_batch(input_x, perturbed, label_or_target, theta)

        # Current distance to original
        dist = self.compute_distance(perturbed, input_x)

        # 3) Main loop
        iteration = 0
        while self.query < self.max_queries:

            if iteration == 0:
                delta = 0.1
            else:
                if self.norm == "l2":
                    delta = math.sqrt(d) * theta * dist_post_update
                else:
                    delta = d * theta * dist_post_update

            # Number of gradient evaluations increases like init_num_evals * sqrt(iter)
            num_evals = int(self.init_num_evals * math.sqrt(iteration + 1))
            num_evals = int(min(num_evals, self.max_num_evals))

            # Estimate gradient direction at boundary point
            gradf = self.approximate_gradient(perturbed, label_or_target, num_evals, delta)

            # Update direction depends on norm
            update = torch.sign(gradf) if self.norm == "inf" else gradf

            # Step size search
            if self.stepsize_search == "geometric_progression":
                eps_step = self.geometric_progression_for_stepsize(
                    perturbed, label_or_target, update, dist, iteration + 1
                )

                # Take a step and keep within domain if clamping=True
                perturbed = self.clip_image(perturbed + eps_step * update)

                # Project back to decision boundary
                perturbed, dist_post_update = self.binary_search_batch(
                    input_x, perturbed, label_or_target, theta
                )

            elif self.stepsize_search == "grid_search":
                # Try a grid of step sizes and pick the best candidate that stays adversarial
                epsilons = np.logspace(-4, 0, num=20, endpoint=True) * dist
                eps_t = torch.tensor(epsilons, device=self.device, dtype=torch.float32)

                # Broadcast epsilons over sample shape
                view = (20,) + (1,) * perturbed.dim()

                perturbeds = perturbed.unsqueeze(0) + eps_t.view(view) * update.unsqueeze(0)
                perturbeds = self.clip_image(perturbeds)

                idx = self.decision_function(perturbeds, label_or_target)
                if np.sum(idx) > 0:
                    chosen = perturbeds[torch.tensor(idx, device=self.device, dtype=torch.bool)]
                    perturbed, dist_post_update = self.binary_search_batch(input_x, chosen, label_or_target, theta)

            # Update distance & check stopping criteria
            dist = self.compute_distance(perturbed, input_x)

            # self.logger.print_it_same_line(f"{self.name} [DEBUG]: iteration: {iteration+1:d}, {self.norm} distance {dist:.4f}, queries {self.query}", console_only=True)

            iteration += 1
        
        # self.logger.set_logger_newline()

        return perturbed.unsqueeze(0), dist, self.query


    def run(self, inputs: torch.Tensor, labels: torch.Tensor, model: torch.nn.Module, targeted: bool, device: torch.device = None):
        """
        Entry point similar to DecisionBlackBoxAttack.run(), but simplified.

        Args:
            inputs:
                - images:  Tensor (B, C, H, W)
                - tabular: Tensor (B, D)
            labels: Tensor (B,)
            model: torch.nn.Module returning logits (B, num_classes)
            targeted: bool (True for targeted, False for untargeted)

        Returns:
            adv: Tensor (B, ...) (currently B==1)
            dist: Tensor (B,) distances of adversarials from original inputs
            q: int number of queries used
        """
        if device is None:
            self.logger.print_it(f"{self.name} [WARNING]: No device specified. Defaulting to CPU. For better performance, provide a GPU or GPU-like device.")
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.model = model.to(self.device)
        self.model = model
        self.targeted = targeted

        inputs = inputs.detach().float().to(self.device)
        labels = labels.detach().to(self.device)

        adversarials = []
        distances = []
        queries = []
        n_inputs = inputs.shape[0]
        start = time.time()
        for i in range(n_inputs):
            # self.logger.print_it_same_line(f"Running {self.name} on sample {i+1}/{n_inputs} [elapsed time: {time.time() - start:.1f} s]. This may take a while...")
            single_input = inputs[i:i+1]  # (1, C, H, W) or (1, D)
            single_label = labels[i:i+1]   # (1,)
            prediction = self.predict_label(single_input)[0].item()
            if prediction != single_label.item():
                # self.logger.print_it(f"Sample {i+1} is already misclassified. Skipping attack.")
                adversarials.append(single_input)
                distances.append(0.0)
                queries.append(1)
                continue 
            else:
                # self.logger.print_it(f"Sample {i+1} is correctly classified. Running attack.")
                pass
            adv, dist, q = self.find_adversarial_single(single_input, single_label)
            adversarials.append(adv)
            distances.append(dist)
            queries.append(q)
        
        adv = torch.cat(adversarials, dim=0)
        distances = torch.tensor(distances, device=self.device)
        return adv, distances, queries


    def run_single(self, inputs: torch.Tensor, labels: torch.Tensor, model: torch.nn.Module, targeted: bool, device: torch.device = None):
        """
        Wrapper for single-sample attack, matching BB HSJA structure.

        Args:
            inputs: Tensor (1, C, H, W) or (1, D)
            labels: Tensor (1,)
            model: torch.nn.Module returning logits (B, num_classes)
            targeted: bool (True for targeted, False for untargeted)
        Returns:
            adv: Tensor (1, ...) adversarial example
            dist: float distance of adversarial example from original input
            q: int number of queries used
        """
        if device is None:
            self.logger.print_it(f"{self.name} [WARNING]: No device specified. Defaulting to CPU. For better performance, provide a GPU or GPU-like device.")
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.model = model.to(self.device)
        self.targeted = targeted

        inputs = inputs.detach().float().to(self.device)
        labels = labels.detach().to(self.device)
        adv, dist, q = self.find_adversarial_single(inputs, labels)
        return adv, dist, q