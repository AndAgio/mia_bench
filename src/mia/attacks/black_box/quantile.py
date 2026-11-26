import copy
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.special import logsumexp, ndtri
from scipy.stats import binom
from src.data.helpers import IndexedDataset, SubsampledDataset
from src.data.helpers import MultiDatasets
from src.mia.attacks.base_mia import BaseMIA, MembershipScores
from src.mia.helpers.shadow_manager import ShadowManager
from src.utils.configs import AttackerConfigs, OptimizerConfigs, SchedulerConfigs, TrainConfigs
from src.utils.log import get_logger
from src.utils import convert_to_hms
from typing import Union
import time
from src.trainer.train_manager import TrainManager
from src.utils.variables import (QUANTILE_DEFAULT_VAL_SPLIT, QUANTILE_PAPER_BATCH_SIZE, QUANTILE_PAPER_EPOCHS,
                                 QUANTILE_PAPER_GRAD_CLIP, QUANTILE_PAPER_OPTIMIZER, QUANTILE_TAIL_HIGH,
                                 QUANTILE_TAIL_LEVELS, QUANTILE_TAIL_LOW, QUANTILE_VALIDATION_WARNING_P_VALUE)

class QuantileMIA(BaseMIA):
    """Quantile regression: a model trained on the attacker's non-members predicts each sample's threshold.

    Paper: "Scalable Membership Inference Attacks via Quantile Regression", https://arxiv.org/abs/2307.03694.
    Documentation: docs/attacks/quantile.md.
    """
    def __init__(self, 
                defender_model: torch.nn.Module,
                attacker_configs: AttackerConfigs):
        """Check alpha, build the pinball levels, sample the attacker's data and define the regression model."""
        super().__init__(defender_model=defender_model, attacker_configs=attacker_configs)
        self.logger.print_it(f"Working with Quantile MIA!")
        if not 0 < self.attack_configs.quantile_alpha < 1:
            raise ValueError(f"--attacker_quantile_alpha must lie strictly between 0 and 1, got "
                             f"{self.attack_configs.quantile_alpha}.")
        decision_level = 1 - self.attack_configs.quantile_alpha
        self.quantile = pinball_levels(n=self.attack_configs.n_quantile, low=self.attack_configs.low_quantile,
                                       high=self.attack_configs.high_quantile,
                                       use_logscale=self.attack_configs.use_logscale,
                                       alpha=self.attack_configs.quantile_alpha).reshape([1, -1])
        if not self.attack_configs.use_gaussian:
            self.logger.print_it(f"Quantile MIA attacker: {self.quantile.shape[1]} pinball levels from "
                                 f"{self.quantile.min().item():.4g} to {self.quantile.max().item():.4g}, including "
                                 f"{QUANTILE_TAIL_LEVELS} from {QUANTILE_TAIL_LOW} to {QUANTILE_TAIL_HIGH}, "
                                 f"deciding at {decision_level:.4g}.")
        assert self.shadow_configs.n_shadow_datasets == 1, \
            f"Quantile MIA trains one shadow model, set by the configs, got {self.shadow_configs.n_shadow_datasets}."
        self.shadow_manager = ShadowManager(logger=self.logger)
        self.logger.print_it('Quantile MIA attacker: sampling of shadow datasets...')
        self.shadow_manager.sample_shadow_datasets(attacker_data_distribution=self.attacker_data_distribution,
                                                    auditing_dataset=self.audit_manager,
                                                    calibration_dataset=self.calibration_manager,
                                                    shadow_configs=self.shadow_configs,
                                                    attacker_hash=self.attacker_hash)
        self.logger.print_it('Quantile MIA attacker: definition of quantile model...')
        self.model_configs.num_classes = 2 if self.attack_configs.use_gaussian else self.quantile.shape[1]
        self.shadow_manager.build_shadow_models(n_models=1,
                                                model_configs=self.model_configs)

    def optimize(self, train_config: TrainConfigs):
        """Train the regression model on the attacker's data, keeping the epoch with the lowest validation loss."""
        self.logger.print_it('Quantile MIA attacker: setting quantiles and loss function...')
        train_config = paper_train_config(train_config)
        self.logger.print_it(f"Quantile MIA attacker: the regression model trains with the authors' recipe "
                             f"(AdamW, lr {QUANTILE_PAPER_OPTIMIZER['lr']}, weight decay "
                             f"{QUANTILE_PAPER_OPTIMIZER['weight_decay']}, gradient clipping "
                             f"{QUANTILE_PAPER_GRAD_CLIP}, batch {QUANTILE_PAPER_BATCH_SIZE}, "
                             f"{QUANTILE_PAPER_EPOCHS} epochs), "
                             f"not the --attacker_* training settings.")
        # Set up quantiles
        device = self.get_device(dev_str=train_config.device)
        self.quantile = self.quantile.to(device)
        self.quantile_loss_fn = GaussianLoss().to(device) if self.attack_configs.use_gaussian else PinballLoss(quantile=self.quantile).to(device)
        train_config.loss = self.quantile_loss_fn

        self.logger.print_it('Quantile MIA attacker: constructing quantile dataset...')
        self.defender_model.eval()
        features = []
        target_scores = []
        original_indices = []
        shadow_dataset = self.shadow_manager.get_dataset(index=0,
                                                        labels='original')
        shadow_loader = DataLoader(shadow_dataset, batch_size=1, shuffle=False)
        start_data = time.time()
        with torch.no_grad():
            for data, target, original_index, _ in shadow_loader:
                self.logger.print_it_same_line(f'Quantile MIA attacker: processing sample {len(features)+1}/{len(shadow_dataset)}...', console_only=True)
                features.append(data)
                original_indices.append(int(original_index))
                target_score, _ = self.defender_scoring_fn(data, target, device=train_config.device)
                target_scores.append(target_score)
            self.logger.set_logger_newline(console_only=True)
        features = torch.cat(features)
        target_scores = torch.cat(target_scores)
        quantile_dataset = IndexedDataset(TensorDataset(features, target_scores), original_indices=original_indices)
        # The epoch is chosen on a held-out share of the attacker's own data, never on the samples it attacks.
        val_split = self.attack_configs.val_split
        val_split = QUANTILE_DEFAULT_VAL_SPLIT if val_split is None else val_split
        fit_positions, val_positions = validation_split(len(quantile_dataset), val_split, seed=train_config.seed)
        quantile_dataset = MultiDatasets([SubsampledDataset.from_positions(quantile_dataset, fit_positions),
                                          SubsampledDataset.from_positions(quantile_dataset, val_positions)],
                                         ids=['train', 'val'])
        self.logger.print_it(f"Quantile MIA attacker: constructed quantile dataset in {time.time() - start_data:.2f}s: "
                             f"{len(fit_positions)} samples to train the regression model, {len(val_positions)} to "
                             f"choose its epoch.")

        self.logger.print_it('Quantile MIA attacker: training quantile model. This will take a while. Sit back and chill...')
        start = time.time()
        logger = get_logger(name='{} quantile regressor'.format(self.logger.name),
                            log_folder=self.logger.get_folder(),
                            mode=self.logger.get_mode())
        train_manager = TrainManager(train_configs=train_config,
                                    name='quantile regressor',
                                    logger=logger)
        train_manager.initialize_train(dataset=quantile_dataset,
                                        model=self.shadow_manager.get_model(index=0),
                                        configs=train_config)
        quantile_model = train_manager.train(extra_configs={'quantiles': self.quantile},
                                            return_best_model=True,
                                            return_last_model=False,
                                            return_stats=False)
        self.shadow_manager.update_model(index=0,
                                        model=quantile_model)
        # The attacker's own estimate of its false-positive rate: its validation samples are known non-members.
        quantile_model.eval()
        with torch.no_grad():
            model_device = next(quantile_model.parameters()).device
            predictions = torch.cat([quantile_model(chunk.to(model_device)).cpu()
                                     for chunk in features[val_positions].split(256)])
        _, val_decisions = self.membership_scores(target_scores[val_positions].numpy(), predictions.numpy())
        alpha = self.attack_configs.quantile_alpha
        self.logger.print_it(f"Quantile MIA attacker: {val_decisions.mean():.2%} of the {len(val_positions)} "
                             f"validation samples, all non-members, exceed their threshold: the attacker's estimate "
                             f"of its false-positive rate at alpha = {alpha}.")
        p_value = binom.sf(int(val_decisions.sum()) - 1, len(val_positions), alpha)
        if p_value < QUANTILE_VALIDATION_WARNING_P_VALUE:
            advice = '' if self.attack_configs.use_gaussian else ', or the Gaussian head'
            self.logger.print_it(f"Quantile MIA attacker [WARNING]: that is significantly more than alpha "
                                 f"(one-sided binomial test, p = {p_value:.2g}), so the attack's own decisions "
                                 f"(--decision_rule attack) will likely flag more non-members than alpha. Consider "
                                 f"the calibrated rule{advice}.")
        stop = time.time()
        self.reset_logger()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Quantile MIA attacker: Done optimizing. It took {}:{:02d}:{:02d}...'.format(h, m, s))

    def defender_scoring_fn(self, data: torch.Tensor, target: torch.Tensor, device: Union[torch.device, str] = 'cpu'):
        """The target's hinge score of each sample (its true-class logit minus its largest other logit), and logits."""
        if isinstance(device, str):
            device = QuantileMIA.get_device(dev_str=device)
        self.defender_model.eval()
        with torch.no_grad():
            logits = self.defender_model(data.to(device)).detach().cpu()
            onehot_label = torch.nn.functional.one_hot(target, num_classes=logits.shape[-1]).bool()
            score = logits[onehot_label]
            # Mask out the true label before taking max over incorrect labels
            logits_masked = logits.masked_fill(onehot_label, float('-inf'))
            score -= torch.max(logits_masked, dim=1)[0]
        return score, logits

    def score(self, dataset, device: Union[torch.device, str] = 'cpu'):
        """Score each sample by how far the target's score exceeds its predicted threshold: members exceed it.

        The Gaussian head scores (s - mu) / sigma, the pinball head the distance to its 1 - alpha head.
        """
        self.logger.print_it('Computing Quantile MIA scores. This may take a while...')
        start = time.time()
        if isinstance(device, str):
            device = QuantileMIA.get_device(dev_str=device)
        quantile_model = self.shadow_manager.get_model(index=0).to(device)
        quantile_model.eval()
        self.defender_model.eval()
        audit_dataset = dataset
        self.reset_logger()
        tot_samples = len(audit_dataset)
        audit_loader = DataLoader(audit_dataset, batch_size=1, shuffle=False)
        target_scores, predictions = [], []
        self.logger.print_it(f"Computing scores for all {tot_samples} samples. This may take a while...", file_only=True)
        if self.attack_configs.use_gaussian:
            self.logger.print_it("Gaussian quantile scores are signed-log z-scores, to avoid scale overflow.")
        for sample_index, (sample, label, _, _) in enumerate(audit_loader):
            self.logger.print_it_same_line(f'Quantile MIA attacker: computing score for sample {sample_index+1}/{tot_samples}...', console_only=True)
            with torch.no_grad():
                target_score, _ = self.defender_scoring_fn(sample, label, device=device)
                target_scores.append(target_score.detach().cpu().numpy())
                predictions.append(quantile_model(sample.to(device)).detach().cpu().numpy())
        self.logger.set_logger_newline(console_only=True)
        scores, decisions = self.membership_scores(np.concatenate(target_scores), np.concatenate(predictions))
        stop = time.time()
        h, m, s = convert_to_hms(stop-start)
        self.logger.print_it('Quantile MIA attacker: score computation done! Time taken to compute: {}:{:02d}:{:02d}...'.format(h, m, s))
        return [MembershipScores(scores, decisions=decisions)]

    def membership_scores(self, target_scores, predictions):
        """Scores, higher for members, and decisions at 1 - alpha from the target's scores and the model's outputs.

        The paper's rule: a member when the target's score exceeds its predicted (1 - alpha) quantile.
        """
        target_scores = np.asarray(target_scores, dtype=np.float64).reshape(-1)
        predictions = np.asarray(predictions, dtype=np.float64)
        alpha = self.attack_configs.quantile_alpha
        if self.attack_configs.use_gaussian:
            # A finite log_std can overflow exp(-log_std), even in a poorly calibrated but otherwise finite model.
            scores = gaussian_z_scores(target_scores, predictions)
            # The (1 - alpha) quantile of N(mu, sigma) is mu + sigma * Phi^-1(1 - alpha), so the z-score must exceed
            # Phi^-1(1 - alpha), on the same signed-log scale as the scores.
            z_alpha = ndtri(1 - alpha)
            return scores, scores > np.sign(z_alpha) * np.log1p(np.abs(z_alpha))
        # As the authors do, each sample's thresholds are sorted, so that they increase with the level.
        decision_index = int(torch.argmin(torch.abs(self.quantile.reshape(-1).cpu() - (1 - alpha))))
        scores = target_scores - np.sort(predictions, axis=1)[:, decision_index]
        return scores, scores > 0
    

def paper_train_config(train_config):
    """A copy of the attacker's train configs with the authors' recipe for the regression model."""
    config = copy.deepcopy(train_config)
    config.optimizer_config = OptimizerConfigs(**QUANTILE_PAPER_OPTIMIZER)
    config.scheduler_config = SchedulerConfigs(name='const', epochs=QUANTILE_PAPER_EPOCHS)
    config.batch_size = QUANTILE_PAPER_BATCH_SIZE
    config.grad_clip = QUANTILE_PAPER_GRAD_CLIP
    return config


def quantile_levels(n: int, low: float, high: float, use_logscale: bool) -> torch.Tensor:
    """The pinball head's n levels from low to high, evenly spaced or, on a log scale, crowded near 1.

    The log scale spaces 1 - level evenly in log, as the authors' 1 - logspace grid does: small alphas need levels
    close to 1.
    """
    if not use_logscale:
        return torch.linspace(low, high, n)
    if not (0 <= low and high < 1):
        raise ValueError(f"A log-scale grid needs 0 <= --attacker_quantile_low and --attacker_quantile_high < 1, got "
                         f"{low} and {high}.")
    return torch.sort(1 - torch.logspace(np.log10(1 - high), np.log10(1 - low), n))[0]


def pinball_levels(n: int, low: float, high: float, use_logscale: bool, alpha: float) -> torch.Tensor:
    """The pinball head's levels: the grid of the options, the levels near 1 (QUANTILE_TAIL_LEVELS) and 1 - alpha.

    1 - alpha is the level the attack decides at, added so that it uses the alpha it is given, not the nearest level.
    """
    levels = quantile_levels(n=n, low=low, high=high, use_logscale=use_logscale)
    tail = 1 - torch.logspace(np.log10(1 - QUANTILE_TAIL_LOW), np.log10(1 - QUANTILE_TAIL_HIGH), QUANTILE_TAIL_LEVELS)
    for level in tail.tolist() + [1 - alpha]:
        levels = with_level(levels, level)
    return levels


def with_level(levels: torch.Tensor, level: float) -> torch.Tensor:
    """The levels with `level` added in sorted order, unless one of them already equals it."""
    level = torch.tensor([level], dtype=levels.dtype)
    if torch.isclose(levels, level, rtol=0, atol=1e-6).any():
        return levels
    return torch.sort(torch.cat([levels, level]))[0]


def validation_split(n_samples: int, val_split: float, seed: int):
    """Split positions 0..n_samples-1 at random into a training part and a validation part of val_split of them."""
    if not 0 < val_split < 1:
        raise ValueError(f"The validation share must lie strictly between 0 and 1, got {val_split}.")
    n_val = int(round(val_split * n_samples))
    if not 0 < n_val < n_samples:
        raise ValueError(f"A validation share of {val_split} of {n_samples} samples leaves one part empty.")
    positions = np.random.default_rng(seed).permutation(n_samples)
    return np.sort(positions[n_val:]), np.sort(positions[:n_val])


def gaussian_z_scores(target_scores, predictions):
    """Score each target by its z-score under the Gaussian head, z = (s - mu) / sigma, as sign(z) * log1p(|z|).

    Ranking by z is the paper's ROC: a sample is a member at level 1 - alpha when z exceeds Phi^-1(1 - alpha). The
    signed log is strictly increasing, so it keeps that ranking, and it is computed from log_std without forming
    sigma, so that no predicted scale overflows.
    """
    targets = np.asarray(target_scores, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (len(targets), 2):
        raise ValueError("Gaussian predictions must contain one (mu, log_std) pair per target")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Gaussian targets and predictions must be finite")
    residual = targets - predictions[:, 0]
    with np.errstate(divide="ignore"):
        log_abs_z = np.log(np.abs(residual)) - predictions[:, 1]
    return np.sign(residual) * np.logaddexp(0., log_abs_z)


def gaussian_margin_scores(target_scores, predictions, quantile):
    """Compute signed-log Gaussian quantile margins without exponentiating log_std.

    For margin s = target - (mu + exp(log_std) * Phi^-1(quantile)), return
    sign(s) * log1p(abs(s)). This strictly increasing transform keeps ROC ranking,
    membership orientation and the zero-threshold decision. In particular, it
    does not clip large margins together or change the predicted variance.
    """
    targets = np.asarray(target_scores, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != (len(targets), 2):
        raise ValueError("Gaussian predictions must contain one (mu, log_std) pair per target")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Gaussian targets and predictions must be finite")
    if not np.isfinite(quantile) or not 0 < quantile < 1:
        raise ValueError("Gaussian quantiles must lie strictly between 0 and 1")

    residual = targets - predictions[:, 0]
    z = ndtri(quantile)
    with np.errstate(divide="ignore"):
        log_residual = np.log(np.abs(residual))
        log_scaled_z = predictions[:, 1] + np.log(np.abs(z))
    log_magnitude, sign = logsumexp(
        np.stack((log_residual, log_scaled_z)), axis=0,
        b=np.stack((np.sign(residual), np.full_like(residual, -np.sign(z)))),
        return_sign=True,
    )
    scores = sign * np.logaddexp(0., log_magnitude)
    if not np.isfinite(scores).all():
        raise ValueError("Gaussian margin computation produced non-finite scores")
    return scores


# Custom loss functions
class PinballLoss(torch.nn.Module):
    """Pinball loss of each level, the quantile regression loss: its minimum is the level's quantile."""
    def __init__(self, quantile: torch.Tensor):
        """Keep the levels, one per output of the model, as a (1, levels) tensor."""
        super(PinballLoss, self).__init__()
        self.quantile = quantile
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """Loss of each sample and level: level x (target - output) when the target is above, else (1 - level) x
        (output - target)."""
        target = target.reshape([-1, 1])
        delta_score = target - input
        loss = torch.nn.functional.relu(delta_score) * self.quantile + torch.nn.functional.relu(-delta_score) * (1.0 - self.quantile)
        return loss


class GaussianLoss(torch.nn.Module):
    """Negative log-likelihood of a Gaussian, up to a constant: the model predicts a mean and a log scale per sample."""
    def __init__(self):
        """No parameters."""
        super(GaussianLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """log sigma + (target - mu)^2 / (2 sigma^2) for each sample, from the outputs (mu, log sigma)."""
        mu = input[:, 0]
        log_std = input[:, 1]
        loss = log_std + 0.5 * torch.exp(-2 * log_std) * (target - mu) ** 2
        return loss
