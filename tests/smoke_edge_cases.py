"""Five small, valid runtime cases per smoke-matrix component.

This module contains only case definitions; it imports no training dependencies.
Aliases use their canonical component's cases in smoke_test_all.py.
"""
from dataclasses import dataclass


EDGE_NAMES = ("baseline", "minimal", "uneven", "strong", "alternate")


@dataclass(frozen=True)
class EdgeSpec:
    description: str
    args: tuple[str, ...] = ()


def spec(description, **options):
    args = []
    for name, value in options.items():
        if value is False:
            continue
        args.append("--" + name)
        if value is not True:
            args.extend(str(v) for v in (value if isinstance(value, (tuple, list)) else (value,)))
    return EdgeSpec(description, tuple(args))


def attack_specs(component):
    baseline = spec("Baseline small workload")
    if component in ("online_robust", "offline_robust"):
        online = component.startswith("online")
        return (baseline,
                spec("Smallest in/out reference ensemble and one population point",
                     n_shadows=2 if online else 1, attacker_robust_rand_pop_size=1),
                spec("Odd batches, multiple alpha values and a strict likelihood ratio",
                     attacker_robust_alphas=(.01, .5, 1.), attacker_robust_gamma=2.),
                spec("Larger reference population and permissive likelihood ratio",
                     attacker_robust_rand_pop_size=32, attacker_robust_gamma=.1),
                spec("Six reference models and alternate shadow optimizer",
                     n_shadows=6, attacker_optimizer="adam", attacker_lr=.001))
    if component == "lira":
        return (baseline,
                spec("Two shadows: one in and one out per audit point", n_shadows=2),
                spec("Odd audit prior and shadow batch boundaries", n_shadows=4),
                spec("Six shadows for nontrivial in/out variance", n_shadows=6),
                spec("Alternate shadow optimizer and one full shadow batch",
                     attacker_optimizer="adam", attacker_lr=.001))
    if component == "quantile":
        return (baseline,
                spec("Single median quantile", attacker_quantile_n=1,
                     attacker_quantile_low=.5, attacker_quantile_high=.5, attacker_quantile_alpha=.5),
                spec("Log-spaced tail quantiles", attacker_quantile_use_logscale=True,
                     attacker_quantile_n=5, attacker_quantile_low=.001, attacker_quantile_high=.999),
                spec("Gaussian mean/variance head", attacker_quantile_use_gaussian=True,
                     attacker_quantile_n=5, attacker_quantile_low=.01, attacker_quantile_high=.99),
                spec("Gaussian head with log-spaced quantiles and full batches",
                     attacker_quantile_use_gaussian=True, attacker_quantile_use_logscale=True,
                     attacker_quantile_n=7, attacker_quantile_low=.001, attacker_quantile_high=.999,
                     attacker_lr=.001))
    if component.startswith("neural_"):
        return (baseline,
                spec("Narrowest hidden layer and two audit points", attacker_neural_model_layers=(1,)),
                spec("Two hidden layers and uneven batches", attacker_neural_model_layers=(8, 4)),
                spec("Wider/deeper attack network with member-heavy audit prior",
                     attacker_neural_model_layers=(32, 16, 8), attacker_neural_model_lr=.001),
                spec("Alternate shadow optimizer and full batches",
                     attacker_optimizer="adam", attacker_lr=.001, attacker_neural_model_lr=.05))
    if component.startswith(("pmia_", "rmia_")):
        prefix = "attacker_pmia_alpha" if component.startswith("pmia_") else "attacker_rmia_alpha"
        return (baseline,
                spec("Lower endpoint quantile; single reference model/dataset", **{prefix: 0., "n_shadows": 1}),
                spec("Lower-tail quantile with uneven audit prior", **{prefix: .001}),
                spec("Upper-tail quantile with member-heavy audit prior", **{prefix: .999}),
                spec("Upper endpoint quantile and full batches", **{prefix: 1., "n_shadows": 3}))
    if component.startswith(("sba_", "uba_")):
        unsupervised = component.startswith("uba_")
        minimum = 2 if component.endswith("-pca") else 1
        return (baseline,
                spec("Reduced query budget and minimum calibration population",
                     attacker_boundary_n_queries=128, attacker_boundary_calibration_samples=minimum),
                spec("L-infinity distance and low-dimensional search",
                     attacker_boundary_norm="linf", attacker_boundary_qeba_reduction_factor=16,
                     **({"attacker_boundary_quantile": 0.} if unsupervised else {})),
                spec("Higher query budget and upper calibration quantile",
                     attacker_boundary_n_queries=512, attacker_boundary_qeba_reduction_factor=2,
                     **({"attacker_boundary_quantile": 1.} if unsupervised else {})),
                spec("Odd calibration population and non-power-of-two reduction",
                     attacker_boundary_calibration_samples=7, attacker_boundary_qeba_reduction_factor=3))
    if component == "noise_robustness":
        return (baseline,
                spec("One query and zero noise", attacker_noise_robust_n_queries=1,
                     attacker_noise_robust_sigmas=(0.,)),
                spec("Multiple sigmas including zero", attacker_noise_robust_n_queries=5,
                     attacker_noise_robust_sigmas=(0., .01, .2)),
                spec("Large input noise", attacker_noise_robust_n_queries=7,
                     attacker_noise_robust_sigmas=(1.,)),
                spec("Repeated sigma values and alternate shadow optimizer",
                     attacker_noise_robust_sigmas=(.05, .05), attacker_optimizer="adam", attacker_lr=.001))
    if component.startswith("transfer_"):
        return (baseline,
                spec("Two audit points and tiny surrogate batches", attacker_batch_size=2),
                spec("Relabeling with incomplete surrogate batches", attacker_batch_size=7),
                spec("Different surrogate optimizer", attacker_optimizer="adam", attacker_lr=.001),
                spec("One full surrogate batch with momentum and decay disabled",
                     attacker_momentum=0., attacker_weight_decay=0.))
    if component.startswith("oslo_"):
        return (baseline,
                spec("Zero perturbation budget", attacker_oslo_max_epsilon=0.),
                spec("Two source models and one validation model", attacker_oslo_n_models=3,
                     attacker_oslo_source_models_ratio=2 / 3, attacker_oslo_threshold=0.),
                spec("One source and two validation models with multiple attack steps",
                     attacker_oslo_n_models=3, attacker_oslo_source_models_ratio=1 / 3,
                     attacker_oslo_K=2, attacker_oslo_N=2, attacker_oslo_max_epsilon=.05),
                spec("Early-stopping endpoint threshold", attacker_oslo_threshold=1.,
                     attacker_oslo_K=2, attacker_oslo_N=2))
    if component.startswith("dh_"):
        return (baseline,
                spec("Single surrogate and one inference query", attacker_dh_n_models=1, attacker_dh_n_queries=1),
                spec("Three surrogates and odd query budget", attacker_dh_n_models=3, attacker_dh_n_queries=5),
                spec("Larger query budget", attacker_dh_n_queries=16),
                spec("Alternate distilled-surrogate optimizer", attacker_optimizer="adam", attacker_lr=.001))
    if component in ("online_yoqo", "offline_yoqo"):
        return (baseline,
                spec("Smallest reference ensemble and no perturbation penalty",
                     n_shadows=2 if component.startswith("online") else 1, attacker_yoqo_gamma=0.),
                spec("Zero out-model weight and no early stopping", attacker_yoqo_alpha=0.,
                     attacker_yoqo_adv_opt_loss_threshold=0.),
                spec("Strong input penalty and multiple optimization steps", attacker_yoqo_gamma=10.,
                     attacker_yoqo_adv_opt_max_iter=3, attacker_yoqo_adv_opt_lr=.001),
                spec("Early stopping with full surrogate batches",
                     attacker_yoqo_adv_opt_loss_threshold=100., attacker_yoqo_alpha=2.))
    raise ValueError(f"No edge cases defined for attack {component}")


def defense_specs(component, epochs):
    baseline = spec("Baseline small workload")
    if component in ("none", "hamp_test"):
        return (baseline,
                spec("Tiny training batches, no momentum or weight decay",
                     defender_momentum=0., defender_weight_decay=0.),
                spec("Odd training batches and Adam optimizer", defender_optimizer="adam", defender_lr=.001),
                spec("Larger training batches and low learning rate", defender_lr=.001),
                spec("One full training batch", defender_batch_size=256))
    if component == "dp":
        return (baseline,
                spec("Hooks backend with zero noise", defender_dp_grad_sample_mode="hooks",
                     defender_dp_noise_multiplier=0.),
                spec("Hooks backend with per-layer clipping", defender_dp_grad_sample_mode="hooks",
                     defender_dp_clip_per_layer=True, defender_dp_max_grad_norm=.1),
                spec("Ghost clipping with stronger noise", defender_dp_noise_multiplier=2.,
                     defender_dp_max_grad_norm=.5),
                spec("Hooks backend with loose gradient clipping", defender_dp_grad_sample_mode="hooks",
                     defender_dp_max_grad_norm=10.))
    if component == "mem_guard":
        return (baseline,
                spec("Zero perturbation budget", defender_mem_guard_budget=0.),
                spec("Tiny budget and narrow membership classifier", defender_mem_guard_budget=1e-6,
                     defender_mem_guard_shadow_model_layers=(1,)),
                spec("Large perturbation budget", defender_mem_guard_budget=1.),
                spec("Coarse query quantization and deeper membership classifier",
                     defender_mem_guard_randomness_quantization=.1,
                     defender_mem_guard_shadow_model_layers=(16, 8)))
    if component == "relax_loss":
        return (baseline,
                spec("Zero target loss: descent branch", defender_relax_loss_alpha=0.),
                spec("High target loss: ascent and flatten-all branches", defender_relax_loss_alpha=10.,
                     defender_relax_loss_flatten="all", defender_relax_loss_upper=.2),
                spec("High target loss: flatten-incorrect branch", defender_relax_loss_alpha=10.,
                     defender_relax_loss_flatten="incorrect", defender_relax_loss_upper=1.),
                spec("SAM closure with posterior flattening", defender_optimizer="sam",
                     defender_relax_loss_alpha=10., defender_relax_loss_flatten="all"))
    if component == "adv_reg":
        return (baseline,
                spec("Zero membership regularization", defender_adv_reg_lambda=0.),
                spec("Multiple adversary steps per target step", defender_adv_reg_shadow_attacker_k=2),
                spec("Strong membership regularization", defender_adv_reg_lambda=5.),
                spec("Narrow adversary and alternate target optimizer",
                     defender_adv_reg_shadow_attacker_model_layers=(1,), defender_optimizer="adam", defender_lr=.001))
    if component == "mixup":
        return (baseline,
                spec("Zero mixup strength", defender_mixup_alpha=0.),
                spec("Near-endpoint beta draws", defender_mixup_alpha=.01),
                spec("Concentrated beta draws", defender_mixup_alpha=10.),
                spec("SAM optimizer closure", defender_optimizer="sam", defender_mixup_alpha=.5))
    if component in ("hamp_train", "hamp_full"):
        return (baseline,
                spec("Zero entropy target and zero entropy regularization",
                     defender_hamp_gamma=0., defender_hamp_alpha=0.),
                spec("Maximum entropy target", defender_hamp_gamma=1., defender_hamp_alpha=.01),
                spec("Strong entropy regularization", defender_hamp_gamma=.5, defender_hamp_alpha=1.),
                spec("SAM closure with almost-uniform soft labels", defender_optimizer="sam",
                     defender_hamp_gamma=.99, defender_hamp_alpha=.01))
    if component == "selena":
        return (baseline,
                spec("Smallest nontrivial teacher ensemble", defender_selena_K=2, defender_selena_L=1),
                spec("One excluded teacher out of three", defender_selena_K=3, defender_selena_L=1),
                spec("All but one teacher excluded per example", defender_selena_K=3, defender_selena_L=2),
                spec("Even teacher/exclusion split", defender_selena_K=4, defender_selena_L=2))
    if component in ("mist", "mist_mixup"):
        mix = component == "mist_mixup"
        return (baseline,
                spec("Zero cross-difference penalty", defender_mist_lambda=0.,
                     **({"defender_mixup_alpha": 0.} if mix else {})),
                spec("Stratified partitions", defender_mist_split_method="stratified",
                     **({"defender_mixup_alpha": .01} if mix else {})),
                spec("Three partitions and stronger cross-difference penalty", defender_mist_num_submodels=3,
                     defender_mist_lambda=5., **({"defender_mixup_alpha": 10.} if mix else {})),
                spec("Two inner passes and stratified partitions", defender_mist_split_method="stratified",
                     defender_mist_submodel_epochs=2))
    if component == "weighted_smoothing":
        return (baseline,
                spec("Zero prediction noise", defender_weighted_smoothing_sigma_noise=0.),
                spec("Warmup-to-smoothing transition", defender_weighted_smoothing_warmup_epochs=min(1, epochs - 1)),
                spec("Large prediction noise", defender_weighted_smoothing_sigma_noise=1.),
                spec("Final-epoch smoothing transition", defender_weighted_smoothing_warmup_epochs=epochs - 1))
    if component == "purifier":
        return (baseline,
                spec("One-dimensional latent and zero latent noise", defender_purifier_reformer_latent_dim=1,
                     defender_purifier_reformer_noise_sigma=0., defender_purifier_reformer_batch_size=2),
                spec("Zero classification penalty and exact swap threshold", defender_purifier_reformer_lambda=0.,
                     defender_purifier_swap_threshold=0., defender_purifier_reformer_batch_size=5),
                spec("Larger latent noise and classification penalty", defender_purifier_reformer_noise_sigma=2.,
                     defender_purifier_reformer_lambda=1.),
                spec("Single full reformer batch and coarse query quantization",
                     defender_purifier_reformer_batch_size=256, defender_purifier_noise_quantization=.1))
    if component in ("mmd", "mmd_mixup"):
        mix = component == "mmd_mixup"
        return (baseline,
                spec("Zero MMD penalty", defender_mmd_lambda=0., **({"defender_mixup_alpha": 0.} if mix else {})),
                spec("Weak MMD penalty and uneven batches", defender_mmd_lambda=.001,
                     **({"defender_mixup_alpha": .01} if mix else {})),
                spec("Strong MMD penalty", defender_mmd_lambda=10.,
                     **({"defender_mixup_alpha": 10.} if mix else {})),
                spec("SAM closure with MMD and reference-data reuse", defender_optimizer="sam"))
    if component == "ldl":
        return (baseline,
                spec("One query and zero Gaussian/bit-flip noise", defender_ldl_n_queries=1,
                     defender_ldl_noise_scale=0., defender_ldl_flip_prob=0.),
                spec("Odd query count and tiny Gaussian/bit-flip noise", defender_ldl_n_queries=5,
                     defender_ldl_noise_scale=1e-6, defender_ldl_flip_prob=1e-6),
                spec("Large Gaussian noise / always flip binary features", defender_ldl_n_queries=7,
                     defender_ldl_noise_scale=.5, defender_ldl_flip_prob=1.),
                spec("More noisy queries / balanced binary flips", defender_ldl_n_queries=16,
                     defender_ldl_noise_scale=.1, defender_ldl_flip_prob=.5))
    if component == "data_augmentation":
        return (baseline,
                spec("All augmentation disabled", defender_augment_horizontal_flip_prob=0.,
                     defender_augment_rotation=0., defender_augment_erase_prob=0.),
                spec("Always flip", defender_augment_horizontal_flip_prob=1.,
                     defender_augment_rotation=0., defender_augment_erase_prob=0.),
                spec("Large rotations and always erase", defender_augment_rotation=90., defender_augment_erase_prob=1.),
                spec("Augmentation through a SAM closure", defender_optimizer="sam"))
    raise ValueError(f"No edge cases defined for defense {component}")


def edge_spec(kind, component, name, epochs=5):
    index = EDGE_NAMES.index(name)
    cases = attack_specs(component) if kind == "attack" else defense_specs(component, epochs)
    if len(cases) != 5:
        raise ValueError(f"Expected five edge cases for {kind}:{component}")
    case = cases[index]
    if kind == "attack":
        shared = ({}, {"n_auditing_samples": 2, "attacker_batch_size": 2},
                  {"audit_in_perc": .25, "attacker_batch_size": 7},
                  {"audit_in_perc": .75, "attacker_batch_size": 32},
                  {"n_auditing_samples": 10, "attacker_batch_size": 256})[index]
    else:
        shared = ({}, {"defender_batch_size": 2}, {"defender_batch_size": 7},
                  {"defender_batch_size": 32}, {})[index]
    return EdgeSpec(case.description, spec("", **shared).args + case.args)
