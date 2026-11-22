# RMIA

RMIA ([Zarifzadeh, Liu and Shokri, ICML 2024](https://arxiv.org/abs/2312.03262)) compares each sample with random
population samples. For a sample x, it computes the likelihood ratio Pr(x|θ) / Pr(x): Pr(x|θ) is the target's
probability of x's true class, and Pr(x) the same probability averaged over reference models, the attacker's shadow
models. A member tends to have a high ratio, since the target fits it better than models that did not train on it.
x's score is the share of population samples z that it dominates, those whose ratio x's exceeds by more than a factor
γ. The authors' code is [here](https://github.com/privacytrustlab/ml_privacy_meter).

The attack has two modes, which differ in how the shadow models see the audited samples:

- `online_robust` trains every audited sample into half of the shadow models. Pr(x) averages the models trained on x
  (IN) and those not trained on it (OUT). It needs at least 2 shadow models.
- `offline_robust` trains the shadow models without any audited sample, and approximates the missing IN models from
  the OUT ones: Pr(x)_IN ≈ a Pr(x)_OUT + 1 − a.

The cost is mostly the training of the `--n_shadows` shadow models, with the attacker's training options (see
[Training defaults](../reference/training.md)). Scoring then queries every shadow model and the target once for each
audited sample and each population sample.

The population is `--attacker_robust_rand_pop_size` samples of the attacker's data, drawn once and shared by every
value of a and by the auditing and calibration samples. It leaves out the samples whose Pr(z) cannot be computed like
Pr(x): offline, those that every shadow model trained on, and online, also those that none trained on. A score is a
share of the population, in steps of 1 / `--attacker_robust_rand_pop_size`, so a larger population separates samples
more finely, at the cost of more queries.

RMIA has no decision rule of its own, since the paper reports ROC curves only: with `--decision_rule attack`, its
decisions come from the calibrated threshold instead (see `--calibration_fpr` in
[Command-line options](../reference/cli.md)).

## Usage

```bash
python run.py --attacker_mode offline_robust --dataset cifar10 --defender_mode none
```

The attack is {py:class}`~src.mia.attacks.black_box.robust.RobustMIA`.

## Parameters

`--attacker_robust_a` is a, which only offline mode uses. When left unset, it takes the paper's value for the dataset
(App. B.2.2):

```{include} ../_generated/robust_parameters.md
```

Other datasets have no default, so an offline run on them needs `--attacker_robust_a`. The paper picks a by attacking
one shadow model with the others as references and keeping the a with the best AUC; `tests/tune_robust_a.py` runs
this procedure on the attacker's own shadow models, with no target model:

```bash
python tests/tune_robust_a.py --dataset svhn --n_shadows 2 --device mps
```

A tuned a depends on how the shadow models are trained, so tune it with the training options you will attack with.
Several values of `--attacker_robust_a` give one result each, from the same shadow models.

`--attacker_robust_gamma` is γ, 2 as in the paper.

```{include} ../_generated/options_robust.md
```

## Differences from the paper

- The population is `--attacker_robust_rand_pop_size` samples of the attacker's data, rather than every non-member of
  the target, the paper's default (Sec. 5.6): a realistic attacker does not know all of the target's non-members.
