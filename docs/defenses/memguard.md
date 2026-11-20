# MemGuard

MemGuard ([Jia et al., CCS 2019](https://arxiv.org/abs/1909.10594)) adds noise to each output so that a membership
classifier cannot tell members from non-members, without changing the predicted label. The defender trains its own
membership classifier, the defense classifier, on the model's outputs on training members and validation samples. For
each query, it finds noise that makes the defense classifier output 0.5, and adds it with a probability that keeps
the expected L1 distortion of the scores within a budget. Training is unchanged. The authors' code is
[here](https://github.com/jinyuan-jia/MemGuard).

Repeated queries get the same output. Each query runs an optimization, so queries are much slower than without a
defense.

## Usage

```bash
python run.py --defender_mode mem_guard --dataset cifar10 --attacker_mode neural_prob
```

`neural_feat` reads the model's internal features, which MemGuard does not change, so against MemGuard it measures the
undefended model.

The defense is {py:class}`~src.mia.defenses.post_hoc.mem_guard.MemGuardDefender`.

## Parameters

`--defender_mem_guard_budget` is the budget ε on the expected L1 distortion of the scores. The defense classifier has
the hidden layers of `--defender_mem_guard_shadow_model_layers`, and trains with SGD for
`--defender_mem_guard_shadow_model_epochs` epochs at learning rate `--defender_mem_guard_shadow_model_lr`, as in the
paper. Before a query's random draw, each input value is rounded to a multiple of
`--defender_mem_guard_randomness_quantization`, so inputs that round to the same values share the draw.

The noise search uses the settings of the paper and the authors' code:

```{include} ../_generated/memguard_parameters.md
```

```{include} ../_generated/options_memguard.md
```
