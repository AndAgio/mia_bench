# MemGuard

MemGuard ([Jia et al., CCS 2019](https://arxiv.org/abs/1909.10594)) adds noise to each output so that a membership
classifier cannot tell members from non-members, without changing the predicted label. The defender trains its own
membership classifier, the defense classifier. For each query, Phase I finds noise that makes the defense classifier
output 0.5, as an adversarial example against it. Phase II adds that noise with a probability chosen so that the
expected L1 distortion of the scores stays within a budget. Training is unchanged. The authors' code is
[here](https://github.com/jinyuan-jia/MemGuard).

The defense classifier trains on the sorted confidences of training members and of validation samples, which play
the non-members. The non-members that the attacks audit, in the test split, are never used. Phase II's random draw is
seeded by a hash of the query, so repeated queries get the same output, as in the paper's one-time randomness. Queries
return the defended probabilities as log-probabilities, so an attack's softmax gives them back exactly. Phase I runs
an optimization on every query, so queries are much slower than without a defense.

## Usage

```bash
python run.py --defender_mode mem_guard --dataset cifar10 --attacker_mode neural_prob
```

The defense is {py:class}`~src.mia.defenses.post_hoc.mem_guard.MemGuardDefender`.

## Parameters

`--defender_mem_guard_budget` is the budget ε on the expected L1 distortion of the scores. The defense classifier has
the hidden layers of `--defender_mem_guard_shadow_model_layers`, and trains with SGD for
`--defender_mem_guard_shadow_model_epochs` epochs at learning rate `--defender_mem_guard_shadow_model_lr`, as in the
paper. Before hashing a query, each input value is rounded to a multiple of
`--defender_mem_guard_randomness_quantization`, in the units of the model's inputs, so inputs that round to the same
values share the random draw.

Phase I uses the settings of the paper and the authors' code:

```{include} ../_generated/memguard_parameters.md
```

```{include} ../_generated/options_memguard.md
```

## Differences from the paper

- The paper quantizes each query before hashing it, but gives no width. The width is
  `--defender_mem_guard_randomness_quantization`.
- The defense classifier learns from as many training members as validation samples, picked at random, which keeps
  the balance of members and non-members that the paper uses.
- Algorithm 1 searches ever larger c3 until it fails. As in the authors' code, we stop at the largest c3 in the table
  above.
- As printed, Algorithm 1 resets e′ to 0 for each value of c3, so it would always return no noise. As in the authors'
  code, a failed search returns the last noise that succeeded.
- The paper only considers attacks on the outputs. `neural_feat` reads the trained model's features, which MemGuard
  does not change, so it measures the undefended model: what it leaks to an attacker who can read its features.
