# Noise robustness

The noise robustness attack ([Choquette-Choo et al., ICML 2021](https://arxiv.org/abs/2007.14321), Sec. 3.4) is
label-only: it reads only the labels the target predicts. It sends the target many noisy copies of a sample and scores
the sample by the share of copies the target still labels correctly. A misclassified sample scores 0. Training samples
tend to sit further from the decision boundary, so their copies keep the label more often, and a sample is called a
member when its score is above a threshold. The authors' code is
[here](https://github.com/cchoquette/membership-inference).

The noise depends on the data, as in the paper:

- Images get Gaussian noise on their pixels, and each copy is clipped back to [0, 1], as in the authors' code, so every
  query is an image the target could be sent.
- Purchase and Texas have binary features, so each feature of a copy is flipped with some probability instead.
- News, whose TF-IDF features are neither pixels nor bits, gets Gaussian noise on every feature.

The noise level and the threshold are chosen on one shadow model, trained on the attacker's data with the attacker's
training options (see [Training defaults](../reference/training.md)). Each level is tried on {{nr_threshold_samples}}
of the shadow model's members and as many of its non-members, with the threshold that best separates them by accuracy.
The level with the best accuracy is kept. One shadow model is used whatever `--n_shadows`, as in the paper. Unless
`--attacker_noise_robust_sigmas` is set, the levels are:

- images: standard deviations of {{nr_image_sigmas}} pixels, spanning the 0.02 to 0.1 of the authors' code;
- Purchase and Texas: flip probabilities that flip {{nr_flipped_features}} features of a record on average, as in the
  authors' code;
- News: standard deviations of {{nr_tabular_sigmas}}. The paper has no text dataset. In a test with a tabular MLP
  trained on 5,000 News documents, 0.005 separated members from non-members best (AUC 0.82); the image levels, from
  0.02 to 0.1, gave 0.74 down to 0.65.

The authors' code uses 2,500 members and 2,500 non-members. Each of them costs `--attacker_noise_robust_n_queries`
queries for every level, so the attack uses {{nr_threshold_samples}}: for the [supervised boundary attack](sba.md), as
many chose as good a threshold as 2,500 in a test on CIFAR-10. Here, though, these samples also choose the noise level,
which sets the scores and so the AUC. If the AUC is lower than you expect, raise `THRESHOLD_SAMPLES_PER_SIDE` in
`src/mia/attacks/label_only/noise_robustness.py`. When fewer samples lie outside the shadow dataset (the shadow
dataset takes {{shadow_dataset_share}} of the attacker's data), both sides take as many as there are.

The cost is the queries: `--attacker_noise_robust_n_queries` copies of each correctly classified sample, as in the
authors' code. The levels are tried on the shadow model only, once per run, and the target is queried at the chosen
level only. Copies are sent in batches, and each has its own noise, so batching changes only the speed. With the
defaults, a ResNet-18 on CIFAR-10 takes about 1.7 seconds per sample on a laptop GPU: up to about two and a half hours
to choose the level, then 1.7 seconds per audited or calibration sample.

The threshold only sets the attack's own decisions (`--decision_rule attack`), which call a sample a member when its
score is strictly above the threshold, as in the paper. The calibrated and oracle rules do not use it.

## Usage

```bash
python run.py --attacker_mode noise_robust --dataset cifar10 --defender_mode none
```

By default, the decisions come from the calibrated rule, as for every attack. For the paper's own decisions,
described above, add `--decision_rule attack` (see [Decision rules](decision_rules.md)).

The attack is {py:class}`~src.mia.attacks.label_only.noise_robustness.NoiseRobustnessMIA`.

## Parameters

`--attacker_noise_robust_sigmas` replaces the levels above. On Purchase and Texas they are flip probabilities, between
0 and 1.

```{include} ../_generated/options_noise_robustness.md
```
