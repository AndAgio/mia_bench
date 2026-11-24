# Supervised boundary distance

The decision-boundary attack ([Choquette-Choo et al., ICML 2021](https://arxiv.org/abs/2007.14321)) is label-only: it
reads only the labels the target predicts. For each sample, it estimates the L2 distance to the target's decision
boundary, the smallest change to the input that changes the predicted label. A misclassified sample has distance 0.
Training samples tend to sit further from the boundary, so a sample is called a member when its distance is above a
threshold. The authors' code is [here](https://github.com/cchoquette/membership-inference).

The threshold is chosen on one shadow model, trained on the attacker's data with the attacker's training options (see
[Training defaults](../reference/training.md)). It is the distance that best separates the shadow model's members from
its non-members by accuracy, on {{sba_threshold_samples}} of each. The authors' code uses 2,500 of each. In a test on
CIFAR-10, with the logit margin standing in for the distance, a threshold chosen on 500 of each did as well on the target
as one chosen on 2,500 (94.0% accuracy against 94.1%), for a fifth of the boundary searches. One shadow model is used
whatever `--n_shadows`: the paper finds that more do not help (App. F, Fig. 7).

This works with any split of the data between the defender and the attacker (`--defender_data_split_perc` and
`--attacker_data_split_perc`). The shadow dataset takes {{shadow_dataset_share}} of the attacker's data, and its
non-members are the rest. When the rest holds fewer than {{sba_threshold_samples}} samples, the threshold is chosen on
all of them and on as many members, so both sides stay the same size, as in the paper. The threshold only sets the
attack's own decisions (`--decision_rule attack`): the scores, and the calibrated and oracle rules, do not depend on it.
In the same test, a threshold chosen on 100 of each still did almost as well (93.9%), while one chosen on 25 of each
lost about a point, and one chosen on 10 of each about four.

The mode sets how the distance is searched for:

- `sba` uses HopSkipJump ([Chen et al., IEEE S&P 2020](https://arxiv.org/abs/1904.02144)), as in the paper.
- `sba_qeba` (also `sba_qeba-spatial`), `sba_qeba-dct`, `sba_qeba-pca` and `sba_qeba-custom` use QEBA
  ([Li et al., CVPR 2020](https://arxiv.org/abs/2005.14137)), which searches in a lower-dimensional subspace of the
  input, as the [unsupervised boundary attack](https://dl.acm.org/doi/pdf/10.1145/3460120.3484575) does.

Images are searched as pixels in [0, 1], as in the paper, so every query is an image the target could be sent, and the
distances are in pixels. Tabular inputs are searched as they are, with no bounds.

The cost is the boundary search: up to `--attacker_boundary_n_queries` queries per sample, {{sba_n_queries}} unless
set, as in the paper. The attack searches the shadow model on the threshold samples, then the target on the audited and
calibration samples. With the default budget, a ResNet-18 on CIFAR-10 takes about 2 seconds per correctly classified
sample on a laptop GPU, so the threshold takes about half an hour, and each audited or calibration sample another
2 seconds.

The attack's own decisions call a sample a member when its distance is strictly above the threshold, as in the paper.

## Usage

```bash
python run.py --attacker_mode sba --dataset cifar10 --defender_mode none
```

The attack is {py:class}`~src.mia.attacks.label_only.supervised_boundary.SupervisedBoundaryMIA`.

## Parameters

`--attacker_boundary_norm linf` searches for the smallest change in the largest pixel instead; the paper uses L2.
`--attacker_boundary_qeba_reduction_factor` only applies to the QEBA modes.

```{include} ../_generated/options_sba.md
```
