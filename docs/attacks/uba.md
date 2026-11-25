# Unsupervised boundary distance

The boundary attack of [Li and Zhang, CCS 2021](https://arxiv.org/abs/2007.15528) is label-only: it reads only the
labels the target predicts. For each sample, it estimates the L2 distance to the target's decision boundary, the
smallest change to the input that moves the predicted label away from the sample's label. A misclassified sample has
distance 0. Training samples tend to sit further from the boundary, so a sample is called a member when its distance is
above a threshold. The authors' code is [here](https://github.com/zhenglisec/Decision-based-MIA).

Unlike the [supervised boundary attack](sba.md), this one needs no data and trains no shadow model. Its threshold comes
from random inputs, as in the paper (Sec. 4.4, Alg. 3). The attack draws {{uba_calibration_samples}} of them and
searches each one's distance to the boundary. A random input belongs to no class, so its search starts from the label
the target predicts for it. The threshold is the {{uba_quantile}} quantile of these distances, set by
`--attacker_boundary_quantile`. The paper takes the "top t percentile" instead: a quantile q is its t = 100 × (1 − q).
The paper finds that t from 30% to 80% works (Fig. 11), so q from 0.2 to 0.7. In a test with a ResNet-18 trained on
5,000 CIFAR-10 images, with 100 members and 100 non-members, the quantiles 0.2, 0.5 and 0.7 gave 78.5%, 82.0% and 85.0%
accuracy, against 89.5% for the best possible threshold.

The random inputs depend on the data:

- images have uniform random pixels, as in the paper;
- Purchase and Texas get random bits, so that each input is a valid record;
- News gets uniform values in [0, 1] for every feature. The paper has no text dataset, and these inputs are unlike
  real documents, whose TF-IDF rows are sparse with unit norm. In a test with a tabular MLP trained on 5,000 News
  documents, the thresholds at the quantiles 0.2 to 0.7 (163 to 298) were above every document's distance (at most
  45), so the attack's own decisions called every document a non-member. Random sparse rows of unit norm missed the
  other way, calling 93% of the documents members. The scores, and so the AUC (0.71 there), do not depend on the
  threshold. On News, use `--decision_rule calibrated` for decisions.

The threshold only sets the attack's own decisions (`--decision_rule attack`), which call a sample a member when its
distance is strictly above it, as in the paper (Alg. 2). The scores, and the calibrated and oracle rules, do not depend
on it.

The mode sets how the distance is searched for:

- `uba` uses HopSkipJump ([Chen et al., IEEE S&P 2020](https://arxiv.org/abs/1904.02144)), as in the paper.
- `uba_qeba` (also `uba_qeba-spatial`) uses QEBA ([Li et al., CVPR 2020](https://arxiv.org/abs/2005.14137)), the
  paper's other search, which estimates the boundary's direction from random changes drawn in a smaller subspace of
  the image. The subspace is spatial, low-resolution noise upsampled to the image size, as in the authors' code. The
  default `--attacker_boundary_qeba_reduction_factor` is the authors': on CIFAR-10, it keeps 768 of the 3,072
  dimensions.
- `uba_qeba-dct` uses QEBA's low-frequency subspace instead, and `uba_qeba-custom` one indicator per channel and block
  of pixels, which is not in the paper.

QEBA's third subspace, built from the gradients of models the attacker owns, has no mode here, since this attacker has
neither data nor models (the [supervised boundary attack](sba.md) has one). On tabular data, QEBA's subspaces do not
apply: the QEBA modes search as HopSkipJump does, and `uba_qeba-custom` stops with an error. Each of QEBA's estimates
spends as many queries as HopSkipJump's do, where the authors' code spends 30. In the test above, 30 per estimate gave
an AUC of 0.907 against 0.889, a difference within the noise of 200 samples, for three times the time: the budget
stays the same, but more of it goes to queries sent one at a time.

Images are searched as pixels in [0, 1], as in the authors' code, so every query is an image the target could be sent,
and the distances are in pixels. Tabular inputs are searched as they are, with no bounds.

The cost is the boundary search: up to `--attacker_boundary_n_queries` queries per sample, {{boundary_n_queries}}
unless set. The paper's main results use 15,000 queries for HopSkipJump and 7,000 for QEBA (Sec. 4.3), but its AUC
stops improving after about 2,500 (Fig. 9). In the test above, HopSkipJump gave an AUC of 0.857 with 1,000 queries and
0.893 with 2,500, at about 1 and 2 seconds per correctly classified sample on a laptop GPU. If the AUC is lower than you
expect, raise `--attacker_boundary_n_queries` first: the time grows about as fast as the budget. A misclassified sample
costs one query. With the default budget, the random inputs take a few minutes, then each audited or calibration
sample about 2 seconds.

## Usage

```bash
python run.py --attacker_mode uba --dataset cifar10 --defender_mode none
```

The attack is {py:class}`~src.mia.attacks.label_only.unsupervised_boundary.UnsupervisedBoundaryMIA`.

## Parameters

`--attacker_boundary_norm linf` searches for the smallest change in the largest pixel instead; the paper uses L2.
`--attacker_boundary_qeba_reduction_factor` only applies to the QEBA modes.

```{include} ../_generated/options_uba.md
```
