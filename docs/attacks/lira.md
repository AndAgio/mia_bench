# LiRA

LiRA ([Carlini et al., IEEE S&P 2022](https://arxiv.org/abs/2112.03570)) tests each sample against shadow models. It
scales the target's probability p of the sample's true class as φ = log(p / (1 − p)), which is close to Gaussian
across models, and compares the target's φ with Gaussians fitted to the shadow models' φ on the same sample. The
authors' code is [here](https://github.com/tensorflow/privacy/tree/master/research/mi_lira_2021).

The attack has two modes, which differ in how the shadow models see the audited samples:

- `lira` (also `online_lira`) trains every audited sample into half of the shadow models. It fits one Gaussian to the
  models trained on the sample (IN) and one to the others (OUT), and scores the log likelihood ratio of the target's φ
  under the two. It needs at least 4 shadow models, so that both Gaussians have two values.
- `offline_lira` trains the shadow models without any audited sample, so they can be trained before the audit. It fits
  only the OUT Gaussian and scores the one-sided test of the paper's Eq. 4: the further the target's φ lies above the
  OUT mean, the likelier the sample is a member. It needs at least 2 shadow models.

The cost is mostly the training of the `--n_shadows` shadow models, with the attacker's training options (see
[Training defaults](../reference/training.md)). Scoring then queries every shadow model and the target once for each
audited sample.

Each Gaussian's variance is estimated for each sample, as in the paper's main attack. The paper finds that with fewer
than 64 shadow models, a variance shared by all samples gives a stronger attack (Sec. VI-B). Here the attack stays the
same whatever their number, so more shadow models make it stronger rather than different.

LiRA has no decision rule of its own, since the paper reports ROC curves only. Its decisions come from the
calibrated rule, the default for every attack, also with `--decision_rule attack` (see
[Decision rules](decision_rules.md)).

## Usage

```bash
python run.py --attacker_mode lira --dataset cifar10 --defender_mode none
```

The attack is {py:class}`~src.mia.attacks.black_box.lira.LiRA`.

## Parameters

`--n_shadows` is the number of shadow models, shared with the other attacks that train several.

```{include} ../_generated/options_lira.md
```

## Differences from the paper

- Each sample is queried once. The authors also query its mirror image and average the test over both queries, which
  helps because their models train with flips; here the target and shadow models train without augmentation unless a
  defense adds it.
