# Attack-R and Attack-P

Attack-R and Attack-P ([Ye et al., CCS 2022](https://arxiv.org/abs/2111.09679)) test whether the target's loss on a
sample is low compared with a reference distribution of losses. They differ in where that distribution comes from:

- Attack-R (`rmia_*` modes) trains `--n_shadows` reference models on the attacker's data, none of them on an audited
  sample, and compares the target's loss on each sample with the reference models' losses on the same sample. Each
  sample thus gets its own threshold.
- Attack-P (`pmia_*` modes) trains no model. It compares the target's loss on each sample with the target's own losses
  on population data: one shadow dataset, {{shadow_dataset_share}} of the attacker's data, with no audited sample. All
  samples share the threshold.

The `rmia_*` modes run Attack-R, not [RMIA](robust.md). The authors' code is
[here](https://github.com/privacytrustlab/ml_privacy_meter).

A sample's score is −α*, where α* is the smallest false-positive rate at which the test calls it a member, computed as
in the authors' code for the paper's figures: the larger of the sample's rank among the reference losses and the tail
probability of a Gaussian fitted to them after a logit rescaling. The attack's own decisions call a sample a member
when α* is at most `--attacker_rmia_alpha` or `--attacker_pmia_alpha`. The share of non-members it then calls members
is close to α when the reference losses match the target's on non-members; for an exact false-positive rate, use
`--decision_rule calibrated`.

Attack-R costs the training of its reference models, with the attacker's training options (see
[Training defaults](../reference/training.md)), and one pass of each over the audited samples. Attack-P only queries
the target, on the population data and the audited samples.

## Usage

```bash
python run.py --attacker_mode rmia_loss --dataset cifar10 --defender_mode none
```

The suffix chooses the score the test compares:

- `_loss`: the cross-entropy loss, as in the paper.
- `_confidence`: the highest softmax probability c, compared as the loss −log c.
- `_entropy`: the entropy of the prediction, compared like a loss.

The last two need no label. The attacks are {py:class}`~src.mia.attacks.black_box.attack_r.AttackRMIA` and
{py:class}`~src.mia.attacks.black_box.attack_p.AttackPMIA`.

## Parameters

`--attacker_rmia_alpha` and `--attacker_pmia_alpha` are the α of Attack-R and Attack-P. They only set the attacks' own
decisions, so they change neither the scores nor the AUC and the TPR at fixed false-positive rates.

```{include} ../_generated/options_attack_r_p.md
```
