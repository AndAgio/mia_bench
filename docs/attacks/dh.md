# DHAttack

DHAttack ([Li et al., USENIX Security 2025](https://www.usenix.org/system/files/usenixsecurity25-li-hao.pdf)) is
label-only: it reads only the labels the target predicts, and needs few queries per sample. The authors' code is
[here](https://github.com/AIPAG/DHAttack).

The attack moves each sample toward a fixed input in K equal steps, and counts the steps before the model's label
stops being the sample's: its fixed boundary distance, or fixedBD (Alg. 2). A model tends to keep the label of a sample
it trained on for longer, but samples differ widely in how far they are from the boundary. So the attack compares each
sample with itself as a non-member. It has the target label its shadow dataset, trains `--n_shadows` shadow models on
those labels, and measures the sample's fixedBD on each of them. None of them trained on the sample, so a Gaussian
fitted to their distances describes the sample as a non-member. The relScore is the CDF of the target's fixedBD under
this Gaussian (Alg. 1): it is close to 1 when the target keeps the label much longer than models that never saw the
sample.

The score is −log(1 − relScore), which orders the samples as the relScore does. The relScore itself rounds to
exactly 1.0 for every sample more than about 8.3 standard deviations above the shadow models' mean, and for every
sample above it when all the shadow models give the same distance. No threshold separates those samples. In a test on
CIFAR-10, with a small CNN trained on 5,000 images (100% training and 66% test accuracy), 32 shadow models trained on
10,000 other images, and 2,000 members and 2,000 non-members audited, 10.4% of members and 1.3% of non-members scored
exactly 1.0. TPR at 0.1% and 1% FPR was then 0, against 0.6% and 8.9% with −log(1 − relScore), for about the same AUC
(0.763 and 0.764). When all the shadow models give the same distance, 1e-30 is added to their standard deviation, as
in the authors' code, so these samples score above all others, in the order of how far the target's distance exceeds
theirs.

The mode sets the fixed input:

| Mode | Images | Tabular data |
|---|---|---|
| `dh` or `dh_white` | The all-white image, every pixel at 255: the paper's default | Every feature at the sample's largest value, as in the paper (Sec. 6.1): all ones on Purchase and Texas |
| `dh_black` | The all-black image, which the paper also tests (Table 7) | Every feature at the sample's smallest value |
| `dh_random` | One image of uniform random pixels, the same for every sample | One random record, the same for every sample: random bits on Purchase and Texas, uniform values in [0, 1] on News |

`dh_random` is not in the paper. The paper also tries real images as fixed inputs, and finds that images from outside
the target's data work better than images from it, and the all-white image best (Sec. 5.2, Fig. 8).

On images, the attack's own threshold comes from random images (Sec. 5.3). The attack draws {{dh_synthetic_samples}}
images of uniform random pixels from `--seed`, and sets the threshold at the {{dh_synthetic_percentile}}th percentile
of their scores. The paper finds the best threshold between the 95th and the
100th percentile (Fig. 13). A random image belongs to no class, so its fixedBD starts from the label the target
predicts for it, as in the [unsupervised boundary attack](uba.md); the paper does not say which label it uses. In the
CIFAR-10 test, this threshold gave an FPR of 0.5% and a TPR of 6.4%.

On tabular data, the attack has no threshold of its own, so `--decision_rule attack` uses the calibrated rule. The
paper's main threshold (Sec. 3.2) is the one that reaches a set FPR on the non-members of a shadow model, attacked as
the target is, and the calibrated rule lands in the same place. In a test on CIFAR-10, with 5,000 members and 10,000
non-members audited, at target FPRs of 10%, 5% and 1% the paper's threshold gave 11.1%, 5.1% and 0.95%, and the
calibrated rule 8.7%, 5.3% and 1.15%. The threshold only sets the attack's own decisions: the scores, and the
calibrated and oracle rules, do not depend on it.

The shadow dataset takes {{shadow_dataset_share}} of the attacker's data, as for the attacks with one shadow model,
and the target labels it once. In the paper, a reference dataset twice as large as the target's training set works
best (Sec. 5.3, Table 4). Every shadow model trains on the same shadow dataset, with the attacker's training options
(see [Training defaults](../reference/training.md)) and with `--seed` plus its index. The seeds make their batches and
augmentation differ, not only their initial weights: in the CIFAR-10 test, 10 shadow models trained with the same seed
gave an AUC of 0.742, against 0.756 with a seed each. The paper trains 256 shadow models, and finds that the gains
level off between 128 and 256 (Sec. 4.3 and 5.3, Fig. 12). In the CIFAR-10 test, 32 gave an AUC of 0.764 against
0.756 with 10, and fewer samples on which all the shadow models agree (11% against 17%).

The authors' code differs from the paper in three ways. It moves each image toward a constant image at the image's
own largest value. It fits the Gaussian to a transform of the distance. And it gives a sample whose label is wrong from
the first step a small distance, which shrinks the longer the label stays wrong, so it keeps querying such samples. In
the CIFAR-10 test, the first two changed the results by no more than the noise. The third, measured on 5,000 members
and 10,000 non-members, raised TPR at 0.1% FPR from 1.4% to 3.2%, lowered it at 1% FPR from 11.6% to 9.2%, and took 42%
more queries. This library follows the paper.

The paper finds the attack weaker on non-image data than on images (Sec. 6.1). In a test on Purchase, with a tabular
MLP trained for 300 epochs on 5,876 records (99.6% training and 68.6% test accuracy, close to the paper's 100% and
71.6%), 32 shadow models trained on 11,215 other records, and 2,000 members and 2,000 non-members audited, the AUC was
0.661 and TPR at 1% FPR 2.5%. The paper reports an AUC of 0.727 (Table 8). On the same target, calling every correctly
classified record a member gives an AUC of 0.658, and on a target trained the same way the
[noise robustness attack](noise_robustness.md) reached 0.772. Two things hold DHAttack back here. The steps toward the
all-ones record barely tell correctly classified members from non-members: on those records, the target's fixedBD
alone gives an AUC of 0.55, and the relScore 0.60. And the shadow models learn the target's labels, wrong on 31% of
the shadow dataset, so they are more accurate on the target's members (79.5%) than on its non-members (63%), though
they trained on neither. They then give the members longer distances too, which cancels part of the members' lead. With
16 shadow models trained on the true labels instead, the paper's "−relabel" (Fig. 16), the AUC was 0.692 and TPR at
1% FPR 2.7%, against 0.658 and 1.7% with 16 trained on the target's labels.

Scoring a sample sends the target one query per step until its label changes, at most `--attacker_dh_n_queries`
({{dh_n_queries}}, the paper's K). The paper finds K from 10 to 50 enough (Sec. 3.2). The paper's last step, the fixed
input itself, is not sent: the fixedBD is K whether or not the label changes there. A sample took 12 queries on average
in the CIFAR-10 test, and 5 on Purchase. Labelling the shadow dataset costs one query per sample, and the random images
one each; these setup queries are not in the query counts. Most of the time goes to training the shadow models.

## Usage

```bash
python run.py --attacker_mode dh --dataset cifar10 --defender_mode none
```

By default, the decisions come from the calibrated rule, as for every attack. For the paper's own decisions on images,
described above, add `--decision_rule attack` (see [Decision rules](decision_rules.md)).

On tabular data, the target and the shadow models are MLPs:

```bash
python run.py --attacker_mode dh --dataset purchase --defender_model tabular_mlp --attacker_model tabular_mlp \
    --defender_mode none
```

The attack is {py:class}`~src.mia.attacks.label_only.dh_attack.DHAttack`.

## Parameters

`--n_shadows` is the number of shadow models, shared with the other attacks that train several. The shadow models'
architecture and training come from the attacker's options, and `--attacker_data_split_perc` sets how much data the
attacker has.

```{include} ../_generated/options_dh.md
```
