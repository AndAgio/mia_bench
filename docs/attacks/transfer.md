# Transfer

The transfer attack ([Li and Zhang, CCS 2021](https://arxiv.org/abs/2007.15528), Sec. 3) is label-only: it reads only
the labels the target predicts. The attacker sends its shadow dataset to the target, trains a shadow model on the
labels it gets back, and scores each sample on the shadow model alone, so scoring sends the target no query. Samples
the target was trained on tend to score as members on the shadow model too: the target's membership transfers to it.
The authors' code is [here](https://github.com/zhenglisec/Decision-based-MIA).

The shadow dataset takes {{shadow_dataset_share}} of the attacker's data, as for the other attacks with one shadow
model. The shadow model is `--attacker_model`, trained with the attacker's training options (see
[Training defaults](../reference/training.md)). The paper uses the target's architecture, and finds that others work
about as well (Sec. 3.4). It also finds that a larger shadow dataset gives a stronger attack: its CIFAR-10 shadow
dataset holds 46,000 images, against 100 to 3,000 for the targets (App. Table 7).

The mode sets the score, computed on the shadow model:

- `transfer_loss`: the cross-entropy at the sample's true label, as in the paper. A sample is a member when its loss is
  at most the threshold (Alg. 1).
- `transfer_confidence`: the largest probability.
- `transfer_entropy`: the entropy of the probabilities. The paper divides it by log K, which ranks the samples the same.

The paper finds the loss the strongest of the three (Fig. 5).

The threshold is chosen on the shadow model: the score that best separates its members from as many samples outside
its shadow dataset, by accuracy. It uses all of them, since each costs one pass through the shadow model. The paper
suggests a threshold taken from shadow samples held out as non-members instead (Sec. 3.4), which is what the
calibrated rule does. In two tests on CIFAR-10, the two gave similar accuracy. The threshold only sets the
attack's own decisions (`--decision_rule attack`): the scores, and the calibrated and oracle rules, do not depend on it.

The authors' code sends each sample to the target first and gives the largest loss to the ones the target
misclassifies, which adds the baseline attack (a sample is a member when the target classifies it correctly) to the
transfer attack. The paper describes the attack without these queries (Sec. 3.2, Alg. 1, Table 4), as this library
does.

In a test with the authors' CNN on CIFAR-10, the attack beat the baseline only against the more overfit target. Its
shadow model trained for 30 epochs, with crops and flips, where theirs trains for 200 with stronger augmentation.

| Target trained on | Transfer (paper) | Authors' code | Member when classified correctly |
|---|---|---|---|
| 100 images | 0.916 | 0.938 | 0.860 |
| 3,000 images | 0.568 | 0.621 | 0.670 |

The values are AUCs. For the target trained on 100 images, the paper reports 0.94 for the transfer attack and 0.815
for the baseline (Fig. 2).

The cost is one query per shadow sample, to label it, and training the shadow model.

## Usage

```bash
python run.py --attacker_mode transfer_loss --dataset cifar10 --defender_mode none
```

By default, the decisions come from the calibrated rule, as for every attack. For the paper's own decisions,
described above, add `--decision_rule attack` (see [Decision rules](decision_rules.md)).

The attack is {py:class}`~src.mia.attacks.label_only.transfer.TransferMIA`.

## Parameters

The transfer attack has no options of its own: the mode sets the score, the shadow model's architecture and training
come from the attacker's options (`--attacker_model`, `--attacker_epochs` and the others in
[Training defaults](../reference/training.md)), and `--attacker_data_split_perc` sets how much data the attacker has.
