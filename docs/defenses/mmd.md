# MMD

MMD ([Li, Li and Ribeiro, CODASPY 2021](https://arxiv.org/abs/2002.12062)) trains the model so that its outputs on
members look like its outputs on non-members. After each training epoch, if the model's accuracy on the training data
exceeds its accuracy on a set of reference non-members by at least {{mmd_min_accuracy_gap}}, a second pass over the
training data minimizes the maximum mean discrepancy (MMD) between the softmax outputs of each training batch and
those of reference samples of the same classes. Only the training outputs get gradients. Inference is unchanged. The
authors' code is [here](https://github.com/colaalex111/MMD-mixup-Defense).

The reference non-members are {{mmd_reference_share}} of the validation split, stratified by class; the rest selects
the best epoch, so that the samples MMD trains against never decide which model is kept. As in the authors' code, the
MMD pass runs only in epochs where the model overfits, which is rarely the case in the first ones, and never in the
last epoch. Each epoch also measures the training and reference accuracies, and an epoch with the MMD pass takes one
more pass over the training data.

## Usage

```bash
python run.py --defender_mode mmd --dataset cifar10 --attacker_mode lira
```

`mmd_mixup` also trains the regular epochs with mixup, the paper's choice on CIFAR and Texas.

The defense is {py:class}`~src.mia.defenses.training.mmd.MmdDefender`.

## Parameters

`--defender_mmd_lambda` is λ, the weight of the MMD loss; 0 turns MMD off. When left unset, it takes the paper's value
for the dataset (Table 5):

```{include} ../_generated/mmd_parameters.md
```

The paper picks, for each dataset and model, the largest λ whose test accuracy stays within 1% of the undefended
model's, with mixup on CIFAR and Texas and without it on Purchase. On CIFAR, the table above takes the values of
ResNet-20, the paper's model closest to the default ResNet-18. λ varies a lot with the model in the paper, so for
another model, or to trade accuracy and privacy differently, tune it: train with several values of
`--defender_mmd_lambda` and keep the largest whose validation accuracy, printed in the training log, stays close
enough to that of `--defender_mode none`. Tuning on the validation accuracy rather than the test accuracy keeps the
audited non-members, drawn from the test split, out of the choice.

With `mmd_mixup`, `--defender_mixup_alpha` is the α of mixup.

```{include} ../_generated/options_mmd.md
```
