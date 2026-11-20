# Mixup

Mixup ([Zhang, Cisse, Dauphin and Lopez-Paz, ICLR 2018](https://arxiv.org/abs/1710.09412)) trains on convex
combinations of pairs of samples and of their labels: each step mixes the batch with a shuffled copy of itself, with a
weight λ ~ Beta(α, α). The model never trains on the samples themselves, so it memorizes them less.
[Li, Li and Ribeiro (CODASPY 2021)](https://arxiv.org/abs/2002.12062) evaluate mixup as a membership inference
defense, alone and with an MMD regularizer (`mmd_mixup`). Inference is unchanged. The authors' code is
[here](https://github.com/facebookresearch/mixup-cifar10).

Mixup trains without the random crops and flips of the authors' code, which are the separate `data_augmentation`
defense here.

## Usage

```bash
python run.py --defender_mode mixup --dataset cifar10 --attacker_mode lira
```

The defense is {py:class}`~src.mia.defenses.training.mixup.MixupDefender`.

## Parameters

`--defender_mixup_alpha` is α. The paper uses α = 1 on CIFAR, which spreads λ uniformly over [0, 1], and mixup
becomes normal training as α goes to 0. The same option sets the mixup of `mist_mixup` and `mmd_mixup`.

```{include} ../_generated/options_mixup.md
```
