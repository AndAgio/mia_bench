# Mixup

Mixup ([Zhang, Cisse, Dauphin and Lopez-Paz, ICLR 2018](https://arxiv.org/abs/1710.09412)) trains on convex
combinations of pairs of samples and of their labels. Each training step mixes the batch with a shuffled copy of
itself, with one weight λ ~ Beta(α, α) for the whole batch: the inputs become λ·x_i + (1 − λ)·x_j, and the loss
λ·l(f(x), y_i) + (1 − λ)·l(f(x), y_j), which for the cross-entropy is the loss against the mixed label. The model
never trains on the samples themselves, so it memorizes them less.
[Li, Li and Ribeiro (CODASPY 2021)](https://arxiv.org/abs/2002.12062) evaluate mixup as a membership inference
defense, alone and with an MMD regularizer (`mmd_mixup`). Inference is unchanged. The authors' code is
[here](https://github.com/facebookresearch/mixup-cifar10).

The training accuracy in the logs is measured on the clean batch, in a second forward pass that leaves the BatchNorm
running statistics unchanged.

## Usage

```bash
python run.py --defender_mode mixup --dataset cifar10 --attacker_mode lira
```

The defense is {py:class}`~src.mia.defenses.training.mixup.MixupDefender`.

## Parameters

`--defender_mixup_alpha` is α. The paper uses α = 1 on CIFAR (Sec. 3.2), which spreads λ uniformly over [0, 1], and
mixup becomes normal training as α goes to 0. The same option sets the mixup of `mist_mixup` and `mmd_mixup`.

```{include} ../_generated/options_mixup.md
```

## Differences from the paper

- The authors' code also augments the training images with random crops and flips by default. Here augmentation is
  its own defense, `data_augmentation`, and mixup trains without it.
- We keep the model with the best validation performance, as for every defense here. The authors' code keeps the
  epoch with the best test accuracy.
