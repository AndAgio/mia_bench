# Weighted smoothing

Weighted smoothing ([Tan, Sun, Xie and Wang, ACSAC 2023](https://doi.org/10.1145/3627106.3627189)) adds Gaussian noise
to the model's predictions during training, more to the samples most exposed to membership inference. After a few
epochs of normal training, each epoch scores every training sample with Mentr
([Song and Mittal, USENIX Security 2021](https://arxiv.org/abs/2003.10595)) and weights its noise by how exposed the
sample is within its class. Inference is unchanged. The code the paper links, github.com/BennyTMT/weighted-smoothing,
no longer exists.

As with the other defenses, training keeps the epoch with the best validation score, but only among the epochs after
the warmup, which trains normally. The modified loss defends on its own, even without noise: runs with
`--defender_weighted_smoothing_sigma_noise 0` separate the effect of the loss from that of the noise.

## Usage

```bash
python run.py --defender_mode weighted_smoothing --dataset cifar10 --attacker_mode lira
```

The defense is {py:class}`~src.mia.defenses.training.weighted_smoothing.WeightedSmoothingDefender`.

## Parameters

`--defender_weighted_smoothing_sigma_noise` is σ, the standard deviation of the noise before the weights scale it.
The paper gives no single value: it varies σ to trace the trade-off between accuracy and privacy.

`--defender_weighted_smoothing_warmup_epochs` is the number of epochs of normal training before the smoothing starts,
and must be below `--defender_epochs`. When left unset, it takes the values of Sec. 5.4:

```{include} ../_generated/weighted_smoothing_parameters.md
```

```{include} ../_generated/options_weighted_smoothing.md
```
