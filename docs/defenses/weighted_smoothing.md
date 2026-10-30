# Weighted smoothing

Weighted smoothing ([Tan, Sun, Xie and Wang, ACSAC 2023](https://doi.org/10.1145/3627106.3627189)) adds Gaussian noise
to the model's predictions during training, more to the samples most exposed to membership inference. After a few
epochs of normal training, each epoch starts by scoring every training sample on the current model with Mentr
([Song and Mittal, USENIX Security 2021](https://arxiv.org/abs/2003.10595)), which is low for confident predictions.
The scores are standardized within each class into weights w = 1 − (Mentr − μ_c)/σ_c, and each training step adds
noise w·N(0, σ²I) to the softmax output of every sample. Inference is unchanged. The paper links its code at
github.com/BennyTMT/weighted-smoothing, which no longer exists, and no archive holds a copy.

The noise is symmetric, so only the size |w| of the weight matters. It is 1 at the class mean, larger for the more
confident samples, and zero one standard deviation above the mean. Further up it grows again, as in Algorithm 1:
samples more than two standard deviations above the mean, the least exposed ones, get more noise than an average
sample.

Mentr is computed in eval mode, with the population standard deviation, and a class with a single sample gets weight 1.
Each step draws its noise once, so optimizers like SAM, whose later passes use perturbed weights or (ESAM) part of the
batch, reuse the noise of the batch. As in Algorithm 1, training returns the model of the last epoch, where the other
defenses here keep the one with the best validation performance.

## Usage

```bash
python run.py --defender_mode weighted_smoothing --dataset cifar10 --attacker_mode lira
```

The defense is {py:class}`~src.mia.defenses.training.weighted_smoothing.WeightedSmoothingDefender`.

## Parameters

`--defender_weighted_smoothing_sigma_noise` is σ, the standard deviation of the noise before the weight scales it. The
paper gives no single value: it varies σ to trace the trade-off between accuracy and privacy (Fig. 6 and 11c), and
picks it per experiment to reach a target attack success rate (Table 3).

`--defender_weighted_smoothing_warmup_epochs` is the number of epochs of normal training before the smoothing starts,
and must be below `--defender_epochs`. When left unset, it takes the values of Sec. 5.4:

```{include} ../_generated/weighted_smoothing_parameters.md
```

```{include} ../_generated/options_weighted_smoothing.md
```

## Differences from the paper

- σ defaults to 0.1. The paper gives no default value.
- The paper does not say which loss takes the noisy predictions. We pass them to the cross-entropy as scores, so the
  softmax is applied twice. This is what inserting the noise before the usual loss gives, and the paper's ablation
  (Sec. 6.2) moves the noise between layers under the same training. The alternative, −log of the noisy true-class
  score clamped above 0, stops the model from learning: with ResNet-18 trained with Adam for 100 epochs on 5,000
  CIFAR-10 samples, σ = 0.1 pushes 22% of these scores to 0 or below and ends at 16.5% test accuracy.
- The loss alone also defends, while the paper's curves reach the undefended leakage at the smallest noise (Fig. 6).
  In the same setup, with the learning rate decaying by 0.97 per epoch, σ = 0 keeps the test accuracy of normal
  training (63.6% against 63.5%) but fits 96.9% of the training set instead of all of it, and lowers a Mentr threshold
  attack from 81.1% to 76.5%. A sample the model gets confidently wrong pulls on the weights about 100 times less than
  with the plain cross-entropy. Runs with σ = 0 separate the effect of the loss from that of the noise.
