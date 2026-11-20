# Differential privacy

DP-SGD ([Abadi et al., CCS 2016](https://arxiv.org/abs/1607.00133)) trains with differential privacy. Each step clips
the gradient of every sample to a norm of at most C and adds Gaussian noise of standard deviation σ·C to their sum,
so the trained model is (ε, δ)-differentially private: changing one training sample changes the probability of any
outcome of the training by a factor of at most e^ε, plus δ. [Opacus](https://opacus.ai) does the sampling, the
clipping, the noise and the accounting. Inference is unchanged.

After training, the log gives the ε that the whole run spent at δ, and so do `privacy_epsilon` and `privacy_delta` in
`results.json` and `summary.csv`. ε covers every epoch, since the best model is picked among all of them, and a
resumed run keeps counting from where it stopped.

Opacus cannot compute per-sample gradients through BatchNorm, so it replaces each BatchNorm layer with GroupNorm: the
defended model's architecture differs from the undefended one's. Opacus' secure random generator is off, which is
fine to measure attacks but not for a deployment.

With `--defender_optimizer sam`, the training is DP-SAT ([Park et al., ICML 2023](https://arxiv.org/abs/2306.05651),
[code](https://github.com/JinseongP/DPSAT)): SAM's perturbation follows the previous step's noisy gradient, so it
spends the same ε as DP-SGD. The other SAM-like optimizers raise an error with this defense.

## Usage

```bash
python run.py --defender_mode dp --dataset cifar10 --attacker_mode lira
python run.py --defender_mode dp --defender_dp_epsilon 4 --dataset cifar10 --attacker_mode lira
python run.py --defender_mode dp --defender_optimizer sam --dataset cifar10 --attacker_mode lira
```

The defense is {py:class}`~src.mia.defenses.training.differential_privacy.DifferentialPrivacyDefender`.

## Parameters

`--defender_dp_noise_multiplier` is σ and `--defender_dp_max_grad_norm` the clipping bound C, both 1 by default as a
generic starting point for models trained from scratch. With `--defender_dp_epsilon`, Opacus instead picks the σ that
reaches this ε at δ after all the epochs. `--defender_dp_delta` is δ, and with `--defender_optimizer sam`,
`--defender_sam_rho` is DP-SAT's ρ:

```{include} ../_generated/dp_parameters.md
```

With `--defender_dp_clip_per_layer`, the gradient of each of the L layers is clipped to C/√L instead.
`--defender_dp_grad_sample_mode` sets how Opacus computes the per-sample gradients, which decides the memory a step
takes. Memory of one step with CIFAR-10 images, in MB per sample, measured on MPS:

| `--defender_model` | ghost | `hooks` | `functorch` |
| --- | --- | --- | --- |
| `resnet18` | 42 | 75 | 84 |
| `mobile_small` | 35 | 68 | 37 |
| `mobile_large` | 125 | 127 | 72 |

The default batch of 256 thus needs about 11 GB on ResNet-18. If it does not fit, lower `--defender_batch_size`,
which is the expected size of the Poisson batches.

As an example of the budget, with 18,000 training samples, batches of 256 and 100 epochs, σ = 1 spends ε ≈ 7.7 at
δ = 10⁻⁵, σ ≈ 1.5 reaches ε = 4, and σ ≈ 4.6 reaches ε = 1.

```{include} ../_generated/options_dp.md
```
