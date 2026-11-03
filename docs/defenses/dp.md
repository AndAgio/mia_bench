# Differential privacy

DP-SGD ([Abadi et al., CCS 2016](https://arxiv.org/abs/1607.00133)) trains with differential privacy. Each step
draws a batch by Poisson sampling, clips the gradient of each sample to a norm of at most C, and adds Gaussian noise
of standard deviation σ·C to their sum. An accountant tracks the privacy that the steps spend, so the trained model
is (ε, δ)-differentially private: changing one training sample changes the probability of any outcome of the
training by a factor of at most e^ε, plus δ. [Opacus](https://opacus.ai) does the sampling, the clipping, the noise
and the accounting, with its PRV accountant. Inference is unchanged.

After training, the log gives the ε that the whole run spent at δ, and so do `privacy_epsilon` and `privacy_delta` in
`results.json` and `summary.csv`. The best model is picked among the models of all the epochs, so ε covers every
epoch, not only those up to the best one. The validation samples that pick it are never trained on. The accounting
is saved in each checkpoint, so that a resumed run counts the epochs before the resume.

Opacus cannot compute per-sample gradients through BatchNorm, which mixes the samples of a batch, so it replaces each
BatchNorm layer with GroupNorm before training. A Poisson batch can be empty, and still counts as a noisy step.
Opacus' secure mode is off: the noise comes from PyTorch's default random generator, which is enough to measure
attacks but not for a deployment.

With `--defender_optimizer sam`, the training is DP-SAT ([Park et al., ICML 2023](https://arxiv.org/abs/2306.05651),
[code](https://github.com/JinseongP/DPSAT)). SAM computes its gradient at weights moved by ρ along the batch's own
gradient, which DP-SGD would also have to release with noise, so each step would spend privacy twice. DP-SAT moves
the weights along the previous step's noisy gradient instead, which is already released, so it spends the ε of DP-SGD
at about the same cost. The first step is not perturbed, and the training loss in the log is measured at the moved
weights. The adaptive SAM and the other SAM-like optimizers raise an error with this defense.

## Usage

```bash
python run.py --defender_mode dp --dataset cifar10 --attacker_mode lira
python run.py --defender_mode dp --defender_dp_epsilon 4 --dataset cifar10 --attacker_mode lira
python run.py --defender_mode dp --defender_optimizer sam --dataset cifar10 --attacker_mode lira
```

The defense is {py:class}`~src.mia.defenses.training.differential_privacy.DifferentialPrivacyDefender`.

## Parameters

`--defender_dp_noise_multiplier` is σ and `--defender_dp_max_grad_norm` the clipping bound C. With
`--defender_dp_epsilon`, Opacus instead picks the σ that reaches this ε at δ after all the epochs, with the training's
batch size as the expected size of the Poisson batches. `--defender_dp_delta` is δ, and the paper uses 10⁻⁵ in every
experiment. With `--defender_optimizer sam`, `--defender_sam_rho` is DP-SAT's ρ:

```{include} ../_generated/dp_parameters.md
```

With `--defender_dp_clip_per_layer`, the gradient of each of the L layers is clipped to C/√L instead.
`--defender_dp_grad_sample_mode` sets how Opacus computes the per-sample gradients. `hooks` stores them for every
layer, and `functorch` computes them with `torch.func` instead. Ghost clipping only needs their norms, which it gets
without the gradients themselves, but only for Linear and Embedding layers: Opacus computes the other layers as
`hooks` does. Which mode takes the least memory depends on the model. Memory of one step with CIFAR-10 images, in MB
per sample, measured on MPS:

| `--defender_model` | ghost | `hooks` | `functorch` |
| --- | --- | --- | --- |
| `resnet18` | 42 | 75 | 84 |
| `mobile_small` | 35 | 68 | 37 |
| `mobile_large` | 125 | 127 | 72 |

The default batch of 256 thus needs about 11 GB on ResNet-18. Lower `--defender_batch_size` if it does not fit: with
Poisson sampling, it is the expected batch size. On MPS, the batch size changes at every step and MPS keeps memory
cached for each size, so the defense empties the MPS cache after each step. Without that, the cache grew from 2.7 to
14.6 GB in 40 steps of ResNet-18. CUDA and CPU skip this.

As an example of the budget, with 18,000 training samples, batches of 256 and 100 epochs, σ = 1 spends ε ≈ 7.7 at
δ = 10⁻⁵, σ ≈ 1.5 reaches ε = 4, and σ ≈ 4.6 reaches ε = 1.

```{include} ../_generated/options_dp.md
```

## Differences from the paper

- σ and C default to 1, a generic starting point, and any model of the benchmark trains from scratch. On CIFAR-10,
  the paper trains only the fully connected layers of a small network whose convolutions are pre-trained on
  CIFAR-100, with lots of 2,000 or 4,000 samples, σ = 6, C = 3 and data augmentation (Sec. 5.3).
- The accounting uses Opacus' PRV accountant, which gives a tighter ε for the same training than the paper's
  moments accountant.
- BatchNorm becomes GroupNorm, so the defended model's architecture differs from the undefended one, and from the
  shadow models that some attacks train.
- We keep the model with the best validation performance, as for every defense here, and report the ε of the whole
  run.
- DP-SAT's ρ is 0.03 for every dataset and model, the value its paper uses without tuning for all its models but one
  (Appendix G.2). The paper tunes ρ per dataset, model and ε, between 0.01 and 0.1 (Table 5), with batches of 2,048,
  C = 0.1 and a learning rate of 2.
