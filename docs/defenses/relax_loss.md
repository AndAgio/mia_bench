# RelaxLoss

RelaxLoss ([Chen, Yu and Fritz, ICLR 2022](https://arxiv.org/abs/2207.05801)) keeps the training loss near a target
value α instead of driving it to zero, so that the losses of members stay close to those of non-members. Every step
computes the cross-entropy of the batch. Above α, it takes a gradient descent step. Below α, it takes a gradient
ascent step on even epochs, and on odd epochs it flattens the outputs: it trains on soft labels that keep the
true-class score and spread the rest evenly over the other classes. Inference is unchanged. The authors' code is
[here](https://github.com/DingfanChen/RelaxLoss).

As in App. B.3, image datasets flatten only the incorrect predictions, and non-image datasets flatten every sample
with a bound on the true-class score. Each step picks its branch once, on the whole batch at the current weights, so
optimizers like SAM, whose later passes use perturbed weights or (ESAM) part of the batch, keep the branch of the
batch.

## Usage

```bash
python run.py --defender_mode relax_loss --dataset cifar10 --attacker_mode lira
```

The defense is {py:class}`~src.mia.defenses.training.relax_loss.RelaxLossDefender`.

## Parameters

`--defender_relax_loss_alpha` is the target loss α, above 0, and `--defender_relax_loss_upper` the bound on the
true-class score of the soft labels, in (0, 1]. When left unset, both take the values of the official configs:

```{include} ../_generated/relax_loss_parameters.md
```

`--defender_relax_loss_flatten` flattens only the `incorrect` predictions or `all` samples. When left unset, it
follows App. B.3: `incorrect` on image datasets, `all` on Purchase, Texas and News.

```{include} ../_generated/options_relax_loss.md
```

## Differences from the paper

- The paper gives no value for α. We take those of the official configs, with the ResNet-20 values for every model
  on CIFAR; the configs give different ones for VGG11-bn.
- App. B.3 flattens only the incorrect predictions on image data, but does not say what the correct ones get. As in
  the official code, they get gradient ascent.
- The official code gives the flattened predictions gradient ascent as well. We only flatten them, as the paper does.
- Epochs count from 1, as in Algorithm 1, so the first epoch below α flattens. The official code counts from 0 and
  starts with gradient ascent.
- The soft labels are constants, as in the paper. The official code lets the gradient flow through them.
- The official code keeps the model of the last epoch. As for every defense here, we keep the one with the best
  validation performance.
