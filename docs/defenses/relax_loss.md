# RelaxLoss

RelaxLoss ([Chen, Yu and Fritz, ICLR 2022](https://arxiv.org/abs/2207.05801)) keeps the training loss near a target
value α instead of driving it to zero, so that the losses of members stay close to those of non-members. Above α it
takes a normal gradient step; below α it alternates gradient ascent with flattening the outputs, training on soft
labels that keep the true-class score and spread the rest over the other classes. Inference is unchanged. The
authors' code is [here](https://github.com/DingfanChen/RelaxLoss).

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
follows the paper: `incorrect` on image datasets, `all` on Purchase, Texas and News.

```{include} ../_generated/options_relax_loss.md
```
