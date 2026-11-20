# MIST

MIST ([Li, Li and Ribeiro, USENIX Security 2024](https://arxiv.org/abs/2311.00919)) trains the model so that its
confidence on each training sample is about the same whether or not that sample was used to train it. Every epoch
splits the training data into C subsets and trains a copy of the model on each one. Each copy is then moved towards
the others on its own samples, weighted by λ, and the epoch ends by averaging the copies into the model. Inference is
unchanged. The paper does not link to code.

An epoch trains C copies, so training costs more than normal training.

## Usage

```bash
python run.py --defender_mode mist --dataset cifar10 --attacker_mode lira
```

`mist_mixup` also trains the copies with mixup.

The defense is {py:class}`~src.mia.defenses.training.mist.MistDefender`.

## Parameters

`--defender_mist_num_submodels` is the number of copies C, at least 2, and `--defender_mist_lambda` the weight λ. When
left unset, both take the values of Table 7 in the paper, those of ResNet-18 on CIFAR:

```{include} ../_generated/mist_parameters.md
```

`--defender_mist_submodel_epochs` is the number of passes each copy makes over its subset in each phase, one in the
paper. `--defender_mist_split_method` splits the data at random, as the paper does, or with the same share of every
class in each subset. With `mist_mixup`, `--defender_mixup_alpha` sets the α of mixup, which applies while the copies
train on their subsets. Table 7 also gives the learning rate, epochs and schedule of each model, which the training
options set (see [Command-line options](../reference/cli.md)).

```{include} ../_generated/options_mist.md
```
