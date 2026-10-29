# MIST

MIST ([Li, Li and Ribeiro, USENIX Security 2024](https://arxiv.org/abs/2311.00919)) trains the model so that its
confidence on each training sample is about the same whether or not that sample was used to train it. Every epoch
splits the training data into C subsets and trains a copy of the model on each one (Phase 1). Each copy is then moved
towards the others on its own samples (Phase 2), by minimizing λ times the cross-difference loss of Eq. 3: the L1
distance between its probability of the true class and the average of the other copies'. The epoch ends by averaging
the copies into the model. Inference is unchanged. The paper does not link to code.

Every step of an epoch, in both phases, uses the learning rate of that epoch, since Table 7 gives the schedules in
epochs of the whole procedure.

## Usage

```bash
python run.py --defender_mode mist --dataset cifar10 --attacker_mode lira
```

`mist_mixup` also trains with mixup in Phase 1.

The defense is {py:class}`~src.mia.defenses.training.mist.MistDefender`.

## Parameters

`--defender_mist_num_submodels` is the number of copies C, at least 2, and `--defender_mist_lambda` the weight λ of
the cross-difference loss. When left unset, both take the values of Table 7 in the paper:

```{include} ../_generated/mist_parameters.md
```

`--defender_mist_submodel_epochs` is the number of passes over its subset that each copy makes in each phase, which
the paper sets to one. `--defender_mist_split_method` splits the data at random, as the paper does, or with the same
share of every class in each subset. With `mist_mixup`, `--defender_mixup_alpha` sets the α of mixup. Table 7 also
gives the learning rate, epochs and schedule of each model, which the training options set (see
[Command-line options](../reference/cli.md)).

```{include} ../_generated/options_mist.md
```

## Differences from the paper

- For CIFAR-10 and CIFAR-100, Table 7 gives C and λ for each model. We use the ResNet-18 values for every model,
  since the benchmark has no AlexNet or DenseNet.
- The paper does not say where mixup goes in Algorithm 1, nor which α it uses. `mist_mixup` applies mixup in Phase 1
  only, and Phase 2 aligns the copies on clean samples.
- The paper does not say what happens to the optimizer's state. Phase 2 continues with each copy's Phase 1 optimizer,
  so the two losses share its momentum, and at the end of each epoch the copies' optimizer states are averaged into
  the model's, like their parameters.
- The paper returns the model of the last epoch. As for every defense here, we keep the model with the best
  validation performance.
