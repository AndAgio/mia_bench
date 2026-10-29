# SELENA

SELENA ([Tang et al., USENIX Security 2022](https://arxiv.org/abs/2110.08324)) releases a model trained only on
outputs that do not depend on whether a sample was trained on. It first trains K split models, each on part of the
training data: every training sample is left out of L of them, chosen at random, and trains the other K − L. Each
training sample then gets as soft label the average softmax output of the L models that never saw it. A new model
trains on these soft labels, and it is the one that answers queries. Inference is unchanged. The authors' code is
[here](https://github.com/inspire-group/MIAdefenseSELENA).

Training costs K + 1 trainings: each split model trains on a share (K − L) / K of the data, and the final model on
all of it, all with the same training options. The defender's `split_data_manager` tells which split models saw a
training sample: `get_teachers(i)` returns the L that never saw the sample with original index i, and
`get_trained_models(i)` the others.

## Usage

```bash
python run.py --defender_mode selena --dataset cifar10 --attacker_mode lira
```

The defense is {py:class}`~src.mia.defenses.training.selena.SelenaDefender`.

## Parameters

`--defender_selena_K` is the number of split models K, and `--defender_selena_L` the number L of them that never see
each sample, with 1 ≤ L < K. The defaults are the values that the paper uses on all its datasets.

```{include} ../_generated/options_selena.md
```

## Differences from the paper

- The authors' code keeps the final model with the best test accuracy. As for every defense here, we keep the one
  with the best validation performance, since the test split holds the non-members that the attacks audit.
