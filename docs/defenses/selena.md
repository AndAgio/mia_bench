# SELENA

SELENA ([Tang et al., USENIX Security 2022](https://arxiv.org/abs/2110.08324)) releases a model trained only on
outputs that do not depend on whether a sample was trained on. It first trains K split models, each on part of the
training data, with every training sample left out of L of them. Each training sample then gets as soft label the
average output of the L models that never saw it, and a new model trains on these soft labels. That model answers the
queries. Inference is unchanged. The authors' code is [here](https://github.com/inspire-group/MIAdefenseSELENA).

Training costs K + 1 trainings: each split model trains on a share (K − L) / K of the data, and the final model on
all of it, all with the same training options.

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
