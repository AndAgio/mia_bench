# Purifier

Purifier ([Yang et al., AAAI 2023](https://arxiv.org/abs/2212.00612)) makes the model's outputs on members look like
its outputs on non-members. A confidence reformer, an autoencoder trained on the model's outputs on non-members,
rewrites each output with noise in its latent space. A label swapper then gives a share of the members their second
most likely label, so that members are classified correctly no more often than non-members. Training is unchanged.
The authors' repository ([github.com/wljLlla/Purifier_Code](https://github.com/wljLlla/Purifier_Code)) is empty, so
this implementation follows the paper alone.

The reformer trains on the model's outputs on the validation split, which play the paper's reference non-members.
The same query always gets the same output.

## Usage

```bash
python run.py --defender_mode purifier --dataset cifar10 --attacker_mode lira
```

`neural_feat` reads the model's internal features, which Purifier does not change, so against Purifier it measures
the undefended model.

The defense is {py:class}`~src.mia.defenses.post_hoc.purifier.PurifierDefender`.

## Parameters

`--defender_purifier_reformer_hidden_dim` sets the reformer's hidden size h and
`--defender_purifier_reformer_latent_dim` its latent size. When left unset, they and the reformer's epochs and
learning rate take the appendix's values for the dataset:

```{include} ../_generated/purifier_parameters.md
```

`--defender_purifier_reformer_lambda` weights the label loss; the paper gives no value, and at 1 the reformer outputs
one-hot vectors. `--defender_purifier_reformer_noise_sigma` scales the latent noise. A query matches a swap-set member
when their log-confidences are within `--defender_purifier_swap_threshold` of each other.

```{include} ../_generated/options_purifier.md
```

## Differences from the paper

- The reformer's latent is batch-normalized and gets noise of fixed scale. The paper's reformer has no KL term, so a
  learned noise scale would shrink to zero during training and the reformer would not add noise at all.
