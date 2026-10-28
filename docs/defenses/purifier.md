# Purifier

Purifier ([Yang et al., AAAI 2023](https://arxiv.org/abs/2212.00612)) makes the model's outputs on members look like
its outputs on non-members. A confidence reformer, an autoencoder trained on the model's outputs on non-members,
rewrites each output with Gaussian noise in its latent space, conditioned on the predicted label. A label swapper then
gives a fixed share of the members their second most likely label, so that members are classified correctly no more
often than non-members. Training is unchanged. The paper links to the authors' code at
[github.com/wljLlla/Purifier_Code](https://github.com/wljLlla/Purifier_Code), but that repository is empty, so this
implementation follows the paper alone.

The reformer trains on the model's outputs on the validation split, which plays the paper's reference set of
non-members. The swap set is a random share p_swap = (acc_train − acc_test) / acc_train of the training members,
picked once. Queries return the purified probabilities as log-probabilities, so an attack's softmax gives them back
exactly.

## Usage

```bash
python run.py --defender_mode purifier --dataset cifar10 --attacker_mode lira
```

The defense is {py:class}`~src.mia.defenses.post_hoc.purifier.PurifierDefender`.

## Parameters

The reformer's layers are [2K, h, 2h, 4h, latent, 4h, 2h, h, K] for K classes, with batch normalization and ReLU in the
hidden layers, as in the paper's appendix. `--defender_purifier_reformer_hidden_dim` sets h and
`--defender_purifier_reformer_latent_dim` the latent size. When left unset, they and the reformer's epochs and
learning rate take the appendix's values for the dataset:

```{include} ../_generated/purifier_parameters.md
```

`--defender_purifier_reformer_lambda` weights the label loss of Eq. 1, and `--defender_purifier_reformer_noise_sigma`
scales the latent noise. The noise of each query is seeded by a hash of the input, rounded to
`--defender_purifier_noise_quantization`, so the same query always gets the same output. A query matches a swap-set
member when their log-confidences are within `--defender_purifier_swap_threshold` of each other (L2 distance).

```{include} ../_generated/options_purifier.md
```

## Differences from the paper

The numbers below come from an overfit CIFAR-10 ResNet-18, with 99.75% train and 45.5% test accuracy.

- The reformer is a CVAE with noise in its latent space, but Eq. 1 has no KL term. Without one, a learned variance
  shrinks the noise away (the latent std went from 1 to 5e-6 in one epoch), and a latent of free scale grows until the
  noise has no effect. Adding a KL term makes the decoder ignore the latent instead. We batch-normalize the latent,
  with no learnable scale, and add noise of fixed scale.
- The paper gives no value for λ. We use 0.1: with λ = 1 the label loss dominates and the reformer outputs one-hot
  vectors, which turns Purifier into a one-hot defense. The paper's Fig. 7 shows spread-out confidences instead.
- The reconstruction loss, which the paper calls the L2 norm, is the squared L2 distance of each sample.
- The paper does not say how the latent noise is drawn at inference. We seed it per query, so repeated queries get
  the same output, which the paper requires against replay attacks. With fresh noise, the label changed over 20
  repeated queries for 89% of the swapped members and 12% of the non-members, against 0.3% of the other members.
- acc_test in p_swap is measured on the validation split, because the test split holds the non-members that the
  attacks audit.
- The swapper finds the nearest swap-set member (1-NN) by log-confidences, with a distance threshold; the paper uses
  kNN on confidences and gives no k or threshold. Softmax saturates on confident predictions, so different inputs can
  share a confidence vector: on confidences, 63.5% of the non-members matched a member. Log-confidences of the same
  input differ by at most about 1e-5, and those of different inputs by at least 2.4.
- A swap exchanges the two largest scores of the purified vector, so the output keeps its scores. The paper does not
  say how the label is swapped.
- The reformer outputs K scores. The appendix lists 2K outputs for CIFAR10 and FaceScrub530, but K for Purchase100
  and UTKFace.
- The appendix gives the learning rate 0.0005 "for Facescrub530 and CIFAR10", after 0.01 for CIFAR10. We use it for
  CIFAR-100.
- The paper only considers attacks on the outputs. `neural_feat` reads the trained model's features, which Purifier
  does not change, so it measures the undefended model: what it leaks to an attacker who can read its features.
