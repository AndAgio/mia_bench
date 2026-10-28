# LDL

LDL ([Rajabi et al., AsiaCCS 2023](https://arxiv.org/abs/2212.01688)) defends against label-only attacks, which
estimate how far each sample lies from the decision boundary. It answers each query with the model's logits averaged
over noisy copies of the input, so the predicted label stays the same in a sphere around each sample, for members and
non-members alike. Training is unchanged. The authors' code is
[here](https://github.com/rajabia/LDL-A-Defense-for-Label-Based-Membership-Inference-Attacks).

Every query runs the model on all its noisy copies, which go through together, up to the training batch size per
pass. When an attack needs input gradients, they are averaged over the copies too.

## Usage

```bash
python run.py --defender_mode ldl --dataset cifar10 --attacker_mode noise_robust
```

The defense is {py:class}`~src.mia.defenses.post_hoc.ldl.LdlDefender`.

## Parameters

Image datasets get Gaussian noise. `--defender_ldl_noise_scale` is its standard deviation in pixel units of images in
[0, 1]. Since our models take normalized images, the noise is divided by each channel's normalization std. The binary
features of Purchase and Texas get bit flips instead: each feature flips with probability `--defender_ldl_flip_prob`.
`--defender_ldl_n_queries` sets the number of noisy copies per query. When left unset, the number of copies follows
the authors' code, and the flip probability follows Table 1 of the paper:

```{include} ../_generated/ldl_parameters.md
```

```{include} ../_generated/options_ldl.md
```

## Differences from the paper

- The paper releases only the predicted label. We return the averaged logits, so that every attack can run against
  LDL. Label-only attacks take their argmax and see exactly the paper's defense. Against score-based attacks, LDL is a
  smoothing defense that the paper does not evaluate; report these results as an extension.
- The paper gives the noise as σ² = 0.02, but the authors' code uses 0.02 as the standard deviation. We follow the
  code.
- The number of copies follows the authors' code. The paper gives none for images, and uses 10000 for the binary
  datasets.
- News, which the paper does not cover, gets Gaussian noise, and `--defender_ldl_noise_scale` is in the units of its
  TF-IDF features.
- A flipped feature becomes 1 − x. On binary features this is the paper's flip. Label-only attacks also query points
  between binary inputs (noise robustness, HopSkipJump), which then get the same noise as the binary points nearby.
  The authors' code would first round any non-zero feature up to 1.
- The paper only considers attacks on the labels. `neural_feat` reads the trained model's features, which LDL does
  not change, so it measures the undefended model: what it leaks to an attacker who can read its features.
