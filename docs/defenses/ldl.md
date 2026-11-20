# LDL

LDL ([Rajabi et al., AsiaCCS 2023](https://arxiv.org/abs/2212.01688)) defends against label-only attacks, which
estimate how far each sample lies from the decision boundary. It answers each query with the model's logits averaged
over noisy copies of the input, so the predicted label stays the same in a sphere around each sample, for members and
non-members alike. Training is unchanged. The authors' code is
[here](https://github.com/rajabia/LDL-A-Defense-for-Label-Based-Membership-Inference-Attacks).

The paper releases only the predicted label. Here LDL returns the averaged logits, so that score-based attacks can run
against it too; label-only attacks take their argmax and see exactly the paper's defense. Every query runs the model
on all its noisy copies, which makes queries slower.

## Usage

```bash
python run.py --defender_mode ldl --dataset cifar10 --attacker_mode noise_robust
```

`neural_feat` reads the model's internal features, which LDL does not change, so against LDL it measures the
undefended model.

The defense is {py:class}`~src.mia.defenses.post_hoc.ldl.LdlDefender`.

## Parameters

Image datasets get Gaussian noise. `--defender_ldl_noise_scale` is its standard deviation in pixel units of images in
[0, 1]. The binary features of Purchase and Texas get bit flips instead: each feature flips with probability
`--defender_ldl_flip_prob`. News gets Gaussian noise, in the units of its TF-IDF features. `--defender_ldl_n_queries`
sets the number of noisy copies per query. When left unset, the number of copies follows the authors' code, and the
flip probability follows Table 1 of the paper:

```{include} ../_generated/ldl_parameters.md
```

```{include} ../_generated/options_ldl.md
```
