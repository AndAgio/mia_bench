# Shadow-model classifier

The shadow-model attack ([Shokri et al., IEEE S&P 2017](https://arxiv.org/abs/1610.05820)) learns what membership
looks like from shadow models. It trains `--n_shadows` shadow models on the attacker's data, queries each on samples
it was and was not trained on, and trains attack models to tell the two apart from the outputs. As in the paper, there
is one attack model per class, and the attack model of a sample's true class scores the target's output on it. The
co-author's example code is [here](https://github.com/csong27/membership-inference).

The mode sets what the attack models read:

- `neural_prob` reads the probabilities, as in the paper.
- `neural_logit` reads the logits.
- `neural_feat` reads the model's internal features, which assumes the attacker can read them (see
  [Attacks](index.md)).

Each mode also has an online version, `online_neural_prob`, `online_neural_logit` and `online_neural_feat`. In the
offline modes, as in the paper, the shadow datasets hold none of the audited samples. In the online modes, each
audited sample is in half of the shadow models, as in the version of the attack that
[Carlini et al.](https://arxiv.org/abs/2112.03570) evaluate (Sec. V-C). The paper notes that the attack only gets
stronger when the shadow and target datasets overlap. Carlini et al. train one attack model that also reads the label;
here both versions keep the paper's model per class. Online, the shadow models must be trained again for every new
set of audited samples.

Each shadow model gives one training row per sample of the shadow datasets: a member if the sample is in its own
dataset, a non-member otherwise. The attack thus needs at least 2 shadow models. Each shadow dataset holds
{{shadow_dataset_share}} of the attacker's data, so members outnumber non-members. The loss weighs both groups
equally, as the paper's shadow training and test sets of the same size do, so that the attack's own decisions are not
biased towards members. These decisions call a sample a member when its attack model gives membership a probability
of at least 0.5.

The cost is mostly the training of the shadow models, with the attacker's training options (see
[Training defaults](../reference/training.md)). Each shadow model then makes one pass over the shadow datasets, and
the target one pass over the audited samples. The rows are built one class at a time, so memory holds the outputs of
every shadow model on the samples of one class only. For example, `neural_feat` on ResNet-18, with 32 shadow models on
CIFAR-10, holds about 200 MB of rows.

## Usage

```bash
python run.py --attacker_mode neural_prob --dataset cifar10 --defender_mode none
```

The attack is {py:class}`~src.mia.attacks.black_box.neural.NeuralMIA`.

## Parameters

`--n_shadows` is the number of shadow models, shared with the other attacks that train several. The attack models are
trained with Adam and the binary cross-entropy. The paper's attack model has a single hidden layer of 64 units, which
`--attacker_neural_model_layers 64` reproduces. The default, deeper network did as well or better in a test on
CIFAR-10, most of all on logits.

```{include} ../_generated/options_neural.md
```
