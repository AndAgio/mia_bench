# Adversarial regularization

Adversarial regularization ([Nasr, Shokri and Houmansadr, CCS 2018](https://arxiv.org/abs/1807.05852)) trains the
model together with a membership inference attack, as a min-max game. The attack model h reads the prediction vector
and the one-hot label of a sample, and estimates the probability that the sample is a member. Each classifier step
follows k attack model steps (Algorithm 1). An attack model step trains h with the log loss, on a fresh batch of
members from the training set and an equal batch from a reference set of non-members. The classifier step then adds
a penalty on h to the loss of a fresh training batch, which pushes the members' outputs toward those that h gives to
non-members: λ·log h in the paper, and λ·h in the authors' code, which we follow. Inference is unchanged. The
authors' code is [here](https://github.com/SPIN-UMass/ML-Privacy-Regulization).

The reference set is half of the validation split, drawn per class, and the other half selects the best checkpoint.
Neither overlaps the test split, which holds the audited non-members. The attack model reads softmax probabilities,
not logits: shifting all the logits of a sample by a constant changes them without changing the prediction, so the
classifier could fool an attack on logits without leaking less.

Members and reference samples go through the classifier in one forward pass, in the attack model steps as in the
classifier step, so that they share the BatchNorm batch statistics: with separate passes, the statistics alone would
tell the members apart. The attack model steps leave the BatchNorm running statistics unchanged.

## Usage

```bash
python run.py --defender_mode adv_reg --dataset cifar10 --attacker_mode lira
```

The defense is {py:class}`~src.mia.defenses.training.adversarial_regularization.AdvRegDefender`.

## Parameters

`--defender_adv_reg_lambda` is λ, the weight of the penalty in the classifier loss, and
`--defender_adv_reg_attacker_lr` the learning rate of the attack model's Adam optimizer. When left unset, λ takes
the values of Table 3, and the learning rate that of the authors' Purchase code:

```{include} ../_generated/adv_reg_parameters.md
```

`--defender_adv_reg_penalty` is `linear` for the authors' λ·h, or `log` for the paper's λ·log h.
`--defender_adv_reg_shadow_attacker_k` is k, the number of attack model steps per classifier step. When
`--defender_adv_reg_shadow_attacker_model_layers` is left unset, the attack model is the paper's (Fig. 3): a branch
for the prediction vector (C → 1024 → 512 → 64), one for the label (C → 512 → 64), and a common one on both outputs
(128 → 256 → 64 → 1), with ReLU activations, weights drawn from N(0, 0.01) and zero biases. When set, it is a plain
MLP with these hidden sizes on the prediction vector and the label.

```{include} ../_generated/options_adv_reg.md
```

## Differences from the paper

- The penalty is λ·h, as in the authors' Purchase and Texas code, not the paper's λ·log h, which has no lower bound:
  the classifier can keep lowering it by giving up on the task. We checked both with ResNet-18, trained with Adam for
  60 epochs on 5,000 CIFAR-10 samples, at λ = 6. With λ·log h the classifier ended at chance accuracy, and its best
  checkpoint lost 4.7 more points of test accuracy than λ·h for the same attack accuracy. The authors' CIFAR penalty,
  λ·h² in our notation, collapsed as well.
- The attack model learns at 0.0001, as in the authors' Purchase code, not at the paper's 0.001 (Sec. 5.3). In the
  same check, at 0.001 a whole layer of the attack model's prediction branch stopped activating early, and the
  defense did nothing. At the code's CIFAR value, 0.00001, the attack model learned too late to defend.
- We keep the model with the best validation performance, as for every defense here. The paper does not say, and
  the Purchase script loads the checkpoint of epoch 13. With the defaults, this matters: in the check, test accuracy
  peaked at epoch 10 and then degraded, to 41% at epoch 60. The best model lowers a Mentr threshold attack from 81.5%
  to 59.2%, for a test accuracy of 58.8% against 63.3% without defense.
- The paper gives no k. We take 1, as the authors' CIFAR script; their Purchase and Texas scripts take 52 attack
  model batches for every 2 classifier batches.
- CIFAR-10 and the other image datasets take the λ of CIFAR-100, the only image dataset in the paper. CIFAR-10's
  cross-entropy is smaller, so the same λ weighs more.
- The reference set is half of the validation split, so its size follows the benchmark's splits. The paper uses
  20,000 samples on Purchase and 5,000 on Texas and CIFAR-100 (Table 3), and the authors' CIFAR script uses the test
  set, on which it also evaluates the attack.
- Elsewhere the authors' code departs from the paper, and we follow the paper: the code trains its attack model with
  the squared error on logits, uses λ = 1 on Texas and CIFAR, and trains the classifier normally for 1 or 4 epochs
  first.
