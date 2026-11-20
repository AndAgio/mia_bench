# Adversarial regularization

Adversarial regularization ([Nasr, Shokri and Houmansadr, CCS 2018](https://arxiv.org/abs/1807.05852)) trains the
model together with a membership inference attack, as a min-max game. An attack model learns to tell training members
from reference non-members, from the model's prediction and the label, and the model's loss is penalized when the
attack succeeds. Inference is unchanged. The authors' code is
[here](https://github.com/SPIN-UMass/ML-Privacy-Regulization).

Each classifier step follows k attack model steps, so training costs more than normal training. The reference
non-members are half of the validation split; the other half picks the best checkpoint.

## Usage

```bash
python run.py --defender_mode adv_reg --dataset cifar10 --attacker_mode lira
```

The defense is {py:class}`~src.mia.defenses.training.adversarial_regularization.AdvRegDefender`.

## Parameters

`--defender_adv_reg_lambda` is λ, the weight of the penalty in the classifier loss, and
`--defender_adv_reg_attacker_lr` the learning rate of the attack model. When left unset, λ takes the values of the
paper's Table 3, and the learning rate that of the authors' Purchase code:

```{include} ../_generated/adv_reg_parameters.md
```

`--defender_adv_reg_penalty` chooses the penalty on the attack model's output h: λ·h, as in the authors' code, or
the paper's λ·log h, which has no lower bound and can drive the classifier to chance accuracy.
`--defender_adv_reg_shadow_attacker_k` is k. When `--defender_adv_reg_shadow_attacker_model_layers` is left unset,
the attack model is the paper's (Fig. 3); when set, it is a plain MLP with these hidden sizes.

```{include} ../_generated/options_adv_reg.md
```
