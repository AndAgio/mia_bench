# Defenses

Choose a defense with `--defender_mode`. Only RelaxLoss, SELENA, MIST, MemGuard, LDL, Purifier and HAMP have their own pages so far.

Every defense can be paired with every attack, including attacks outside the defense's threat model. LDL, MemGuard,
Purifier and `hamp_test` act only at inference, so the trained model's internal features are not defended. `neural_feat`
reads those features: against these defenses, it measures what the undefended model leaks to an attacker who can read
them. The run log notes these pairings.

| Defense | `--defender_mode` | Paper |
|---|---|---|
| No defense | `none` | |
| Differential privacy | `dp` | |
| Mixup | `mixup` | |
| Data augmentation | `data_augmentation` | |
| Adversarial regularization | `adv_reg` | [paper](https://dl.acm.org/doi/pdf/10.1145/3243734.3243855) |
| [RelaxLoss](relax_loss.md) | `relax_loss` | [paper](https://arxiv.org/abs/2207.05801) |
| [SELENA](selena.md) | `selena` | [paper](https://arxiv.org/abs/2110.08324) |
| [MIST](mist.md) | `mist`, `mist_mixup` | [paper](https://arxiv.org/abs/2311.00919) |
| Weighted smoothing | `weighted_smoothing` | [paper](https://dl.acm.org/doi/pdf/10.1145/3627106.3627189) |
| MMD | `mmd`, `mmd_mixup` | [paper](https://dl.acm.org/doi/pdf/10.1145/3422337.3447836) |
| [MemGuard](memguard.md) | `mem_guard` | [paper](https://arxiv.org/abs/1909.10594) |
| [LDL](ldl.md) | `ldl` | [paper](https://arxiv.org/abs/2212.01688) |
| [Purifier](purifier.md) | `purifier` | [paper](https://arxiv.org/abs/2212.00612) |
| [HAMP](hamp.md) | `hamp`, `hamp_train`, `hamp_test` | [paper](https://arxiv.org/abs/2307.01610) |

```{toctree}
:hidden:

relax_loss
selena
mist
memguard
ldl
purifier
hamp
```
