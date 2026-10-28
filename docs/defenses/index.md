# Defenses

Choose a defense with `--defender_mode`. Only HAMP has its own page so far.

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
| RelaxLoss | `relax_loss` | [paper](https://arxiv.org/abs/2207.05801) |
| SELENA | `selena` | [paper](https://www.usenix.org/system/files/sec22-tang.pdf) |
| MIST | `mist`, `mist_mixup` | [paper](https://www.usenix.org/system/files/usenixsecurity24-li-jiacheng.pdf) |
| Weighted smoothing | `weighted_smoothing` | [paper](https://dl.acm.org/doi/pdf/10.1145/3627106.3627189) |
| MMD | `mmd`, `mmd_mixup` | [paper](https://dl.acm.org/doi/pdf/10.1145/3422337.3447836) |
| MemGuard | `mem_guard` | [paper](https://dl.acm.org/doi/pdf/10.1145/3319535.3363201) |
| LDL | `ldl` | [paper](https://arxiv.org/abs/2212.01688) |
| Purifier | `purifier` | [paper](https://arxiv.org/abs/2212.00612) |
| [HAMP](hamp.md) | `hamp`, `hamp_train`, `hamp_test` | [paper](https://arxiv.org/abs/2307.01610) |

```{toctree}
:hidden:

hamp
```
