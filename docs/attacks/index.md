# Attacks

Choose an attack with `--attacker_mode`. Only LiRA, RMIA, quantile regression, the shadow-model classifier, Attack-R,
Attack-P, the supervised and unsupervised boundary distances, noise robustness, the transfer attack, DHAttack and YOQO
have their own pages so far.

Every attack can be paired with every defense. Score-based attacks read the scores the defended model returns, and
label-only attacks read only its predicted labels. `neural_feat` reads the model's internal features instead. LDL,
MemGuard, Purifier and `hamp_test` act only at inference and leave these features undefended. Against them,
`neural_feat` measures what the undefended model leaks to an attacker who can read its features. The run log notes
these pairings.

| Attack | `--attacker_mode` | Reads | Paper |
|---|---|---|---|
| [LiRA](lira.md) | `lira`, `offline_lira` | scores | [paper](https://arxiv.org/abs/2112.03570) |
| [RMIA](robust.md) | `online_robust`, `offline_robust` | scores | [paper](https://arxiv.org/abs/2312.03262) |
| [Quantile regression](quantile.md) | `quantile` | scores | [paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/01328d0767830e73a612f9073e9ff15f-Abstract-Conference.html) |
| [Shadow-model classifier](neural.md) | `neural_prob`, `neural_logit`, `neural_feat`, `online_neural_*` | scores, or features | [paper](https://arxiv.org/abs/1610.05820) |
| [Attack-R](attack_r_p.md) | `rmia_loss`, `rmia_confidence`, `rmia_entropy` | scores | [paper](https://arxiv.org/abs/2111.09679) |
| [Attack-P](attack_r_p.md) | `pmia_loss`, `pmia_confidence`, `pmia_entropy` | scores | [paper](https://arxiv.org/abs/2111.09679) |
| [Boundary distance, supervised](sba.md) | `sba`, `sba_qeba`, `sba_qeba-dct`, `sba_qeba-pca`, `sba_qeba-custom` | labels | [paper](https://arxiv.org/abs/2007.14321) |
| [Boundary distance, unsupervised](uba.md) | `uba`, `uba_qeba`, `uba_qeba-dct`, `uba_qeba-custom` | labels | [paper](https://arxiv.org/abs/2007.15528) |
| [Noise robustness](noise_robustness.md) | `noise_robust` | labels | [paper](https://arxiv.org/abs/2007.14321) |
| [Transfer](transfer.md) | `transfer_loss`, `transfer_confidence`, `transfer_entropy` | labels | [paper](https://arxiv.org/abs/2007.15528) |
| OSLO | `oslo`, `oslo_mifgsm`, `oslo_tifgsm`, `oslo_tmifgsm` | labels | [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/71f88122d414cfeb455ac0ed932fbe1f-Paper-Conference.pdf) |
| [DHAttack](dh.md) | `dh`, `dh_black`, `dh_random` | labels | [paper](https://www.usenix.org/system/files/usenixsecurity25-li-hao.pdf) |
| [YOQO](yoqo.md) | `online_yoqo`, `offline_yoqo` | labels | [paper](https://tianweiz07.github.io/Papers/24-iclr-1.pdf) |

The `rmia_*` modes run Attack-R, not RMIA. `sba` and `uba` find the boundary with HopSkipJump, and the `_qeba` modes
with QEBA.

## Decisions

By default, every attack's decisions come from the same calibrated rule: a threshold set on samples the attacker knows
are non-members, at the same false-positive rate for all attacks. To run an attack as in its paper, with the paper's
own decisions, add `--decision_rule attack`. [Decision rules](decision_rules.md) explains the rules and lists each
attack's own rule. The rule changes only the decisions, and so accuracy, precision, recall and F1: AUC and TPR at a
fixed FPR are the same under every rule.

## An attacker with little data

`--attacker_data_split_perc` sets the attacker's share of the dataset. With a small share, three of its uses can need
more samples than the attacker has, and the run then stops with an error naming the option to change:

- The calibration samples are held out of the attacker's data first. Pass `--n_calibration_samples` as a share of that
  data, such as 0.2, instead of a number of samples.
- The online attacks put every audited and calibration sample in their shadow datasets, so they need fewer audited
  samples (`--n_auditing_samples`).
- RMIA draws its population from the attacker's data, so it needs a smaller `--attacker_robust_rand_pop_size`.

For example, `--attacker_data_split_perc 0.01 --n_calibration_samples 0.2` gives the attacker 600 CIFAR-10 images, of
which 120 calibrate the thresholds.

```{toctree}
:hidden:

decision_rules
lira
robust
quantile
neural
attack_r_p
sba
uba
noise_robustness
transfer
dh
yoqo
```
