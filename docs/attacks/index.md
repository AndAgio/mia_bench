# Attacks

Choose an attack with `--attacker_mode`. Only LiRA, RMIA, quantile regression, the shadow-model classifier, Attack-R,
Attack-P and the supervised boundary distance have their own pages so far.

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
| Boundary distance, unsupervised | `uba`, `uba_qeba`, `uba_qeba-dct`, `uba_qeba-pca`, `uba_qeba-custom` | labels | [paper](https://dl.acm.org/doi/pdf/10.1145/3460120.3484575) |
| Noise robustness | `noise_robust` | labels | [paper](https://proceedings.mlr.press/v139/choquette-choo21a/choquette-choo21a.pdf) |
| Transfer | `transfer_loss`, `transfer_confidence`, `transfer_entropy` | labels | [paper](https://dl.acm.org/doi/pdf/10.1145/3460120.3484575) |
| OSLO | `oslo`, `oslo_mifgsm`, `oslo_tifgsm`, `oslo_tmifgsm` | labels | [paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/71f88122d414cfeb455ac0ed932fbe1f-Paper-Conference.pdf) |
| DHAttack | `dh`, `dh_black`, `dh_random` | labels | [paper](https://www.usenix.org/system/files/usenixsecurity25-li-hao.pdf) |
| YOQO | `online_yoqo`, `offline_yoqo` | labels | [paper](https://tianweiz07.github.io/Papers/24-iclr-1.pdf) |

The `rmia_*` modes run Attack-R, not RMIA. `sba` and `uba` find the boundary with HopSkipJump, and the `_qeba` modes
with QEBA.

```{toctree}
:hidden:

lira
robust
quantile
neural
attack_r_p
sba
```
