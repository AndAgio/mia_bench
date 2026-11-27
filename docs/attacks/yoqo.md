# YOQO

YOQO ([Wu et al., ICLR 2024](https://tianweiz07.github.io/Papers/24-iclr-1.pdf)) is label-only, and sends the target a
single query per sample. The authors' code is [here](https://github.com/WU-YU-TONG/YOQO).

A sample that a model trains on pushes the model's decision boundary away from it. Around the sample, some inputs then
get the sample's label only from models that trained on it: the paper's improvement area. YOQO searches for a query in
this area on shadow models, and sends it to the target once. The sample is called a member when the target gives the
query the sample's label. The score is that decision, 0 or 1.

The search is gradient descent from the sample, on shadow models trained on the attacker's data with the true labels.
Each shadow model that did not train on the sample, an out-model, should give the query its own most likely label
other than the sample's. The mode sets what else the search asks for:

- `online_yoqo`: the shadow models that trained on the sample, the in-models, should keep the sample's label (the
  paper's Eq. 4). `--attacker_yoqo_alpha` weighs the out-models' part, {{yoqo_alpha}} as in the paper.
- `offline_yoqo`: no shadow model trains on the sample, and the query stays close to it instead (Eq. 5).
  `--attacker_yoqo_gamma` weighs the mean squared distance to the sample, {{yoqo_gamma}} as in the paper.

The losses are scaled as in the authors' code, which the paper's experiments ran: each in/out pair's loss by
{{yoqo_online_scale}}/(1 + α), and each out-model's loss, with the distance counted once per out-model, by
{{yoqo_offline_scale}}/(1 + γ). At the paper's α and γ both scales are 1, and the losses are Eq. 4 and Eq. 5 but for
the distance, which Eq. 5 counts once. Away from them, α and γ also rescale the shadow models' part, which is how they
move the attack's decisions in the paper's Fig. 6. Under Eq. 5 as written, the distance, averaged over the input's
values, is too small to matter: in the test below, every γ from 0 to 5 gave the same decisions.

The search stops once its loss, divided by the number of in/out pairs (online) or of out-models (offline), is at most
`--attacker_yoqo_adv_opt_loss_threshold`, or after `--attacker_yoqo_adv_opt_max_iter` steps. Unset, both are the
paper's: {{yoqo_online_threshold}} and {{yoqo_online_max_iter}} steps online, {{yoqo_offline_threshold}} and
{{yoqo_offline_max_iter}} steps offline. The paper gives its thresholds, {{yoqo_paper_online_loss}} and
{{yoqo_paper_offline_loss}}, for the loss summed over its {{yoqo_paper_pairs}} pairs or out-models; dividing by their
number keeps the stop the same whatever `--n_shadows`. The step size, `--attacker_yoqo_adv_opt_lr` ({{yoqo_lr}}), is the
authors' code's, since the paper does not give it.

The attack trains `--n_shadows` shadow models ({{n_shadows}} unless set), each on its own shadow dataset of
{{shadow_dataset_share}} of the attacker's data, with the attacker's training options (see
[Training defaults](../reference/training.md)). Online, every audited and calibration sample is in exactly half of
the shadow datasets, so by default each sample has 16 in-models and 16 out-models: the paper's 16 pairs. The paper
pairs them more strictly: for each sample, it trains 16 in-models, each on an out-model's data plus the sample, which
would take 16 models per audited sample. The shadow models here serve every sample, as LiRA's do, and the online
attack still reached the paper's accuracy in the test below. Offline, no shadow dataset holds an audited sample, so
every shadow model is an out-model; the paper's offline attack uses 16, with `--n_shadows 16`.

Every decision rule gives the attack's own decisions, since its scores are its decisions. No threshold on scores of 0
or 1 reaches a set false-positive rate, so the calibrated rule keeps them and scores no calibration samples (see
[Decision rules](decision_rules.md)). For the same reason, the AUC is the balanced accuracy, and the attack has a
single operating point: its TPR at 0.1%, 1% and 10% FPR is reported as 0 unless its own false-positive rate is that
low, which the paper notes too (Sec. 4). α (online) and γ (offline) move the operating point, but not that far: in the
test below, the false-positive rate went from 42% (α = 0.5) to 27% (α = 32) online, and from 38% (γ = 5) to 23%
(γ = 0) offline. To compare YOQO with another attack, read the other attack's TPR at YOQO's false-positive rate from
its ROC curve, which its results store, or compare their balanced accuracies.

In a test in the paper's main setting, the authors' CNN7 trained with their recipe on 2,500 CIFAR-10 images (100%
training and 65% test accuracy), with 32 shadow models trained on 2,500 other images plus their half of the audited
samples, and 250 members and 250 non-members audited, the online attack reached 81.2% accuracy and the offline
attack 77.8%. The paper reports 81.65% and 79.19% (Table 2). Calling every correctly classified image a member gave
66.0% (the paper's 67.85%). Both attacks called about a third of the non-members members. With 32 out-models, the
default, instead of the paper's 16, the offline attack reached 80.2%, and called 27% of the non-members members.

The target answers one query per sample. The cost is the shadow models' training, then the search: each step runs
every shadow model forward and backward on the queries. In the test above, the online search for the 500 queries took
about 100 seconds on a laptop GPU.

## Usage

```bash
python run.py --attacker_mode online_yoqo --dataset cifar10 --defender_mode none
```

The scores are the decisions, 0 or 1, so every decision rule gives the paper's decisions (see
[Decision rules](decision_rules.md)). The attack is {py:class}`~src.mia.attacks.label_only.yoqo.Yoqo`.

## Parameters

`--n_shadows` is the number of shadow models, shared with the other attacks that train several. The shadow models'
architecture and training come from the attacker's options, and `--attacker_data_split_perc` sets how much data the
attacker has.

```{include} ../_generated/options_yoqo.md
```
