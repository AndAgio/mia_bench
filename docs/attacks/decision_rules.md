# Decision rules

The attacks score samples, larger scores for likelier members. A decision rule, set with `--decision_rule`, turns the
scores into decisions: member or non-member. The decisions set accuracy, balanced accuracy, precision, recall, F1,
specificity, the counts of true and false positives and negatives, and the per-sample predictions. The rule does not
change the scores, so AUC, TPR at a fixed FPR, the membership advantage and the ROC curve are the same under every
rule. Most papers compare attacks on these.

## The three rules

- `calibrated`, the default: the threshold is set on samples the attacker knows are not in the target's training set,
  so that a share `--calibration_fpr` ({{calibration_fpr}}) of them score above it. A sample is a member when its score
  is above the threshold. The `--n_calibration_samples` ({{n_calibration_samples}}) calibration samples are held out
  of the attacker's data, which the target never trains on, and the shadow datasets treat them as they treat the
  audited samples. Every attack decides in the same way, at the same false-positive rate, so differences in their
  accuracy or precision come from their scores.
- `attack`: the attack's own rule, as in its paper, made without the audited samples' labels. Each paper draws its
  line in its own place, often at a very different false-positive rate (see the table below), so these decisions
  reproduce a paper rather than compare attacks. Attacks without a rule of their own use the calibrated rule.
- `oracle`: the threshold that maximizes TPR − FPR on the audited samples' membership labels, which no attacker has.
  It is the best any threshold on the scores can do: the strongest attacker a defender may face, not a realistic one.

To compare attacks or defenses, use the default. To reproduce a paper, use `attack`. To see how far any threshold
could go, use `oracle`.

## Running an attack as in its paper

Add `--decision_rule attack` to make the decisions as the attack's paper does:

```bash
python run.py --attacker_mode sba --dataset cifar10 --defender_mode none --decision_rule attack
```

The scores are the paper's under every rule, apart from the differences listed on each attack's page: the rule only
changes the decisions. Several rules give one result each from the same scores, so the attack runs once:

```bash
python run.py --attacker_mode sba --dataset cifar10 --defender_mode none --decision_rule calibrated attack oracle
```

Each result names the rule asked for in `decision_rule`, and the rule applied in `decision_source`: `calibrated`
when an attack without a rule of its own is asked for `attack`. The results also record the threshold, and the
`calibration_fpr` of calibrated decisions. Results from before the decision rules mark the oracle threshold as
`decision_source: "roc_youden"`.

## Each attack's own rule

| Attack | Its own rule (`--decision_rule attack`) | Closest to the paper |
|---|---|---|
| [LiRA](lira.md), [RMIA](robust.md) | None: the calibrated rule | Either: the papers report ROC curves only |
| [Quantile regression](quantile.md) | A member when the score exceeds the sample's own threshold, set for a false-positive rate `--attacker_quantile_alpha` | `attack` |
| [Shadow-model classifier](neural.md) | A member when the attack model gives membership a probability of at least 0.5 | `attack` |
| [Attack-R, Attack-P](attack_r_p.md) | A member when α*, the smallest false-positive rate at which the test calls it a member, is at most `--attacker_rmia_alpha` or `--attacker_pmia_alpha` | `attack` |
| [Supervised boundary distance](sba.md), [noise robustness](noise_robustness.md) | The threshold that best separates a shadow model's members from its non-members, by accuracy | `attack` |
| [Transfer](transfer.md) | The same, on the shadow model, as in the authors' code | `calibrated`: the paper suggests a threshold from shadow samples held out as non-members (Sec. 3.4) |
| [Unsupervised boundary distance](uba.md) | The {{uba_quantile}} quantile of the distances of random inputs: random images, or random records on tabular data | `attack`. The paper covers images only, and the random records follow its random images |
| [DHAttack](dh.md) | Images: the {{dh_synthetic_percentile}}th percentile of the scores of random images. Tabular data: none, the calibrated rule | Both: the paper's main threshold reaches a false-positive rate on a shadow model's non-members (Sec. 3.2), which the calibrated rule matches, and it adds the random images (Sec. 5.3) |
| OSLO, YOQO | The scores are 0 or 1, so each score is the decision | `attack` |

OSLO's and YOQO's scores of 0 or 1 cannot be cut at any false-positive rate. The calibrated rule keeps their
decisions when at most a share `--calibration_fpr` of the calibration samples score 1, and otherwise calls no sample
a member. Scores with few distinct values, such as noise robustness's counts, also put the calibrated rule's actual
false-positive rate near the target rather than on it.

## What the calibrated rule costs

The calibration samples are scored as the audited samples are, so they cost the same queries per sample. Their
queries are counted apart, in `target_model_queries_calibration`. For the label-only attacks that query the target
many times per sample, such as the boundary attacks, scoring them takes as long as scoring as many audited samples.
`--n_calibration_samples 0` holds none out, which leaves only the oracle rule, and the attack rule of attacks with a
rule of their own.

The calibrated rule gives the attacker {{n_calibration_samples}} known non-members of the target. Most attacks here
already hold data from the target's distribution for their shadow or reference models, and the calibration samples
come from that data, so their threat model barely changes. The
[unsupervised boundary attack](uba.md) is the exception: in its paper, the attacker has no data at all. Under the
calibrated rule, it knows non-members that the paper's attacker does not, while `--decision_rule attack` keeps the
paper's attacker.

## Parameters

```{include} ../_generated/options_decision_rules.md
```
