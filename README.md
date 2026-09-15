
# 🛡️ Membership Inference Attacks Benchmarking Framework

This repository provides a **modular, scalable, and state‑of‑the‑art framework for evaluating Membership Inference Attacks (MIAs)** against neural network models. It supports realistic attacker/defender threat models, flexible datasets and architectures, and modern optimizer backends. This repository is a  research-focused framework supporting the quick definition and implementation of novel attack strategies by wrapping attacker steps into two main methods (attack optimization and its deployment).


## ✨ Key Features

- **Model & dataset flexibility**: easily add new architectures and datasets
- **Explicit attacker/defender modeling** with realistic knowledge assumptions
- **Support for modern optimizers**, including SAM, ASAM, GSAM
- **Highly modular attack design** through attacker, defender, and shadow managers
- **Fast experimentation** with optimized shadow‑model training
- **Distributed training and Apple Silicon support**
- **Easy experiment resuming and organized outputs**
- **Built‑in privacy metrics** such as AUC and TPR@FPR


## 📁 Repository Structure

```
models/        # Model definitions (defender and attacker)
data/          # Dataset wrappers
mia/           # MIA implementations
optimizers/    # SAM, ASAM, GSAM, ...
trainer/       # Module containing code to train defender and attacker models
utils/         # Helper functions and classes
outs/          # Outputs
run.py         # Main file to train defender model and run the MIA
```


## 🧠 Core Classes

- **Victim**: target model under attack; handles training, evaluation, and checkpointing.
- **Attacker**: defines the inference strategy and trains the attack model.
- **ShadowDataManager**: samples and manages shadow datasets.
- **ShadowModelManager**: trains and orchestrates shadow models.
- **AuditDataManager**: constructs audit datasets combining member and non‑member samples.


## 🚀 Quick Start


### Quick overview
- Training entrypoint: `train_defender.py` — trains only the defender model and saves checkpoints.
- End-to-end pipeline: `run.py` — trains defender, (optionally) shadow models, runs attacker optimization, and saves results.
- Helpers: `download_all_datasets.py`, `generate_run_jobs.py` (SLURM job generator).
- Configuration and CLI flags are defined in `src/utils/settings.py`.

### Installation

```bash
git clone https://github.com/AndAgio/mia_bench.git
cd mia_bench
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If using macOS Apple Silicon, you can enable MPS fallback (optional):

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Datasets and outputs
- Default datasets folder: `datas/` (see `src/utils/variables.py`).
- Default outputs folder: `outs/` (experiment results, attacker outputs, checkpoints).

Each completed attack writes four files under its experiment `results/` directory:

- `results.json`: complete metrics, ROC arrays, ID groups, and per-sample predictions.
- `summary.json` and `summary.csv`: convenient aggregate metrics (AUC, model and attack
  accuracy, balanced accuracy, precision/recall/F1, confusion counts, and TPR at low FPR).
- `membership_predictions.jsonl`: one line per audited sample, containing its original
  dataset ID, true/predicted membership, attack score, and whether it was identified correctly.

To inspect a result without manually parsing JSON, run:

```bash
python summarize_results.py path/to/experiment/results --ids members
```

Use `--ids correct` for every correct member/non-member decision, or `--ids members`
for successfully recovered training members only. Other choices are `non-members`,
`false-positives`, and `missed-members`.

`model_accuracy` is evaluated on the test split after the selected defence has been
applied. The best metric observed while training is retained separately under the
`training_best_*` fields. For attacks without a native membership threshold, ID-level
decisions use the ROC threshold that maximizes Youden's J and are marked with
`decision_source: "roc_youden"`; calibrated attacks are marked `"attack"`.

#### Metrics and metadata retained

The following information is produced centrally for every attack and every selected
defence used through `run.py`.

- **Overall MIA effectiveness:** ROC AUC, attack accuracy, balanced accuracy,
  precision, recall/member TPR, F1, specificity/non-member TNR, maximum membership
  advantage (`TPR - FPR`), TPR at FPR 0.1%, 1%, and 10%, and the full ROC arrays
  (`tpr`, `fpr`, and thresholds in `roc`).
- **Counts:** audit sample, member, and non-member counts; true positives, true
  negatives, false positives, false negatives, and recovered-member count.
- **Decision provenance:** whether decisions came from the attack's calibrated
  threshold or a post-hoc Youden-J ROC threshold, plus that threshold when applicable.
- **Dataset ID groups:** all correctly identified IDs, successfully recovered member
  IDs, correctly rejected non-member IDs, false-positive IDs, and missed-member IDs.
- **Per audited sample:** original dataset ID, original class label, true membership,
  membership score, predicted membership, whether the membership decision was correct,
  and whether the defended target model classified the sample correctly.
- **Per-class privacy:** sample/member/non-member counts, AUC, attack and balanced
  accuracy, precision, recall, F1, specificity, confusion counts, and TPR at FPR 0.1%,
  1%, and 10% for every class. We also retain macro class AUC, minimum/maximum class
  AUC, the most vulnerable class label, and the maximum class TPR at each FPR target.
- **By model correctness:** the same group MIA metrics separately for samples that the
  defended model classified correctly and incorrectly.
- **Defended-model utility and generalization:** final train/test accuracy, final
  train/test loss, accuracy gap (`train - test`), loss gap (`test - train`), and the
  best metric, metric name, split, and epoch observed during model training.
- **Attack and defence cost:** defender training time, defence application time,
  attacker optimization time, attack evaluation time, actual target-model queries,
  mean target queries per audit sample, configured maximum queries per sample, audit
  throughput, instantiated shadow-model count, samples per shadow dataset, reference
  population size where applicable, process peak memory, CUDA peak memory, and
  recovered members per 1,000 target queries.
- **Experiment identity:** attack and defence names and complete configurations,
  defender architecture, defender/attacker training configurations, dataset, data
  splits, audit composition and size, configured shadow count, seed, and any swept
  attack parameters such as RobustMIA's `alpha`.

To aggregate repeated target-model seeds, point the summarizer at their common output
root. It groups matching attack, defence, model architecture, and attack parameters and
writes mean, sample standard deviation, and two-sided 95% Student-t confidence intervals
to `aggregate_summary.json` and `aggregate_summary.csv`:

```bash
python summarize_results.py path/to/repeated/runs --aggregate
```

For each group of genuinely comparable runs, the uncertainty report retains the number
of runs and seeds plus the mean, sample standard deviation, and two-sided 95% Student-t
confidence interval for:

- AUC; TPR at FPR 0.1%, 1%, and 10%; attack accuracy; final defended-model accuracy;
  and recovered-member count.
- Train/test accuracy and loss, accuracy and loss generalization gaps.
- Attack optimization/evaluation time, total and mean target-query cost, audit
  throughput, and recovered members per 1,000 target queries.

A single run is reported with zero standard deviation and a point confidence interval;
meaningful uncertainty requires repeated seeds or independently trained target-model
replicas. Runs with different datasets, models, attack/defence settings, audit settings,
or training configurations are deliberately kept in separate aggregate groups.
- Use `python download_all_datasets.py` to download supported datasets to the default `datas/` folder.

### Local smoke matrix

`smoke_test_all.py` exercises every distinct attack and defense and keeps going after
individual failures. The default `quick` preset uses one epoch and tiny inputs. After
that passes, the `medium` preset provides a moderately realistic stress check:

```bash
python smoke_test_all.py --preset medium --device 0
```

The medium preset uses ten epochs for defender, attacker, neural attacker, MemGuard,
MIST, and Purifier training. Its three profiles use 1,600–2,400 samples, batches of
32–64, 64–128 auditing samples, standard-sized auxiliary networks, 50 quantiles or
queries for the common query-driven methods, and moderately larger OSLO, DH, SELENA,
MIST, YOQO, Purifier, and LDL workloads. To try only a few components first:

```bash
python smoke_test_all.py --preset medium --device cpu \
	--only attack:quantile attack:lira defense:mist
```

Use `--dry-run` to inspect the generated commands without training, and
`--list-components` to see all accepted component names. Each case uses an isolated
temporary work directory; its checkpoints, shadow data, generated result artifacts,
and internal logs are deleted as soon as the case finishes. The top-level case logs
and `report.json` are retained. Pass `--keep-work` when you need the full artifacts to
debug a failure.

`--preset moderate` remains accepted as a backward-compatible alias for `medium`.

### Training defenders (and checkpointing for reuse)
Use `train_defender.py` when you only want to train the defender model and persist checkpoints for later reuse across multiple attacks. This is the recommended workflow for large experiments: train a defender once with stable settings, then run several attacks reusing that checkpoint.

Example — train a defender and store checkpoints:

```bash
# Train only the defender with standard SGD and no defense
python train_defender.py \
	--dataset cifar10 \
	--defender_model resnet18 \
	--defender_epochs 100 \
	--defender_batch_size 128 \
	--device 0
```

The repository also supports training a defender model with Differential Privacy (DP) by enabling `--defender_use_dp` and the related DP flags which instruct the training pipeline to use the configured DP mechanism (see `src/utils/configs.py` where `DPConfigs` are assembled). Example:

```bash
# Train only the defender with DP-SGD enabled
python train_defender.py \
	--dataset cifar10 \
	--defender_model resnet18 \
	--defender_epochs 50 \
	--defender_batch_size 128 \
	--defender_use_dp \
	--defender_dp_noise_multiplier 1.2 \
	--defender_dp_max_grad_norm 1.0 \
	--defender_dp_clip_per_layer \
	--device 0
```

*Note*: depending on implementation and dependency (e.g., Opacus), extra packages may be required for DP training; check your environment if you plan to run DP experiments.

#### Notes on checkpointing and `--resume`:
- When you run any command the repository computes deterministic hashes from the *relevant* CLI settings and saves settings and checkpoints under `outs/` using these hashes (see `src/utils/configs.py`).
- Victim checkpoints are stored under `outs/defenders/<defender_hash>/ckpts` and attacker checkpoints under `outs/attackers/<attacker_hash>/ckpts`.
- The hash is computed by `get_hash_from_settings(...)`, which first filters settings with `get_relevant_settings(...)` and then hashes the JSON representation (sorted keys) — so the hash changes if any relevant setting changes.
- `--resume` will only reuse checkpoints whose settings produce the exact same hash (i.e. identical relevant settings). If settings differ, training starts from scratch and new folders are created.

Refer to [src/utils/configs.py](src/utils/configs.py) and [src/utils/settings.py](src/utils/settings.py) for the exact rules used to compute which flags are considered "relevant" for defender/attacker/experiment hashes.

### Two concrete Quick-Start examples

These examples are minimal working commands that use the repository's CLI flags (see `src/utils/settings.py` for the full list).

Example A — Quantile attack on CIFAR-10 (fast, minimal shadows)

```bash
# download data (one-time)
python download_all_datasets.py

# run a short end-to-end experiment: defender + attack (quantile)
python run.py --attacker_mode quantile --dataset cifar10 --defender_model resnet18 --attacker_model resnet18 --defender_epochs 10 --attacker_epochs 5 --n_shadows 1 --n_samples_per_shadow_dataset 2000 --n_auditing_samples 1000 --device 0
```

Example B — LiRA attack on TinyImageNet (larger experiment)

```bash
# run a larger experiment for LiRA on TinyImageNet
python run.py \
	--attacker_mode lira \
	--dataset tinyimagenet \
	--defender_model resnet50 \
	--attacker_model resnet50 \
	--defender_epochs 90 \
	--attacker_epochs 30 \
	--n_shadows 50 \
	--n_samples_per_shadow_dataset 10000 \
	--n_auditing_samples 5000 \
	--defender_batch_size 256 \
	--attacker_batch_size 256 \
	--device 0
```

Notes:
- LiRA and other heavy attacks typically require many shadow models and larger shadow datasets — expect increased runtime and GPU memory usage.


### Options and defaults
The full CLI options and defaults can be found in [src/utils/settings.py](src/utils/settings.py), while the default folders and paths are in [src/utils/variables.py](src/utils/variables.py).

#### Important CLI flags (high level)
Below are the most commonly used flags. See `src/utils/settings.py` for the full authoritative list and exact defaults.

- `--dataset`: dataset name (cifar10, cifar100, svhn, fmnist, cinic10, imagenet, tinyimagenet)
- `--defender_model`: defender architecture (resnet18, resnet50, vgg16, mobile_small, etc.)
- `--defender_epochs`, `--defender_batch_size`: defender training schedule and batch size
- `--defender_lr`, `--defender_lr_sched`: defender learning rate and scheduler
- `--defender_use_dp`, `--defender_dp_noise_multiplier`, `--defender_dp_max_grad_norm`, `--defender_dp_clip_per_layer`: differential privacy training options for the defender
- `--attacker_model`, `--attacker_epochs`, `--attacker_batch_size`, `--attacker_lr`: attacker model and training settings
- `--attacker_mode`: attack strategy (`quantile`, `lira`, `neural_feat`, `rmia_loss`, `pmia_confidence`, ...)
- `--n_shadows`, `--n_samples_per_shadow_dataset`: number of shadow datasets/models and their size (some attacks override these defaults)
- `--n_auditing_samples`, `--audit_in_perc`: auditing set size and in-percentage
- `--device`: device to use (`0`, `1`, `cpu`, `mps`)
- `--resume`: attempt to resume from existing checkpoints for the current relevant settings (see checkpointing note above)
- `--datasets_folder`, `--out_folder`: override default dataset and outputs folders

**Tip:** to get a full list, run any script with `--help` (e.g., `python run.py --help`) or inspect [src/utils/settings.py](src/utils/settings.py).



## 🛣️ Roadmap

- [ ] Label‑only MIAs
- [ ] Differential privacy training support
- [ ] Automated hyperparameter tuning
- [ ] Visualization dashboard



## 🤝 Contributions

Contributions are welcome via issues or pull requests.
