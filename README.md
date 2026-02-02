
# 🛡️ Membership Inference Attacks Benchmarking Framework with Metric privacy

## 🛡️ Metric Privacy protection 
To run the mechanism use:
- `--use_metric`: Uses the defence (flag)
-  `--metric_b`: Set the noise multiplier b of the Laplace distribution (float)
-  `--metric_d`: Set the L1 distance radius to protect (float)

The code will automatically calculate the epsilon and delta that you get, based on the theoretical analysis. 


####################################################

The rest remains the same as the main branch:


## ✨ Key Features

- **Model & dataset flexibility**: easily add new architectures and datasets
- **Explicit attacker/victim modeling** with realistic knowledge assumptions
- **Support for modern optimizers**, including SAM, ASAM, GSAM
- **Highly modular attack design** through attacker, victim, and shadow managers
- **Fast experimentation** with optimized shadow‑model training
- **Distributed training and Apple Silicon support**
- **Easy experiment resuming and organized outputs**
- **Built‑in privacy metrics** such as AUC and TPR@FPR


## 📁 Repository Structure

```
models/        # Model definitions (victim and attacker)
data/          # Dataset wrappers
mia/           # MIA implementations
optimizers/    # SAM, ASAM, GSAM, ...
trainer/       # Module containing code to train victim and attacker models
utils/         # Helper functions and classes
outs/          # Outputs
run.py         # Main file to train victim model and run the MIA
```


## 🧠 Core Classes

- **Victim**: target model under attack; handles training, evaluation, and checkpointing.
- **Attacker**: defines the inference strategy and trains the attack model.
- **ShadowDataManager**: samples and manages shadow datasets.
- **ShadowModelManager**: trains and orchestrates shadow models.
- **AuditDataManager**: constructs audit datasets combining member and non‑member samples.


## 🚀 Quick Start


### Quick overview
- Training entrypoint: `train_victim.py` — trains only the victim model and saves checkpoints.
- End-to-end pipeline: `run.py` — trains victim, (optionally) shadow models, runs attacker optimization, and saves results.
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
- Use `python download_all_datasets.py` to download supported datasets to the default `datas/` folder.

### Training victims (and checkpointing for reuse)
Use `train_victim.py` when you only want to train the victim model and persist checkpoints for later reuse across multiple attacks. This is the recommended workflow for large experiments: train a victim once with stable settings, then run several attacks reusing that checkpoint.

Example — train a victim and store checkpoints:

```bash
# Train only the victim with standard SGD and no defense
python train_victim.py \
	--dataset cifar10 \
	--victim_model resnet18 \
	--victim_epochs 100 \
	--victim_batch_size 128 \
	--device 0
```

The repository also supports training a victim model with Differential Privacy (DP) by enabling `--victim_use_dp` and the related DP flags which instruct the training pipeline to use the configured DP mechanism (see `src/utils/configs.py` where `DPConfigs` are assembled). Example:

```bash
# Train only the victim with DP-SGD enabled
python train_victim.py \
	--dataset cifar10 \
	--victim_model resnet18 \
	--victim_epochs 50 \
	--victim_batch_size 128 \
	--victim_use_dp \
	--victim_dp_noise_multiplier 1.2 \
	--victim_dp_max_grad_norm 1.0 \
	--victim_dp_clip_per_layer \
	--device 0
```

*Note*: depending on implementation and dependency (e.g., Opacus), extra packages may be required for DP training; check your environment if you plan to run DP experiments.

#### Notes on checkpointing and `--resume`:
- When you run any command the repository computes deterministic hashes from the *relevant* CLI settings and saves settings and checkpoints under `outs/` using these hashes (see `src/utils/configs.py`).
- Victim checkpoints are stored under `outs/victims/<victim_hash>/ckpts` and attacker checkpoints under `outs/attackers/<attacker_hash>/ckpts`.
- The hash is computed by `get_hash_from_settings(...)`, which first filters settings with `get_relevant_settings(...)` and then hashes the JSON representation (sorted keys) — so the hash changes if any relevant setting changes.
- `--resume` will only reuse checkpoints whose settings produce the exact same hash (i.e. identical relevant settings). If settings differ, training starts from scratch and new folders are created.

Refer to [src/utils/configs.py](src/utils/configs.py) and [src/utils/settings.py](src/utils/settings.py) for the exact rules used to compute which flags are considered "relevant" for victim/attacker/experiment hashes.

### Two concrete Quick-Start examples

These examples are minimal working commands that use the repository's CLI flags (see `src/utils/settings.py` for the full list).

Example A — Quantile attack on CIFAR-10 (fast, minimal shadows)

```bash
# download data (one-time)
python download_all_datasets.py

# run a short end-to-end experiment: victim + attack (quantile)
python run.py \
	--attack_mode quantile \
	--dataset cifar10 \
	--victim_model resnet18 \
	--att_model resnet18 \
	--victim_epochs 10 \
	--att_epochs 5 \
	--n_shadows 1 \
	--n_samples_per_shadow_dataset 2000 \
	--n_auditing_samples 1000 \
	--device 0
```

Example B — LiRA attack on TinyImageNet (larger experiment)

```bash
# run a larger experiment for LiRA on TinyImageNet
python run.py \
	--attack_mode lira \
	--dataset tinyimagenet \
	--victim_model resnet50 \
	--att_model resnet50 \
	--victim_epochs 90 \
	--att_epochs 30 \
	--n_shadows 50 \
	--n_samples_per_shadow_dataset 10000 \
	--n_auditing_samples 5000 \
	--victim_batch_size 256 \
	--att_batch_size 256 \
	--device 0
```

Notes:
- LiRA and other heavy attacks typically require many shadow models and larger shadow datasets — expect increased runtime and GPU memory usage.


### Options and defaults
The full CLI options and defaults can be found in [src/utils/settings.py](src/utils/settings.py), while the default folders and paths are in [src/utils/variables.py](src/utils/variables.py).

#### Important CLI flags (high level)
Below are the most commonly used flags. See `src/utils/settings.py` for the full authoritative list and exact defaults.

- `--dataset`: dataset name (cifar10, cifar100, svhn, fmnist, cinic10, imagenet, tinyimagenet)
- `--victim_model`: victim architecture (resnet18, resnet50, vgg16, mobile_small, etc.)
- `--victim_epochs`, `--victim_batch_size`: victim training schedule and batch size
- `--victim_lr`, `--victim_lr_sched`: victim learning rate and scheduler
- `--victim_use_dp`, `--victim_dp_noise_multiplier`, `--victim_dp_max_grad_norm`, `--victim_dp_clip_per_layer`: differential privacy training options for the victim
- `--att_model`, `--att_epochs`, `--att_batch_size`, `--att_lr`: attacker model and training settings
- `--attack_mode`: attack strategy (`quantile`, `lira`, `neural_feat`, `rmia_loss`, `pmia_confidence`, ...)
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
