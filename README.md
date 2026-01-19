
# 🛡️ Membership Inference Attacks Benchmarking Framework

This repository provides a **modular, scalable, and state‑of‑the‑art framework for evaluating Membership Inference Attacks (MIAs)** against neural network models.  
It supports realistic attacker/victim threat models, flexible datasets and architectures, and modern optimizer backends.

---

## ✨ Key Features

- **Model & dataset flexibility**: easily add new architectures and datasets
- **Explicit attacker/victim modeling** with realistic knowledge assumptions
- **Support for modern optimizers**, including SAM, ASAM, GSAM
- **Highly modular attack design** through attacker, victim, and shadow managers
- **Fast experimentation** with optimized shadow‑model training
- **Distributed training and Apple Silicon support**
- **Easy experiment resuming and organized outputs**
- **Built‑in privacy metrics** such as AUC and TPR@FPR

---

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

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/AndAgio/mia_bench.git
cd mia_bench
pip install -r requirements.txt
```

(Optional) Apple Silicon:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Run an Experiment

```bash
python run.py  --attack_mode='lira' --dataset='fmnist' --victim_epochs=50 --att_epochs=25 --n_shadows=50 --device='mps' --n_samples_per_shadow_dataset=10000 --n_auditing_samples=5000 --victim_lr_sched='step' --victim_model='mobile_small' --att_model='mobile_small' --resume
```

This command trains the victim model, shadow models, the attacker, and evaluates MIA performance.

---

## ⚙️ Configuration Parameters

Experiments are driven by configuration files. Key configurable components include:

### Victim
- Architecture and dataset
- Optimizer and training schedule
- Regularization and data augmentation

### Attacker
- Attack strategy
- Feature extraction method
- Attack model architecture

### Shadow Setting
- Number of shadow models
- Shadow dataset size
- Sampling strategy

---

## 🧠 Core Classes

### Victim
Target model under attack; handles training, evaluation, and checkpointing.

### Attacker
Defines the inference strategy and trains the attack model.

### ShadowDataManager
Samples and manages shadow datasets.

### ShadowModelManager
Trains and orchestrates shadow models.

### AuditDataManager
Constructs audit datasets combining member and non‑member samples.

---

## 🛣️ Roadmap

- [ ] Label‑only MIAs
- [ ] Differential privacy training support
- [ ] Federated learning scenarios
- [ ] Automated hyperparameter tuning
- [ ] Visualization dashboard


---

## 🤝 Contributions

Contributions are welcome via issues or pull requests.
