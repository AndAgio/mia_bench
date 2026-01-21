import os
import shutil
import glob
import subprocess
from src.utils.yaml import load_secrets_yaml

MODE = 'train_victim'  # 'train_victim' or 'run_attack'

TIMEOUT = load_secrets_yaml()['cluster']['server_timeout']
CLUSTER_MACHINE = load_secrets_yaml()['cluster']['server_qos']
REQUESTED_GPU = "v100"
ACCOUNT = load_secrets_yaml()['cluster']['server_account']
NODES = 1
GPUS_PER_NODE = 1
CPUS_PER_TASK = 1
MEM_PER_CPU = 50000


# Rotating variable
DATASETS = ['cifar10', 'cifar100', 'svhn', 'fmnist', 'tinyimagenet']
VICTIM_MODELS = ['resnet18', 'resnet50', 'vgg16', 'mobile_small', 'mobile_large', 'wideresnet_16_8', 'wideresnet_28_10', 'wideresnet_50_2', "inception_v3"]
ATTACKS = ['on_robust', 'off_robust', "lira", "quantile", "neural_feat", "neural_prob", "neural_logit", 'rmia_loss', 'rmia_confidence', 'rmia_entropy', 'pmia_loss', 'pmia_confidence', 'pmia_entropy']
# ATTACKER_MODEL = ['resnet18']


SGD_HYPERPARAMS = {

    "cifar10": {

        "default": {
            "optimizer": dict(
                name="sgd",
                lr=0.1,
                weight_decay=5e-4,
                momentum=0.9,
                nesterov=True,
            ),
            "training": dict(
                epochs=200,
                batch_size=128,
            ),
            "scheduler": dict(
                name="multistep",
                epochs=200,
                extra=dict(
                    step_milestones=[100, 150],
                    step_gamma=0.1,
                ),
            ),
        },

        "wideresnet_28_10": {
            "optimizer": dict(
                name="sgd",
                lr=0.1,
                weight_decay=5e-4,
                momentum=0.9,
                nesterov=True,
            ),
            "training": dict(
                epochs=200,
                batch_size=128,
            ),
            "scheduler": dict(
                name="warmup_step",
                epochs=200,
                extra=dict(
                    warmup_epochs=10,
                    warmup_multiplier=1.0,
                    step_size=60,
                    step_gamma=0.2,
                ),
            ),
        },
    },

    "cifar100": {

        "default": {
            "optimizer": dict(
                name="sgd",
                lr=0.1,
                weight_decay=5e-4,
                momentum=0.9,
                nesterov=True,
            ),
            "training": dict(
                epochs=200,
                batch_size=128,
            ),
            "scheduler": dict(
                name="multistep",
                epochs=200,
                extra=dict(
                    step_milestones=[100, 150],
                    step_gamma=0.1,
                ),
            ),
        },

        "resnet50": {
            "optimizer": dict(
                name="sgd",
                lr=0.1,
                weight_decay=1e-4,
                momentum=0.9,
                nesterov=True,
            ),
            "training": dict(
                epochs=200,
                batch_size=128,
            ),
            "scheduler": dict(
                name="cosine",
                epochs=200,
                extra=dict(
                    cosine_min=1e-5,
                ),
            ),
        },
    },

    "svhn": {

        "default": {
            "optimizer": dict(
                name="sgd",
                lr=0.05,
                weight_decay=5e-4,
                momentum=0.9,
                nesterov=True,
            ),
            "training": dict(
                epochs=120,
                batch_size=128,
            ),
            "scheduler": dict(
                name="step",
                epochs=120,
                extra=dict(
                    step_size=60,
                    step_gamma=0.1,
                ),
            ),
        },
    },

    "fmnist": {

        "default": {
            "optimizer": dict(
                name="sgd",
                lr=0.01,
                weight_decay=1e-4,
                momentum=0.9,
                nesterov=False,
            ),
            "training": dict(
                epochs=100,
                batch_size=128,
            ),
            "scheduler": dict(
                name="step",
                epochs=100,
                extra=dict(
                    step_size=40,
                    step_gamma=0.1,
                ),
            ),
        },
    },

    "tinyimagenet": {

        "default": {
            "optimizer": dict(
                name="sgd",
                lr=0.1,
                weight_decay=1e-4,
                momentum=0.9,
                nesterov=True,
            ),
            "training": dict(
                epochs=180,
                batch_size=256,
            ),
            "scheduler": dict(
                name="cosine",
                epochs=180,
                extra=dict(
                    cosine_min=1e-5,
                ),
            ),
        },

        "inception_v3": {
            "optimizer": dict(
                name="sgd",
                lr=0.045,
                weight_decay=4e-5,
                momentum=0.9,
                nesterov=True,
            ),
            "training": dict(
                epochs=180,
                batch_size=256,
            ),
            "scheduler": dict(
                name="warmup_cosine",
                epochs=180,
                extra=dict(
                    cycle_step=30,
                    cycle_gamma=1.0,
                    cosine_min=1e-5,
                ),
            ),
        },
    },
}




# Fixed variables
VICTIM_EPOCHS = 100
ATTACKER_EPOCHS = 50
N_SHADOWS = 50
SAMPLES_SHADOW = 10000
SAMPLES_AUDIT = 5000

# Defining folder where to store job files
jobs_dir = 'exes'
if os.path.exists(jobs_dir) and os.path.isdir(jobs_dir):
    shutil.rmtree(jobs_dir)
os.makedirs(jobs_dir, exist_ok=True)

if MODE == 'train_victim':
    for dataset in DATASETS:
        for victim_model in VICTIM_MODELS:
            # Defining job name
            job_name = 'train_victim_{}_with_{}'.format(dataset, victim_model)
            print(f"Generating sbatch file for job with name: {job_name}")
            # Define device usages
            text = "#!/bin/sh\n"
            text += f"\n#SBATCH --account={ACCOUNT} --qos={CLUSTER_MACHINE} --partition={REQUESTED_GPU}"
            text += f"\n#SBATCH --time {TIMEOUT}"
            text += f"\n#SBATCH --nodes={NODES} --gpus-per-node={GPUS_PER_NODE} --cpus-per-task={CPUS_PER_TASK}"
            text += f"\n#SBATCH --job-name {job_name}"
            text += f"\n#SBATCH --output={job_name}.out"
            text += f"\n#SBATCH --error={job_name}.out"
            text += f"\n#SBATCH --mem-per-cpu={MEM_PER_CPU}"
            
            text += "\ncd .."

            cfg = SGD_HYPERPARAMS[dataset]
            cfg = cfg.get(victim_model, cfg["default"])

            # Define python script to launch
            text += f"\n\npython train_victim.py --dataset={dataset} --victim_model={victim_model} "\
                    f"--victim_epochs={cfg['training']['epochs']} --victim_batch_size={cfg['training']['batch_size']} "\
                    f"--victim_optimizer={cfg['optimizer']['name']} --victim_lr={cfg['optimizer']['lr']} --victim_weight_decay={cfg['optimizer']['weight_decay']} --victim_momentum={cfg['optimizer']['momentum']} {'--victim_nesterov' if cfg['optimizer']['nesterov'] else ''} "\
                    f"--victim_lr_sched={cfg['scheduler']['name']} "
            for key, value in cfg['scheduler']['extra'].items():
                if not isinstance(value, list):
                    text += f"--victim_lr_{key}={value} " 
                else:
                    for ind, val in enumerate(value):
                        if ind == 0:
                            text += f"--victim_lr_{key} {val} "
                        else:
                            text += f"{val} "
            text += f" --device=0 "

            # Write file
            with open(os.path.join(jobs_dir, '{}.sbatch'.format(job_name)), 'w') as f:
                f.write(text)
            
elif MODE == 'run_attack':
    for dataset in DATASETS:
        for attack in ATTACKS:
            for victim_model in VICTIM_MODELS:
                attacker_model = victim_model
                # for attacker_model in ATTACKER_MODEL:
                # Defining job name
                job_name = '{}_on_{}_with_vic_{}_and_att_{}'.format(attack, dataset, attacker_model, victim_model)
                print(f"Generating sbatch file for job with name: {job_name}")
                # Define device usages
                text = "#!/bin/sh\n"
                text += f"\n#SBATCH --account={ACCOUNT} --qos={CLUSTER_MACHINE} --partition={REQUESTED_GPU}"
                text += f"\n#SBATCH --time {TIMEOUT}"
                text += f"\n#SBATCH --nodes={NODES} --gpus-per-node={GPUS_PER_NODE} --cpus-per-task={CPUS_PER_TASK}"
                text += f"\n#SBATCH --job-name {job_name}"
                text += f"\n#SBATCH --output={job_name}.out"
                text += f"\n#SBATCH --error={job_name}.out"
                text += f"\n#SBATCH --mem-per-cpu={MEM_PER_CPU}"
                
                text += "\ncd .."

                cfg = SGD_HYPERPARAMS[dataset]
                cfg = cfg.get(victim_model, cfg["default"])

                # Define python script to launch
                text += f"\n\npython run.py --dataset={dataset} --victim_model={victim_model} "\
                        f"--victim_epochs={cfg['training']['epochs']} --victim_batch_size={cfg['training']['batch_size']} "\
                        f"--victim_optimizer={cfg['optimizer']['name']} --victim_lr={cfg['optimizer']['lr']} --victim_weight_decay={cfg['optimizer']['weight_decay']} --victim_momentum={cfg['optimizer']['momentum']} {'--victim_nesterov' if cfg['optimizer']['nesterov'] else ''} "\
                        f"--victim_lr_sched={cfg['scheduler']['name']} "
                for key, value in cfg['scheduler']['extra'].items():
                    if not isinstance(value, list):
                        text += f"--victim_lr_{key}={value} " 
                    else:
                        for ind, val in enumerate(value):
                            if ind == 0:
                                text += f"--victim_lr_{key} {val} "
                            else:
                                text += f"{val} "
                text += f"--att_model={victim_model} "\
                        f"--att_epochs={cfg['training']['epochs']} --att_batch_size={cfg['training']['batch_size']} "\
                        f"--att_optimizer={cfg['optimizer']['name']} --att_lr={cfg['optimizer']['lr']} --att_weight_decay={cfg['optimizer']['weight_decay']} --att_momentum={cfg['optimizer']['momentum']} {'--att_nesterov' if cfg['optimizer']['nesterov'] else ''} "\
                        f"--att_lr_sched={cfg['scheduler']['name']} "
                for key, value in cfg['scheduler']['extra'].items():
                    if not isinstance(value, list):
                        text += f"--att_lr_{key}={value} " 
                    else:
                        for ind, val in enumerate(value):
                            if ind == 0:
                                text += f"--att_lr_{key} {val} "
                            else:
                                text += f"{val} "
                text += f"--attack_mode={attack} "\
                        f"--n_shadows={1 if 'pmia' in attack or 'quantile' in attack else N_SHADOWS} "\
                        f"--n_samples_per_shadow_dataset={5000  if 'pmia' in attack else SAMPLES_SHADOW} "\
                        f"--n_auditing_samples={SAMPLES_AUDIT} "\
                        f"--device=0 "\
                        f"--resume"

                # Write file
                with open(os.path.join(jobs_dir, '{}.sbatch'.format(job_name)), 'w') as f:
                    f.write(text)
else:
    raise ValueError(f"Unknown MODE '{MODE}' specified!")

# # SUBMIT
# files = glob.glob(os.path.join(jobs_dir, '*.sbatch'))
# skeemed_files = []
# for file in files:
#     with open(file) as f:
#         content = f.readlines()
#         if CLUSTER_MACHINE in content[2]:
#             skeemed_files.append(file)

# print('Skeemed files: {}'.format(skeemed_files))

# commands = ['cd {}\nsbatch {}'.format(jobs_dir, filename.split('/')[-1]) for filename in skeemed_files]
# procs = [subprocess.Popen(commands[j], shell=True) for j in range(len(commands))]
# for p in procs:
#     p.wait()


