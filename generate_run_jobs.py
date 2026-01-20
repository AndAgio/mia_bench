import os
import shutil
import glob
import subprocess
from src.utils.yaml import load_secrets_yaml



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

        "default": dict(
            lr=0.1,
            epochs=200,
            weight_decay=5e-4,
            scheduler=dict(
                type="step",
                milestones=[100, 150],
                gamma=0.1
            ),
            nesterov=True
        ),

        "wideresnet_16_8": dict(
            lr=0.1,
            epochs=200,
            weight_decay=5e-4,
            scheduler=dict(
                type="step",
                milestones=[100, 150],
                gamma=0.2
            ),
            nesterov=True
        ),

        "wideresnet_28_10": dict(
            lr=0.1,
            epochs=200,
            weight_decay=5e-4,
            scheduler=dict(
                type="step",
                milestones=[60, 120, 160],
                gamma=0.2
            ),
            nesterov=True
        )
    },

    "cifar100": {

        "default": dict(
            lr=0.1,
            epochs=200,
            weight_decay=5e-4,
            scheduler=dict(
                type="step",
                milestones=[100, 150],
                gamma=0.1
            ),
            nesterov=True
        ),

        "resnet50": dict(
            lr=0.1,
            epochs=200,
            weight_decay=1e-4,
            scheduler=dict(
                type="cosine"
            ),
            nesterov=True
        ),

        "vgg16": dict(
            lr=0.05,
            epochs=200,
            weight_decay=5e-4,
            scheduler=dict(
                type="step",
                milestones=[100, 150],
                gamma=0.1
            ),
            nesterov=False
        )
    },

    "svhn": {

        "default": dict(
            lr=0.05,
            epochs=120,
            weight_decay=5e-4,
            scheduler=dict(
                type="step",
                milestones=[60, 90],
                gamma=0.1
            ),
            nesterov=True
        ),

        "wideresnet_28_10": dict(
            lr=0.05,
            epochs=120,
            weight_decay=5e-4,
            scheduler=dict(
                type="step",
                milestones=[60, 90],
                gamma=0.2
            ),
            nesterov=True
        )
    },

    "fmnist": {

        "default": dict(
            lr=0.01,
            epochs=100,
            weight_decay=1e-4,
            scheduler=dict(
                type="step",
                step_size=40,
                gamma=0.1
            ),
            nesterov=False
        ),

        "resnet18": dict(
            lr=0.02,
            epochs=100,
            weight_decay=1e-4,
            scheduler=dict(
                type="cosine"
            ),
            nesterov=True
        )
    },

    "tinyimagenet": {

        "default": dict(
            lr=0.1,
            epochs=180,
            weight_decay=1e-4,
            scheduler=dict(
                type="cosine"
            ),
            nesterov=True
        ),

        "resnet50": dict(
            lr=0.1,
            epochs=180,
            weight_decay=1e-4,
            scheduler=dict(
                type="cosine"
            ),
            nesterov=True
        ),

        "inception_v3": dict(
            lr=0.045,
            epochs=180,
            weight_decay=4e-5,
            scheduler=dict(
                type="cosine"
            ),
            nesterov=True
        )
    }
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
            text += f"\n\npython run.py --dataset='{dataset}' "\
                    f"--victim_epochs={cfg['epochs']} --victim_lr={cfg['lr']} --victim_weight_decay={cfg['weight_decay']} --victim_lr_sched={cfg['scheduler']['type']} {'--victim_nesterov' if cfg['nesterov'] else ''} "\
                    f"--att_epochs={int(cfg['epochs']/2)} --att_lr={cfg['lr']} --att_weight_decay={cfg['weight_decay']} --att_lr_sched={cfg['scheduler']['type']} {'--att_nesterov' if cfg['nesterov'] else ''} "\
                    f"--attack_mode='{attack}' "\
                    f"--n_shadows={1 if 'pmia' in attack or 'quantile' in attack else N_SHADOWS} "\
                    f"--n_samples_per_shadow_dataset={5000  if 'pmia' in attack else SAMPLES_SHADOW} "\
                    f"--n_auditing_samples={SAMPLES_AUDIT} "\
                    f"--device=0 "\
                    f"--resume"

            # Write file
            with open(os.path.join(jobs_dir, '{}.sbatch'.format(job_name)), 'w') as f:
                f.write(text)

# SUBMIT
files = glob.glob(os.path.join(jobs_dir, '*.sbatch'))
skeemed_files = []
for file in files:
    with open(file) as f:
        content = f.readlines()
        if CLUSTER_MACHINE in content[2]:
            skeemed_files.append(file)

print('Skeemed files: {}'.format(skeemed_files))

commands = ['cd {}\nsbatch {}'.format(jobs_dir, filename.split('/')[-1]) for filename in skeemed_files]
procs = [subprocess.Popen(commands[j], shell=True) for j in range(len(commands))]
for p in procs:
    p.wait()


