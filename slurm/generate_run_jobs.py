import os
import pathlib
import shutil
import glob
import subprocess
PATH_REPO = pathlib.Path(__file__).parent.parent
print(f"PATH_REPO: {PATH_REPO}")
import sys
sys.path.append(str(PATH_REPO))
from src.utils.yaml import load_secrets_yaml


def define_slurm_file_preamble(job_name, secrets):
    text = "#!/bin/sh\n"
    text += f"\n#SBATCH --job-name {job_name}"
    text += f"\n#SBATCH --account={secrets['cluster']['account']}"
    text += f"\n#SBATCH --partition={secrets['cluster']['partition']}"
    text += f"\n#SBATCH --qos={secrets['cluster']['qos']}"
    text += f"\n#SBATCH --time {secrets['cluster']['timeout']}"
    text += f"\n#SBATCH --nodes={secrets['cluster']['nodes']}"
    text += f"\n#SBATCH --tasks-per-node={secrets['cluster']['tasks_per_node']}"
    text += f"\n#SBATCH --cpus-per-task={secrets['cluster']['cpus_per_task']}"
    text += f"\n#SBATCH --mem={secrets['cluster']['mem']}"
    text += f"\n#SBATCH --output={job_name}.out"
    text += f"\n#SBATCH --error={job_name}.out"
    if secrets['cluster']['university'] == 'delft':
        text += f"\n#SBATCH --mail-type=END"
        gpu_type = ":" + secrets['cluster']['gpu_type'] if secrets['cluster']['gpu_type'] != 'none' else ''
        text += f"\n#SBATCH --gres=gpu{gpu_type}:{secrets['cluster']['gpu_num']}"
    elif secrets['cluster']['university'] == 'purdue':
        text += f"\n#SBATCH --gpus-per-node={secrets['cluster']['gpu_num']}"
    text += "\n\ncd .."
    if secrets['cluster']['university'] == 'delft':
        text += f"\nexport APPTAINER_IMAGE={secrets['cluster']['container_path']}"
        text += f"\n\nmodule use /opt/insy/modulefiles"
        text += f"\nmodule load cuda/12.4"
    return text



MODE = 'train_defender'  # 'train_defender' or 'run_attack'
secrets = load_secrets_yaml()


# Rotating variable
DATASETS = ['cifar10'] # ['cifar10', 'cifar100', 'svhn', 'fmnist', 'cinic10', 'tinyimagenet']
VICTIM_MODELS = ['resnet18'] # ['resnet18', 'resnet50', 'vgg16', 'mobile_small', 'mobile_large', 'wideresnet_16_8', 'wideresnet_28_10', 'wideresnet_50_2', "inception_v3"]
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
N_SHADOWS = 50
SAMPLES_SHADOW = 10000
SAMPLES_AUDIT = 5000

# Defining folder where to store job files
jobs_dir = 'exes'
jobs_dir = os.path.join(PATH_REPO, jobs_dir)
if os.path.exists(jobs_dir) and os.path.isdir(jobs_dir):
    shutil.rmtree(jobs_dir)
os.makedirs(jobs_dir, exist_ok=True)

if MODE == 'train_defender':
    for dataset in DATASETS:
        for defender_model in VICTIM_MODELS:
            # Defining job name
            job_name = 'train_defender_{}_with_{}'.format(dataset, defender_model)
            print(f"Generating sbatch file for job with name: {job_name}")
            # Define device usages
            text = define_slurm_file_preamble(job_name, secrets)

            cfg = SGD_HYPERPARAMS[dataset]
            cfg = cfg.get(defender_model, cfg["default"])

            # Define python script to launch
            text += f"\n\n{'srun apptainer exec --nv -B $HOME:$HOME -B /tudelft.net/:/tudelft.net/ $APPTAINER_IMAGE ' if secrets['cluster']['university'] == 'delft' else ''}"\
                    f"python train_defender.py --dataset={dataset} --defender_model={defender_model} "\
                    f"--defender_epochs={cfg['training']['epochs']} --defender_batch_size={cfg['training']['batch_size']} "\
                    f"--defender_optimizer={cfg['optimizer']['name']} --defender_lr={cfg['optimizer']['lr']} --defender_weight_decay={cfg['optimizer']['weight_decay']} --defender_momentum={cfg['optimizer']['momentum']} {'--defender_nesterov' if cfg['optimizer']['nesterov'] else ''} "\
                    f"--defender_lr_sched={cfg['scheduler']['name']} "
            for key, value in cfg['scheduler']['extra'].items():
                if not isinstance(value, list):
                    text += f"--defender_lr_{key}={value} " 
                else:
                    for ind, val in enumerate(value):
                        if ind == 0:
                            text += f"--defender_lr_{key} {val} "
                        else:
                            text += f"{val} "
            text += f" --device=0 "

            # Write file
            with open(os.path.join(jobs_dir, '{}.sbatch'.format(job_name)), 'w') as f:
                f.write(text)
            
elif MODE == 'run_attack':
    for dataset in DATASETS:
        for attack in ATTACKS:
            for defender_model in VICTIM_MODELS:
                attacker_model = defender_model
                # for attacker_model in ATTACKER_MODEL:
                # Defining job name
                job_name = '{}_on_{}_with_vic_{}_and_att_{}'.format(attack, dataset, attacker_model, defender_model)
                print(f"Generating sbatch file for job with name: {job_name}")
                # Define device usages
                text = define_slurm_file_preamble(job_name, secrets)

                cfg = SGD_HYPERPARAMS[dataset]
                cfg = cfg.get(defender_model, cfg["default"])

                # Define python script to launch
                text += f"\n\n{'srun apptainer exec -B $HOME:$HOME -B /tudelft.net/:/tudelft.net/ $APPTAINER_IMAGE ' if secrets['cluster']['university'] == 'delft' else ''}"\
                        f"python run.py --dataset={dataset} --defender_model={defender_model} "\
                        f"--defender_epochs={cfg['training']['epochs']} --defender_batch_size={cfg['training']['batch_size']} "\
                        f"--defender_optimizer={cfg['optimizer']['name']} --defender_lr={cfg['optimizer']['lr']} --defender_weight_decay={cfg['optimizer']['weight_decay']} --defender_momentum={cfg['optimizer']['momentum']} {'--defender_nesterov' if cfg['optimizer']['nesterov'] else ''} "\
                        f"--defender_lr_sched={cfg['scheduler']['name']} "
                for key, value in cfg['scheduler']['extra'].items():
                    if not isinstance(value, list):
                        text += f"--defender_lr_{key}={value} " 
                    else:
                        for ind, val in enumerate(value):
                            if ind == 0:
                                text += f"--defender_lr_{key} {val} "
                            else:
                                text += f"{val} "
                text += f"--att_model={defender_model} "\
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
                with open(os.path.join(jobs_dir, '{}.slurm'.format(job_name)), 'w') as f:
                    f.write(text)
else:
    raise ValueError(f"Unknown MODE '{MODE}' specified!")

# # SUBMIT JOBS
# files = glob.glob(os.path.join(jobs_dir, '*.slurm'))
# skeemed_files = files
# # skeemed_files = []
# # for file in files:
# #     with open(file) as f:
# #         content = f.readlines()
# #         if CLUSTER_MACHINE in content[2]:
# #             skeemed_files.append(file)

# # print('Skeemed files: {}'.format(skeemed_files))

# commands = ['cd {}\nsbatch {}'.format(jobs_dir, filename.split('/')[-1]) for filename in skeemed_files]
# procs = [subprocess.Popen(commands[j], shell=True) for j in range(len(commands))]
# for p in procs:
#     p.wait()


