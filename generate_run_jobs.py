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
VICTIM_MODELS = ['resnet18']
ATTACKS = ['quantile', 'on_robust', 'off_robust', 'lira']
ATTACKER_MODEL = ['resnet18']

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
            for attacker_model in ATTACKER_MODEL:
                # Defining job name
                job_name = '{}_on_{}_with_vic_{}_and_att_{}'.format(attack, dataset, attacker_model, victim_model)
                print(f'Generating sbatch file for job with name: {job_name}')
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

                # Define python script to launch
                text += f"\n\npython run.py --dataset='{dataset}' --victim_epochs={VICTIM_EPOCHS} --att_epochs={ATTACKER_EPOCHS} "\
                        f"--n_shadows={N_SHADOWS if attack in ['on_robust', 'off_robust', 'lira'] else 1} --device=0 "\
                        f"--n_samples_per_shadow_dataset={SAMPLES_SHADOW} --n_auditing_samples={SAMPLES_AUDIT} --attack_mode='{attack}' "\
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


