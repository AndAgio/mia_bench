#!/bin/sh
#SBATCH --job-name="run-mia"
#SBATCH --account="ewi-insy-cys"
#SBATCH --partition=insy
#SBATCH --qos=medium
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32024
#SBATCH --mail-type=END
#SBATCH --output=run_mia.out
#SBATCH --error=run_mia.err

#SBATCH --gres=gpu:1

# Remaining job commands go below here. For example, to run python code that makes use of GPU resources:
cd ..
export APPTAINER_IMAGE="apptainers/cuda/cuda_container.sif"
# Run script
srun apptainer exec -B $HOME:$HOME -B /tudelft.net/:/tudelft.net/ $APPTAINER_IMAGE python run.py --dataset cifar10 --defender_model mobile_small --att_model mobile_small --defender_epochs 2 --att_epochs 5 --n_samples_per_shadow_dataset 2000 --n_auditing_samples 1000 --n_shadow 10 --device mps --attack_mode pmia_entropy --defense_mode mmd