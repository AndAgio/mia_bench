#!/bin/sh
#SBATCH --job-name="download-datasets"
#SBATCH --account="ewi-insy-cys"
#SBATCH --partition=insy
#SBATCH --qos=long
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50GB
#SBATCH --mail-type=END
#SBATCH --output=download.out
#SBATCH --error=download.out

# Remaining job commands go below here. For example, to run python code that makes use of GPU resources:
cd ..
export APPTAINER_IMAGE="/apptainers/cuda/cuda_container.sif"
# Run script
srun apptainer exec -B $HOME:$HOME -B /tudelft.net/:/tudelft.net/ $APPTAINER_IMAGE python download_all_datasets.py
