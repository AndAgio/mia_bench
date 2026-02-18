#!/bin/sh
#SBATCH --partition=general     # Request partition. Default is 'general' 
#SBATCH --qos=medium            # Request Quality of Service. Default is 'short' (maximum run time: 4 hours)
#SBATCH --time=06:00:00         # Request run time (wall-clock). Default is 1 minute
#SBATCH --ntasks=1              # Request number of parallel tasks per job. Default is 1
#SBATCH --cpus-per-task=2       # Request number of CPUs (threads) per task. Default is 1 (note: CPUs are always allocated to jobs per 2).
#SBATCH --mem=32024             # Request memory (MB) per node. Default is 1024MB (1GB). For multiple tasks, specify --mem-per-cpu instead
#SBATCH --mail-type=END         # Set mail type to 'END' to receive a mail when the job finishes. 
#SBATCH --output=download.out   # Set name of output log. %j is the Slurm jobId
#SBATCH --error=download.err    # Set name of error log. %j is the Slurm jobId

# Remaining job commands go below here. For example, to run python code that makes use of GPU resources:
cd ..
export APPTAINER_IMAGE="/apptainers/cuda/cuda_container.sif"
# Run script
srun apptainer exec -B $HOME:$HOME -B /tudelft.net/:/tudelft.net/ $APPTAINER_IMAGE python download_all_datasets.py
