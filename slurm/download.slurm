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

# If you use GPUs
module use /opt/insy/modulefiles
module load cuda/12.1

# Run script
srun apptainer exec \
    --nv \                
    -B $HOME:$HOME \
    -B /tudelft.net/:/tudelft.net/ \
    $APPTAINER_IMAGE \
    python download_all_datasets.py

# --nv binds NVIDIA libraries from the host (only if you use CUDA)
# --env-file source additional environment variables from e.g. .env file (optional)
# -B /$HOME:/$HOME/ mounts host file-sytem inside container
# The home folder should be mounted by default, but sometimes it is not
# -B can be used several times, change this to match your cluster file-system
# APPTAINER_IMAGE is the full path to the container.sif file
# python script.py is the command that you want to use inside the container