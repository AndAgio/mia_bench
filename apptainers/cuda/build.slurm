#!/bin/sh
#SBATCH --job-name="build-apptainer" # Name of your job
#SBATCH --partition=small            # Request partition.
#SBATCH --time=1:00:00               # Request run time (wall-clock). Default is 1 minute
#SBATCH --nodes=1                    # Number of nodes to allocate
#SBATCH --tasks-per-node=1           # Number of tasks to run per node
#SBATCH --cpus-per-task=12           # Request number of CPUs (threads) per task
#SBATCH --mem=12GB                   # Amount of memory to allocate
#SBATCH --mail-type=END              # Set mail type to 'END' to receive a mail when the job finishes
#SBATCH --output=slurm-%x-%j.out     # Set name of output log. %j is the Slurm jobId
#SBATCH --error=slurm-%x-%j.err      # Set name of error log. %j is the Slurm jobId

# Start measuring execution time
start_time=$(date +%s)

srun apptainer build cuda_container.sif Apptainer.def 2>&1

# End measuring execution time
end_time=$(date +%s)

elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"

