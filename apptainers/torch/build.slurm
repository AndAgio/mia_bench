#!/bin/sh
#SBATCH --job-name="build-torch-apptainer"
#SBATCH --account="ewi-insy-cys"
#SBATCH --partition=insy
#SBATCH --qos=medium
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=12GB
#SBATCH --mail-type=END
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# Start measuring execution time
start_time=$(date +%s)

srun apptainer build torch_container.sif Apptainer.def 2>&1

# End measuring execution time
end_time=$(date +%s)

elapsed_time=$((end_time - start_time))
echo "Elapsed time: $elapsed_time seconds"

