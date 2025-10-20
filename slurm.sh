#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for a multi-step job on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --account=def-hsajjad
#SBATCH --gpus-per-node=h100:3
#SBATCH --tasks-per-node=16  # Cores proportional to GPUs: 6 on Cedar, 16 on Graham.
#SBATCH --mem=256000M        # Memory proportional to GPUs: 31500 Cedar, 63500 Graham.
#SBATCH --time=12:00:00
#SBATCH --job-name=Llama4_PAWS_Quora
#SBATCH --output=./llama_paws.txt
#SBATCH --mail-user=hm888458@dal.ca   # who to email
#SBATCH --mail-type=FAIL              # when to email
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `date`"
# ---------------------------------------------------------------------
echo ""
echo "Job Array ID / Job ID: $SLURM_ARRAY_JOB_ID / $SLURM_JOB_ID"
echo "This is job $SLURM_ARRAY_TASK_ID out of $SLURM_ARRAY_TASK_COUNT jobs."
echo ""
# ---------------------------------------------------------------------
# Run your simulation step here...
source /home/hrk21/projects/def-hsajjad/hrk21/venv/representations/bin/activate
module load StdEnv/2023 python-build-bundle/2024a
module load gcc arrow


python Llama_testing_generation.py