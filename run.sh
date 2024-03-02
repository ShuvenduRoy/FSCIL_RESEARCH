#!/bin/sh
#SBATCH --partition=compute                      # compute_full_node / compute
#SBATCH --nodes=1                               # number of nodes requested
#SBATCH --ntasks=1                              # this should be same as number of nodes
#SBATCH --gpus-per-node=1                       # 1/4
# SBATCH --mail-user=bikash11roy@gmail.com
# SBATCH --mail-type=END,FAIL
#SBATCH --error=/scratch/a/amiilab/shuvendu/OUTPUTS/logs/%A.out
#SBATCH --output=/scratch/a/amiilab/shuvendu/OUTPUTS/logs/%A.out
#SBATCH --open-mode=append                      # Append is important because otherwise preemption resets the file
# SBATCH --array=0-2%1                           # auto submit 2 times
#SBATCH --job-name=main
#SBATCH --time=23:30:00

echo "FSCIT"

module load MistEnv/2020a cuda gcc anaconda3 cmake cudnn swig sox/14.4.2
# trunk-ignore(shellcheck/SC1091)
# trunk-ignore(shellcheck/SC3046)
source activate detect_env

COMMAND="python -W ignore train.py"
echo "${COMMAND}"
${COMMAND}

#cat srun_worker.sh
#srun bash srun_worker.sh

# --dataset cifar100 --self_batch_size 1024  --learning_rate 0.2 --cosine --syncBN
# --dataset cifar10 --self_batch_size 512 --learning_rate 0.1 --cosine --syncBN
# sbatch -p debug_full_node --mail-type NONE --time '0:30:00' --array 0 run.sh
# sbatch -p debug --gpus-per-node 1 --mail-type NONE --time '2:00:00' --array 0 run.sh
# sbatch -p compute_full_node --gpus-per-node 4 run.sh
