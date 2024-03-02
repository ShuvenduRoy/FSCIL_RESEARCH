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

echo "FSCIL Tuning"
MASTER=$(/bin/hostname -s)
MPORT=$(shuf -i 6000-9999 -n 1)

module load MistEnv/2020a cuda gcc anaconda3 cmake cudnn swig sox/14.4.2
source activate detect_env
SLEEP=$(shuf -i 0-30 -n 1)

COMMAND="python -W ignore train.py --exp_name tuning6_sidetune_lora --dataset cifar100 --base_mode ft_cos --new_mode avg_cos --batch_size_base 16 --batch_size_new 16 --lr_base 0.001 --decay 0.0005 --schedule Cosine --temperature 16 --moco_k 8192 --mlp True --moco_t 0.07 --moco_m 0.999 --alpha 0.2 --beta 0.8 --constrained_cropping True --fantasy no_fantacy --num_workers 4 --prototype_loss mse --num_mlp 2 --moco_dim 512 --prototype_loss_factor 1 --moco_loss_factor 1.0 --loss_start_epoch 100000 --etf_vectors -1 --incft False --freeze_vit False --freeze_layer_after 10 --encoder vit-b16 --pet_cls LoRA --adapt_blocks 5 --tune_encoder_epoch 10 --encoder_lr_factor 1 --project tuning5 --epochs_base 35 --rank 25 --pre_trained_url None"
echo "$COMMAND"
$COMMAND

#cat srun_worker.sh
#srun bash srun_worker.sh

# --dataset cifar100 --self_batch_size 1024  --learning_rate 0.2 --cosine --syncBN
# --dataset cifar10 --self_batch_size 512 --learning_rate 0.1 --cosine --syncBN
# sbatch -p debug_full_node --mail-type NONE --time '0:30:00' --array 0 run.sh
# sbatch -p debug --gpus-per-node 1 --mail-type NONE --time '2:00:00' --array 0 run.sh
# sbatch -p compute_full_node --gpus-per-node 4 run.sh
