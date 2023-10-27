#!/bin/bash

#SBATCH --job-name=tr_cGAN 
#SBATCH -p GPU
# requesting one node
# SBATCH -N1
# requesting 12 cpus
# SBATCH -n12
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --cpus-per-task=16           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH --gres=gpu:v100:4            # requesting GPUs
#SBATCH --mem-per-gpu=80
#SBATCH --mail-use=t.liaudat@ucl.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=tr_cGAN_%j.out
#SBATCH --error=tr_cGAN_%j.err



# Load conda and activate environment
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN

export WANDB_DIR=/share/gpu0/tl3/wandb/logs
export WANDB_CACHE_DIR=/share/gpu0/tl3/wandb/.cache/wandb
export WANDB_CONFIG_DIR=/share/gpu0/tl3/wandb/.config/wandb

# Echo commands
set -x

echo $CUDA_VISIBLE_DEVICES
echo $WANDB_DIR
echo $WANDB_CACHE_DIR
echo $WANDB_CONFIG_DIR

cd /home/tl3/repos/project-rcGAN/rcGAN

srun python -u train.py --mri --exp-name rcgan_test --num-gpus 4

