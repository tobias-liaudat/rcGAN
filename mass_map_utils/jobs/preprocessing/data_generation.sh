#!/bin/bash

#SBATCH --job-name=data_generation 
#SBATCH -p GPU
# requesting one node
# SBATCH -N1
# requesting 12 cpus
# SBATCH -n12
#SBATCH --ntasks=1                   # nombre total de tache MPI (= nombre total de GPU)
#SBATCH --ntasks-per-node=1          # nombre de tache MPI par noeud (= nombre de GPU par noeud)
#SBATCH --cpus-per-task=12           # nombre de coeurs CPU par tache (un quart du noeud ici)
#SBATCH --gres=gpu:v100:4            # requesting GPUs
#SBATCH --mail-use=jessica.whitney.22
#SBATCH --mail-type=ALL
#SBATCH --output=data_generation_%j.out
#SBATCH --error=data_generation_%j.err



# Load conda and activate environment
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN

# Echo commands
set -x

echo $CUDA_VISIBLE_DEVICES

cd /home/jjwhit/rcGAN

srun python -u mass_map_utils/scripts/convergence_map_generation.py

