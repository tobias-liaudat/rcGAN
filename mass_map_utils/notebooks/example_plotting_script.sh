#!/bin/bash
#SBATCH --partition GPU
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu 8G
##SBATCH --gres=gpu:v100:1
#SBATCH --gres=gpu:a100:1
#SBATCH --time 5:00:00
#SBATCH --job-name plot
#SBATCH --mail-use=ucasjjw@ucl.ac.uk
#SBATCH --mail-type=ALL


# Load conda and activate environment
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
conda activate cGAN

cd /home/jjwhit/rcGAN

srun python -u example_plotting.py