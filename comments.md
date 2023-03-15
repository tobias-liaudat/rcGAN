
# Comments

## Installation
To install the requirements you can create a conda environment with everything with the following command
`conda env create -f environment.yml`

You need to have conda installed (miniconda3 recommended). You can change the name of the environment if you desire.


# Paths
The data has to be stored in the `data.path` path in the mri config yaml file. Once downloaded from the fastMRI dataset, it has to be extracted using `tar -xvf brain_multicoil_train_batch_2.tar.xz` for example. Two directories should be created `multicoil_train` and `multicoil_test`

The GPU should be specified before running the script with the bash command `export CUDA_VISIBLE_DEVICES=X` with `X` being `0`, `1`, or `2`, for each GPU. 
If you want to use a multi-GPU environment, you can set the devices with a comma-separated list, for example:
`export CUDA_VISIBLE_DEVICES=1,2`


# Train the model

Set the data and checkpoint paths in `config/mri.yml` .

Activate conda environment and set
``` bash
conda activate cGAN_2
export CUDA_VISIBLE_DEVICES=0,1,2
```

To train the model as it is, one should use the three GPUs using a batch size of `12` (could be increased a little)
``` bash
python train.py --data-parallel --is-mri
```