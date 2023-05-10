
# Comments

## Installation
To install the requirements you can create a conda environment with everything with the following command `conda env create -f environment.yml`. This may take a while ~15min.

You need to have conda installed (miniconda3 recommended). You can change the name of the environment if you desire.

Note: We use numpy version version 1.23.x or lower because Sigpy is incompatible with numpy>=1.24.x.

# Paths
The data has to be stored in the `data.path` path in the mri config yaml file. Once downloaded from the fastMRI dataset, it has to be extracted using `tar -xvf brain_multicoil_train_batch_2.tar.xz` for example. Two directories should be created `multicoil_train` and `multicoil_test`

The MRI data has been downloaded to `msslxai` in the path `"/disk/xray0/tl3/project-cGAN/datasets/subsample_fastMRI"`. Do not try to download it again, it is around 1.5T.

The GPU should be specified before running the script with the bash command `export CUDA_VISIBLE_DEVICES=X` with `X` being `0`, `1`, or `2`, for each GPU.  If you want to use a multi-GPU environment, you can set the devices with a comma-separated list, for example: `export CUDA_VISIBLE_DEVICES=1,2`


# Train the model

1. Set the data and checkpoint paths in `config/mri.yml`.
2. Adjust the batch size to the GPU devices available. For `mssslxai` it should be `<=14`, ideally `12`. The variables concerned are `train.batch_size`, `validate.batch_size`, and `test.batch_size`.

Note: The `sense.device` parameter should match one of the IDs of the available GPUs. Better to leave it in `0`.

Activate conda environment and set the GPU visibilities
``` bash
conda activate cGAN_4
export CUDA_VISIBLE_DEVICES=0,1,2
```

To train the model as it is, one should use the three GPUs
``` bash
python train.py --data-parallel --is-mri
```