# rcGAN development version  


# Installation

If in the Hypatia cluster, first run:
``` bash
source /share/apps/anaconda/3-2022.05/etc/profile.d/conda.sh
```


First install the conda dependencies setting the correct channels:
``` bash
conda create --name cGAN --file conda_requirements.txt --channel pytorch --channel nvidia --channel conda-forge --channel defaults
```

Then activate the conda environment and install the pip requirements:
``` bash
conda activate cGAN
pip install -r pypi_requirements.txt
```

# Logging

## Weight and biases

Parameters and environmental variables
WANDB_CACHE_DIR
WANDB_DATA_DIR

logs -> `./wandb` -> `WANDB_DIR`
artifacts -> `~/.cache/wandb` -> `WANDB_CACHE_DIR`
configs -> `~/.config/wandb` -> `WANDB_CONFIG_DIR`

# Set the variables
``` bash
export WANDB_DIR=/share/gpu0/jjwhit/wandb/logs
export WANDB_CACHE_DIR=/share/gpu0/jjwhit/wandb/.cache/wandb
export WANDB_CONFIG_DIR=/share/gpu0/jjwhit/wandb/.config/wandb
```

# Training the model

Training is as simple as running the following command:
```python
python train.py --config ./configs/mass_map.yml --exp-name rcgan_test --num-gpus X
```
where ```X``` is the number of GPUs you plan to use. Note that this project uses Weights and Biases (wandb) for logging.
See [their documentation](https://docs.wandb.ai/quickstart) for instructions on how to setup environment variables.
Alternatively, you may use a different logger. See PyTorch Lightning's [documentation](https://lightning.ai/docs/pytorch/stable/extensions/logging.html) for options.

If you need to resume training, use the following command:
```python
python train.py --config ./configs/mass_map.yml --exp-name rcgan_test --num-gpus X --resume --resume-epoch Y
```
where ```Y``` is the epoch to resume from.

By default, we save the previous 50 epochs. Ensure that your checkpoint path points to a location with sufficient disk space.
If disk space is a concern, 50 can be reduced to 25.
This is important for the next step, validation.


