# rcGAN development version  


# Installation

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
export WANDB_DIR=/share/gpu0/tl3/wandb/logs
export WANDB_CACHE_DIR=/share/gpu0/tl3/wandb/.cache/wandb
export WANDB_CONFIG_DIR=/share/gpu0/tl3/wandb/.config/wandb
```



