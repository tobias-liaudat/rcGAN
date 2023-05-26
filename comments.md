
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

