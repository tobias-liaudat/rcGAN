
# Mass mapping rcGAN


## Setup
- [X] Set up the working environment on Hypatia
    - [X] Install the conda environement and the cGAN model. See `comments.md`
    - [X] Make sure you can launch a notebook running on one of Hypatia GPUs
        - [X] Can you launch a notebook using current environment?
        - [X] Try on GPU

## Preprocessing
- [X] Separate the dataset (in `/share/gpu0/tl3/mass_map_dataset`) into the three subset directories `kappa_test`, `kappa_train`, `kappa_val`.
-> Stored in `/share/gpu0/jjwhit/mass_map_dataset/kappa20/`

## Model + Traning
- [X] Make sure that the generator can forward pass the data `rcGAN.forward()`
    - [ ] Watch out with the `rcGAN.readd_measures()` need modifications
- [X] Make sure that the discriminator can forward pass the data, `real_pred = self.discriminator(input=x, y=y)`

- [X] Make sure the different losses can run and produce an output.
- [X] Start modifying the training of the networks with the different losses used in `rcGAN`

- [X] Generate new `mmcGAN` to add all the mass mapping modifications to the MRI GAN 


- [X] Redefine and clean the configuration `mass_map.yml` file
- [X] Run the training!

## Modifying the Output
- [ ] Modify the plotting file to produce relevant figures
- [ ] Refactor code
