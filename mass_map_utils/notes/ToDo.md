
# Mass mapping rcGAN


## Setup
- [ ] Set up the working environment on Hypatia
    - [ ] Install the conda environement and the cGAN model. See `comments.md`
    - [ ] Make sure you can launch a notebook running on one of Hyptaia GPUs

## Preprocessing
- [ ] Separate the dataset (in `/share/gpu0/tl3/mass_map_dataset`) into the three subset directories `kappa_test`, `kappa_train`, `kappa_val`.

## Model + Traning
- [ ] Make sure that the generator can forward pass the data `rcGAN.forward()`
    - [ ] Watch out with the `rcGAN.readd_measures()` need modifications
- [ ] Make sure that the discriminator can forward pass the data, `real_pred = self.discriminator(input=x, y=y)`

- [ ] Make sure the different losses can run and produce an output.
- [ ] Start modifying the training of the networks with the different losses used in `rcGAN`

- [ ] Generate new `mmcGAN` to add all the mass mapping modifications to the MRI GAN 


- [ ] Redefine and clean the configuration `mass_map.yml` file
- [ ] Run the training!
