
# ToDO

- [x] Make the rcGAN work with radio images without conditioning on the sampling pattern
    - [x] Fix the data loaders

Fourier case
- [ ] Handle the normalisation of the input/output pairs

Image domain case
- [x] Try to input the dirty image and the PSF to the cGAN


- [x] Add the conditioning on the sampling distribution

Data generation
- [ ] add noise
- [ ] use varying psf
- [ ] change to only one output (only real part)

Logging
- [ ] add custom plot function for wandb
## Dataset directories

Simulated data
``` bash
/share/gpu0/mars/TNG_data/preprocessed_360
```

Fourier visibilities
``` bash
/share/gpu0/mars/TNG_data/preprocessed_360/fourier
```

Image domain (dirty image + PSF)
``` bash
/share/gpu0/mars/TNG_data/preprocessed_360/image
```

