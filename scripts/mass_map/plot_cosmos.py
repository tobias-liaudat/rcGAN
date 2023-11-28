import torch
import yaml
import types
import json

import numpy as np
import matplotlib.patches as patches

import sys
sys.path.append('/home/jjwhit/rcGAN/')

from data.lightning.MassMappingDataModule import MMDataModule
from data.lightning.MassMappingDataModule import MMDataTransform #import compute_fourier_kernel, realistic_noise_maker
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.mmGAN import mmGAN
from utils.mri.math import tensor_to_complex_np
from mass_map_utils.scripts.ks_utils import Gaussian_smoothing, ks93, ks93
from scipy import ndimage
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.measure import find_contours

def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":

    #TODO: Step 1: Load model.
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    config_path = args.config

    with open(config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    dm = MMDataModule(cfg)
    fig_count = 1
    dm.setup()
    test_loader = dm.test_dataloader()

    with torch.no_grad():
        mmGAN_model = mmGAN.load_from_checkpoint(
            checkpoint_path=cfg.checkpoint_dir + args.exp_name + '/checkpoint_best.ckpt')

        mmGAN_model.cuda()

        mmGAN_model.eval()

#TODO: Step 2: Load cosmos shear map
    cosmos_shear = np.load('/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_shear_cropped.npy')


#TODO: Step 3: Feed shear map through gan

#TODO Step 4: Generate a bunch of samples
    gens_mmGAN = torch.zeros(size=(cfg.num_z_test, cfg.im_size, cfg.im_size, 2)).cuda()
    for z in range(cfg.num_z_test):
        gens_mmGAN[z, :, :, :] = mmGAN_model.reformat(cosmos_shear) #TODO: Removed batch size, is that okay? or do I need to add first index back?
    #samples = output.detach().cpy().numpy()