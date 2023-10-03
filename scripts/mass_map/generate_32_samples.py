import torch
import yaml
import types
import json

import numpy as np
import matplotlib.patches as patches

import sys
sys.path.append('/home/jjwhit/rcGAN/')

from data.lightning.MassMappingDataModule import MMDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.mmGAN import mmGAN
from utils.mri.math import tensor_to_complex_np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy import ndimage
import sys

def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    #TODO: Refactor config path
    with open('/home/jjwhit/rcGAN/configs/mass_map_8.yml', 'r') as f:
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

        for i, data in enumerate(test_loader):
            y, x, mean, std = data
            y = y.cuda()
            x = x.cuda()
            print(y.size(0))
            mean = mean.cuda()
            std = std.cuda()

            num_samples = 32

            gens_mmGAN = torch.zeros(size=(y.size(0), num_samples, cfg.im_size, cfg.im_size, 2)).cuda()

            for z in range(num_samples):
                gens_mmGAN[:, z, :, :, :] = mmGAN_model.reformat(mmGAN_model.forward(y))

            avg_mmGAN = torch.mean(gens_mmGAN, dim=1)

            gt = mmGAN_model.reformat(x)
            zfr = mmGAN_model.reformat(y)

            save_dic = {
                'mean': mean.detach().cpu().numpy(),
                'std': std.detach().cpu().numpy(),
                'gens_mmGAN': gens_mmGAN.detach().cpu().numpy(),
                'avg_mmGAN': avg_mmGAN.detach().cpu().numpy(),
                'gt': gt.detach().cpu().numpy(),
                'zfr': zfr.detach().cpu().numpy(),
            }

            save_path = '/home/jjwhit/rcGAN/jobs/validate_test/generated_samples_plots.npy'  
            np.save(save_path, save_dic, allow_pickle=True)

            sys.exit()
