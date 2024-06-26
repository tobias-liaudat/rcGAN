import torch
import yaml
import types
import json
import numpy as np
import sys
sys.path.append('/home/jjwhit/rcGAN/')

from data.lightning.MassMappingDataModule import MMDataModule
from utils.parse_args import create_arg_parser
from pytorch_lightning import seed_everything
from models.lightning.mmGAN import mmGAN
from utils.mri import transforms
import time

def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":

    # Load model.
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

# Load cosmos shear map.
    cosmos_shear = np.load('/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_shear_cropped.npy')
    cosmos_shear_tensor = torch.tensor(np.stack((cosmos_shear.real, cosmos_shear.imag), axis=0)).cuda()
    # cosmos_shear_tensor = transforms.to_tensor(cosmos_shear)
    # cosmos_shear_tensor = cosmos_shear_tensor.permute(2, 0, 1).cuda()


# Feed through GAN and generate 32 samples.
    mask = np.load('/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_mask.npy', allow_pickle=True)
    normalized_gamma, mean, std = transforms.normalise_complex(cosmos_shear_tensor)
    # normalized_gamma = normalized_gamma.permute(1,2,0)
    normalized_gamma[:, mask==0] = 0
    normalized_gamma = normalized_gamma.unsqueeze(0).cuda() #Required?

    gens_mmGAN = torch.zeros(size=(1,cfg.num_z_test, cfg.im_size, cfg.im_size, 2)).cuda()
    with torch.no_grad():
        for z in range(cfg.num_z_test):
            gens_mmGAN[:,z, :, :, :] = mmGAN_model.reformat(mmGAN_model.forward(normalized_gamma))

    torch.save(gens_mmGAN,'/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_samps_manual_5')