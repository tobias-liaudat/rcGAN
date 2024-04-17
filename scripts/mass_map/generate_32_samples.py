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
import time

def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
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

        for i, data in enumerate(test_loader):
            if i >= 5:  # Adjust the number based on your requirement
                break
            else:
                y, x, mean, std = data
                y = y.cuda()
                x = x.cuda()
                mean = mean.cuda()
                std = std.cuda()

                batch = 9

                gens_mmGAN = torch.zeros(size=(y.size(0),1000, cfg.im_size, cfg.im_size, 2)).cuda()
                start_time = time.time()
                # for z in range(0, cfg.num_z_test, batch):
                #     end_idx = min(z + batch, cfg.num_z_test)
                #     gens_mmGAN[:,z:end_idx, :, :, :] = mmGAN_model.reformat(mmGAN_model.forward(y))

                for z in range(0, 1000, batch):
                    gens_mmGAN[z:z+batch, :, :, :] = mmGAN_model.reformat(mmGAN_model.forward(y)) #TODO: Reintroduce mean and std

                total_time = time.time() - start_time
                print(f'time is: {total_time}')

                gt = mmGAN_model.reformat(x)


                np.save(f'/share/gpu0/jjwhit/plots/simulation_samps_{i}.npy', gens_mmGAN, allow_pickle=True)
                np.save(f'/share/gpu0/jjwhit/plots/simulation_gt_{i}.npy', gt, allow_pickle=True)

