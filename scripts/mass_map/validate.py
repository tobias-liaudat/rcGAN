import torch
import os
import yaml
import types
import json
import numpy as np
from tqdm import tqdm
from scipy import ndimage


import sys

sys.path.append("/home/jjwhit/rcGAN/")

from mass_map_utils.scripts.ks_utils import pearsoncoeff, psnr, snr, rmse
from data.lightning.MassMappingDataModule import MMDataModule
from utils.parse_args import create_arg_parser
from models.lightning.mmGAN import mmGAN
from pytorch_lightning import seed_everything
from utils.embeddings import VGG16Embedding
from evaluation_scripts.mass_map_cfid.cfid_metric import CFIDMetric

from utils.mri.math import tensor_to_complex_np


def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    args = create_arg_parser().parse_args()
    seed_everything(1, workers=True)

    config_path = args.config

    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        cfg = json.loads(json.dumps(cfg), object_hook=load_object)

    dm = MMDataModule(cfg)
    dm.setup()
    val_loader = dm.val_dataloader()
    best_epoch_cfid = -1
    best_epoch_pearson = -1
    best_epoch_psnr = -1
    best_epoch_snr = -1
    best_epoch_rmse = -1
    inception_embedding = VGG16Embedding()
    best_cfid = 10000000
    best_pearson = -1
    best_psnr = -1
    best_snr = -1
    best_rmse = 10000000
    start_epoch = 80  # Will start saving models after 80 epochs
    end_epoch = cfg.num_epochs
    mask = np.load(
        "/home/jjwhit/rcGAN/mass_map_utils/cosmos/cosmos_mask.npy", allow_pickle=True
    ).astype(bool)

    cfid_vals = []
    psnr_vals = []
    snr_vals = []
    rmse_vals = []
    pearson_vals = []

    with torch.no_grad():

        for epoch in range(start_epoch, end_epoch):
            print(f"VALIDATING EPOCH: {epoch}")

            # Loads the model one epoch at a time
            try:
                model = mmGAN.load_from_checkpoint(
                    checkpoint_path=cfg.checkpoint_dir
                    + args.exp_name
                    + f"/checkpoint-epoch={epoch}.ckpt"
                )
            except Exception as e:
                print(e)
                continue

            if model.is_good_model == 0:
                print("NO GOOD: SKIPPING...")
                continue

            model = model.cuda()
            model.eval()

            cfid_metric = CFIDMetric(
                gan=model,
                loader=val_loader,
                image_embedding=inception_embedding,
                condition_embedding=inception_embedding,
                cuda=True,
                args=cfg,
                ref_loader=False,
                num_samps=1,
            )

            recon, label, gt = cfid_metric._get_generated_distribution()

            cfids = cfid_metric.get_cfid_torch_pinv()

            cfid_val = np.mean(cfids)
            cfid_vals.append(cfid_val.item())
            if cfid_val < best_cfid:
                best_epoch_cfid = epoch
                best_cfid = cfid_val

            # Trying something new - doing new metrics not in embedded space
            for i, data in tqdm(enumerate(val_loader),
                            desc='Generating samples',
                            total=len(val_loader)):
                y, x, mean, std = data
                y = y.cuda()
                x = x.cuda()
                mean = mean.cuda()
                std = std.cuda()

                gens = torch.zeros(size=(y.size(0), cfg.num_z_test, cfg.im_size, cfg.im_size, 2)).cuda()

                for z in range(cfg.num_z_test):
                    gens[:,z,:,:,:] = model.reformat(model.forward(y))
                
                reconstruction = torch.mean(gens, dim=1)
                truth = model.reformat(x)
                kappa_mean = cfg.kappa_mean
                kappa_std = cfg.kappa_std

                for j in range(y.size(0)):
                    np_reconstruction = {'mmGAN': None,}
                    np_gt = None
                    np_gt = ndimage.rotate(torch.tensor(tensor_to_complex_np((truth[j] * kappa_std + kappa_mean).cpu())).numpy(), 180)
                    np_reconstruction['mmGAN'] = ndimage.rotate(torch.tensor(tensor_to_complex_np((reconstruction[j] + kappa_std + kappa_mean).cpu())).numpy(),180)

                truth = np_gt
                reconstruction = np_reconstruction['mmGAN']

                pearson_val = pearsoncoeff(truth.real, reconstruction.real)

                pearson_vals.append(pearson_val.item())
                if pearson_val > best_pearson:
                    best_epoch_pearson = epoch
                    best_pearson = pearson_val

                psnr_val = psnr(truth.real, reconstruction.real)
                psnr_vals.append(psnr_val.item())
                if psnr_val > best_psnr:
                    best_epoch_psnr = epoch
                    best_psnr = psnr_val

                snr_val = snr(truth.real, reconstruction.real)
                snr_vals.append(snr_val.item())
                if snr_val > best_snr:
                    best_epoch_snr = epoch
                    best_snr = snr_val

                rmse_val = rmse(truth.real, reconstruction.real)
                rmse_vals.append(rmse_val.item())
                if rmse_val < best_rmse:
                    best_epoch_rmse = epoch
                    best_rmse = rmse_val

    print(f"BEST EPOCH FOR CFID: {best_epoch_cfid}")
    print(f"BEST EPOCH FOR PSNR: {best_epoch_psnr}")
    print(f"BEST EPOCH FOR SNR: {best_epoch_snr}")
    print(f"BEST EPOCH FOR RMSE: {best_epoch_rmse}")
    print(f"BEST EPOCH FOR R: {best_epoch_pearson}")

    print("CFID | ", cfid_vals)
    print("PSNR | ", psnr_vals)
    print("SNR | ", snr_vals)
    print("RMSE | ", rmse_vals)
    print("PEARSON | ", pearson_vals)

    # for epoch in range(end_epoch):
    #     try:
    #         if epoch != best_epoch:
    #             os.remove(cfg.checkpoint_dir + args.exp_name + f'/checkpoint-epoch={epoch}.ckpt')
    #     except:
    #         pass

    # os.rename(
    #     cfg.checkpoint_dir + args.exp_name + f"/checkpoint-epoch={best_epoch}.ckpt",
    #     cfg.checkpoint_dir + args.exp_name + f"/checkpoint_best.ckpt",
    # )
