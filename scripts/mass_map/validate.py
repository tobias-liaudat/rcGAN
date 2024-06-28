import torch
import os
import yaml
import types
import json
import numpy as np

import sys

sys.path.append("/home/jjwhit/rcGAN/")

from mass_map_utils.scripts.ks_utils import pearsoncoeff, psnr, snr, rmse
from data.lightning.MassMappingDataModule import MMDataModule
from utils.parse_args import create_arg_parser
from models.lightning.mmGAN import mmGAN
from pytorch_lightning import seed_everything
from utils.embeddings import VGG16Embedding
from evaluation_scripts.mass_map_cfid.cfid_metric import CFIDMetric


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
    best_epoch = -1
    inception_embedding = VGG16Embedding()
    best_cfid = 10000000
    best_pearson = -1
    start_epoch = 80  # Will start saving models after 80 epochs
    end_epoch = 100
    mask = np.load(
        self.args.cosmo_dir_path + "cosmos_mask.npy", allow_pickle=True
    ).astype(bool)

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

            reconstruction, label, truth = cfid_metric._get_generated_distribution()

            cfids = cfid_metric.get_cfid_torch_pinv()

            cfid_val = np.mean(cfids)

            if cfid_val < best_cfid:
                best_epoch_cfid = epoch
                best_cfid = cfid_val

            pearson_val = pearsoncoeff(truth, reconstruction, mask)
            if pearson_val > best_pearson:
                best_epoch_pearson = epoch
                best_pearson = pearson_val

            psnr_val = psnr(truth, reconstruction, mask)
            if psnr_val > best_psnr:
                best_epoch_psnr = epoch
                best_psnr = psnr_val

            snr_val = snr(truth, reconstruction, mask)
            if snr_val > best_snr:
                best_epoch_snr = epoch
                best_snr = snr_val

            rmse_val = rmse(truth, reconstruction, mask)
            if rmse_val < best_rmse:
                best_epoch_rmse = epoch
                best_rmse = rmse_val

    print(f"BEST EPOCH FOR CFID: {best_epoch_cfid}")
    print(f"BEST EPOCH FOR PSNR: {best_epoch_psnr}")
    print(f"BEST EPOCH FOR SNR: {best_epoch_snr}")
    print(f"BEST EPOCH FOR RMSE: {best_epoch_rmse}")
    print(f"BEST EPOCH FOR R: {best_epoch_pearson}")

    # for epoch in range(end_epoch):
    #     try:
    #         if epoch != best_epoch:
    #             os.remove(cfg.checkpoint_dir + args.exp_name + f'/checkpoint-epoch={epoch}.ckpt')
    #     except:
    #         pass

    os.rename(
        cfg.checkpoint_dir + args.exp_name + f"/checkpoint-epoch={best_epoch}.ckpt",
        cfg.checkpoint_dir + args.exp_name + f"/checkpoint_best.ckpt",
    )
