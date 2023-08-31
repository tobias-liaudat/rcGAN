import torch
import torch.nn as nn
import yaml
import types
import json

import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from data.lightning.MRIDataModule import MRIDataModule
from utils.parse_args import create_arg_parser
from models.lightning.rcGAN import rcGAN
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from data.lightning.MassMappingDataModule import MMDataModule
from data.lightning.RadioDataModule import RadioDataModule
from models.lightning.mmGAN import mmGAN

def load_object(dct):
    return types.SimpleNamespace(**dct)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    args = create_arg_parser().parse_args()
    seed_everything(0, workers=True)

    print(f"Experiment Name: {args.exp_name}")
    print(f"Number of GPUs: {args.num_gpus}")
    print("Device count: ",torch.cuda.device_count())

    if args.mri:
        with open('configs/mri.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        dm = MRIDataModule(cfg)

        model = rcGAN(cfg, args.exp_name, args.num_gpus)
    elif args.mass_mapping:
        with open('/home/jjwhit/rcGAN/configs/mass_map.yml', 'r') as f:
#        with open('configs/mass_map.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        dm = MMDataModule(cfg)

        model = mmGAN(cfg, args.exp_name, args.num_gpus)

    elif args.mass_mapping_8:
        with open('/home/jjwhit/rcGAN/configs/mass_map_8.yml', 'r') as f:
#        with open('configs/mass_map.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        dm = MMDataModule(cfg)

        model = mmGAN(cfg, args.exp_name, args.num_gpus)
    elif args.mass_mapping_6:
        with open('/home/jjwhit/rcGAN/configs/mass_map_6.yml', 'r') as f:
#        with open('configs/mass_map.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        dm = MMDataModule(cfg)

        model = mmGAN(cfg, args.exp_name, args.num_gpus)
    elif args.mass_mapping_4:
        with open('/home/jjwhit/rcGAN/configs/mass_map_4.yml', 'r') as f:
#        with open('configs/mass_map.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        dm = MMDataModule(cfg)
        
        model = mmGAN(cfg, args.exp_name, args.num_gpus)
        #model =  nn.DataParallel(model, device_ids = [0,1,2,3])
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model.to(device)
    elif args.radio_fourier:
        with open('./configs/radio_fourier.yml', 'r') as f:
#        with open('configs/mass_map.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        dm = RadioDataModule(cfg)
        
        model = mmGAN(cfg, args.exp_name, args.num_gpus)
        #model =  nn.DataParallel(model, device_ids = [0,1,2,3])
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model.to(device)
    elif args.radio_image:
        with open('./configs/radio_image.yml', 'r') as f:
#        with open('configs/mass_map.yml', 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
            cfg = json.loads(json.dumps(cfg), object_hook=load_object)

        dm = RadioDataModule(cfg)
        
        model = mmGAN(cfg, args.exp_name, args.num_gpus)
        #model =  nn.DataParallel(model, device_ids = [0,1,2,3])
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #model.to(device)
    else:
        print("No valid application selected. Please include one of the following args: --mri")
        exit()

    wandb_logger = WandbLogger(
        project="mass_mapping_project",  # TODO: Change to your project name - maybe make this an arg
        name=args.exp_name,
        log_model="all",
        save_dir=cfg.checkpoint_dir + 'wandb'
    )

    checkpoint_callback_epoch = ModelCheckpoint(
        monitor='epoch',
        mode='max',
        dirpath=cfg.checkpoint_dir + args.exp_name + '/',
        filename='checkpoint-{epoch}',
        # every_n_epochs=1,
        save_top_k=20
    )

    trainer = pl.Trainer(accelerator="gpu", devices=args.num_gpus, strategy='ddp',
                         max_epochs=cfg.num_epochs, callbacks=[checkpoint_callback_epoch],
                         num_sanity_val_steps=2, profiler="simple", logger=wandb_logger, benchmark=False,
                         log_every_n_steps=10)
 

    if args.resume:
        trainer.fit(model, dm,
                    ckpt_path=cfg.checkpoint_dir + args.exp_name + f'/checkpoint-epoch={args.resume_epoch}.ckpt')
    else:
        trainer.fit(model, dm)
