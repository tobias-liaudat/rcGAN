from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional
from data.datasets.MM_data import MassMappingDataset_Test, MassMappingDataset_Train, MassMappingDataset_Val

import pathlib
import cv2
import torch
import numpy as np

from utils.mri.espirit import ifft, fft
from utils.mri import transforms
from utils.mri.fftc import ifft2c_new, fft2c_new
from utils.mri.get_mask import get_mask


class DataTransform:
    pass

class MMDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project.
    """
    def __init__(self, args, big_test=False):
        super().__init__()
        self.prepare_data_per_node = True
        self.args = args
        self.big_test = big_test

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        #Assign train/val datasets for use in dataloaders

        train_data = MassMappingDataset_Train(
            data_dir=pathlib.Path(self.args.data_path),
            transform=DataTransform(self.args),
            sample_rate=1,
            restrict_size=False
        )

        dev_data = MassMappingDataset_Val(
            data_dir=pathlib.Path(self.args.data_path),
            transform=DataTransform(self.args, test=True),
            sample_rate=1,
            restrict_size=False,
            big_test=self.big_test
        )    

        test_data = MassMappingDataset_Test(
            data_dir=pathlib.Path(self.args.data_path),
            transform=DataTransform(self.args, test=True),
            sample_rate=1,
            restrict_size=False,
            big_test=True
        )

        self.train, self.validate, self.test = train_data, dev_data, test_data


    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet.
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.args.batch_size,
            num_workers=4,
            drop_last=True,
            pin_memory=False
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validate,
            batch_size=self.args.batch_size,
            num_workers=4,
            drop_last=True,
            pin_memory=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=4,
            num_workers=4,
            pin_memory=False,
            drop_last=False
        )
