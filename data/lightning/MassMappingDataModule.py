from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Optional
from data.datasets.mri_data import SelectiveSliceData, SelectiveSliceData_Val, SelectiveSliceData_Test

import pathlib
import cv2
import torch
import numpy as np

from utils.mri.espirit import ifft, fft
from utils.mri import transforms
from utils.mri.fftc import ifft2c_new, fft2c_new
from utils.mri.get_mask import get_mask


class DataTransform:
    """
    Data Transformer.
    """


class MMDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project.
    """

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
