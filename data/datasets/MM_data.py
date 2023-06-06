#Add data set information and configuration here.

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import os
from torchvision.io import read_image


#Add function to find the required directory?

class MassMappingDataset(Dataset):
    """Loads the data."""
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.        
        """
        #Directory of data
        self.data_dir = data_dir

        #need to make self.datafile HERE
        #Don't want to load here

        #Goes from individual files into full dataset - and converts to PyTorch tensor(?)
        self.dataset=torch.tensor(self.datafile).float()

        self.transform = transform
        
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self,i):
        """Loads and returns a sample from the dataset at a given index 'idx'"""
        image = self.dataset[i]
        #Want to load a specific instance here instead - bc we can't load them all 

        if self.transform:
            image = self.transform(image)
        return image



#define how to separate train val test
#Will need 1 of these files for each#Will need to define 3 dataloaders so need 3 datasets
#npy.load instead of h5py.File
#list of all the paths - then shuffle - don't specify seed
#Make  branch on Tobias' fork.