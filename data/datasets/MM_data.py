#Add data set information and configuration here.

import numpy as np
import torch
import jax.numpy as jnp
import pathlib
import random
from rcgan.fastmri.data.transforms import to_tensor

class MassMappingDataset_Test(torch.utils.data.Dataset):
    """Loads the data."""
    def __init__(self, data_dir, transform=to_tensor, sample_rate=1, big_test=False, test_set=False):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.  
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
        self.test_set = test_set
        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(data_dir).iterdir())

        random.seed()
        np.random.seed()

        random.shuffle(files)

        num_files = len(files)
        f_testing_and_val = sorted(files[-num_files:]) if big_test else sorted(files[0:num_files])

        files = f_testing_and_val

        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files)*sample_rate)
            files = files[:num_files]
        
        for fname in sorted(files):
            self.examples += fname
        
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self,i):
        """Loads and returns a sample from the dataset at a given index."""
        image = jnp.load(self.examples[i])

        if self.transform:
            image = self.transform(image)
            #image = torch.movedim(image, [2,H,W])
        return image

class MassMappingDataset_Val(torch.utils.data.Dataset):

    def __init__(self, data_dir, transform=to_tensor, sample_rate = 1, big_test=False, test_set=False):

        self.test_set = test_set
        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(data_dir).iterdir())

        random.seed()
        np.random.seed()

        random.shuffle(files)

        num_files = (round(len(files)*0.7) if big_test else round(len(files)*0.3))
        f_testing_and_val = sorted(files[-num_files:]) if big_test else sorted(files[0:num_files])

        files = f_testing_and_val

        if sample_rate < 1:
            random.shuffle(files)
            num_files = round(len(files)*sample_rate)
            files = files[:num_files]

        for fname in sorted(files):
            self.examples += fname
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        image = jnp.load(self.examples[i])
        if self.transform:
            image = self.transform(image)
        return image        



class MassMappingDataset_Train(torch.utils.data.Dataset):

    def __init__(self, data_dir, transform=to_tensor, sample_rate=1):

        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(data_dir).iterdir())

        random.seed()
        np.random.seed()

        random.shuffle(files)

        num_files = round(len(files))

        f_training = sorted(files[0:num_files])

        files = f_training

        if sample_rate <1:
            random.shuffle(files)
            num_files = round(len(files)*sample_rate)
            files = files[:num_files]

        for fname in sorted(files):
            self.examples += fname
                
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self,i):
        image = jnp.load(self.examples[i])
        if self.transform:
            image = self.transform(image)
        return image        