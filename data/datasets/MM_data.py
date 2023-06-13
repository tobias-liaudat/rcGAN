#Add data set information and configuration here.
import numpy as np
import torch
import pathlib
import random


class MassMappingDataset_Test(torch.utils.data.Dataset):
    """Loads the test data."""
    def __init__(self, data_dir, transform, sample_rate=1, big_test=False, test_set=False):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
                'to_tensor' is a function that transforms a complex numpy array into a complex PyTorch tensor.
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
        data = np.load(self.examples[i], allow_pickle=True)
        # Tranform data and generate observations
        return self.transform(data)
        

class MassMappingDataset_Val(torch.utils.data.Dataset):
    """Loads the validation data."""

    def __init__(self, data_dir, transform, sample_rate = 1, big_test=False, test_set=False):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
                'to_tensor' is a function that transforms a complex numpy array into a complex PyTorch tensor.
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
        """Loads and returns a sample from the dataset at a given index."""
        data = np.load(self.examples[i], allow_pickle=True)
        # Tranform data and generate observations
        return self.transform(data)




class MassMappingDataset_Train(torch.utils.data.Dataset):
    """Loads the training data."""

    def __init__(self, data_dir, transform, sample_rate=1):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.  
                'to_tensor' is a function that transforms a complex numpy array into a complex PyTorch tensor.
            sample_rate (float, optional): A float between 0 and 1. This controls what fraction
                of the volumes should be loaded.
        """
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
        """Loads and returns a sample from the dataset at a given index."""
        data = np.load(self.examples[i], allow_pickle=True)
        # Tranform data and generate observations
        return self.transform(data)
