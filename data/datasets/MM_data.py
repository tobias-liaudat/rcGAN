# Add data set information and configuration here.
import numpy as np
import torch
import pathlib


class MassMappingDataset_Test(torch.utils.data.Dataset):
    """Loads the test data."""
    def __init__(self, data_dir, transform):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object (class) that pre-processes the raw data into
                appropriate form for it to be fed into the model.
        """
        self.transform = transform
        self.examples = []
        
        # Collects the paths of all files.
        files = list(pathlib.Path(data_dir).iterdir())

        # Shuffle all the files.
        np.random.seed()
        np.random.shuffle(files)

        self.examples = files
        
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self,i):
        """Loads and returns a sample from the dataset at a given index."""

        # Cast input data from float64 to complex128 as we require complex dtype.
        data = np.load(self.examples[i], allow_pickle=True).astype(np.complex128)
        # Tranform data and generate observations.
        return self.transform(data)
        

class MassMappingDataset_Val(torch.utils.data.Dataset):
    """Loads the validation data."""

    def __init__(self, data_dir, transform):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object (class) that pre-processes the raw data into
                appropriate form for it to be fed into the model.
        """
        self.transform = transform

        self.examples = []

        #Collects the paths of all files.
        files = list(pathlib.Path(data_dir).iterdir())

        # Shuffle all the files.
        np.random.seed()
        np.random.shuffle(files)

        self.examples = files
    
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self, i):
        """Loads and returns a sample from the dataset at a given index."""

        # Cast input data from float64 to complex128 as we require complex dtype.
        data = np.load(self.examples[i], allow_pickle=True).astype(np.complex128)
        # Tranform data and generate observations.
        return self.transform(data)

class MassMappingDataset_Train(torch.utils.data.Dataset):
    """Loads the training data."""

    def __init__(self, data_dir, transform):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object (class) that pre-processes the raw data into
                appropriate form for it to be fed into the model.
        """
        self.transform = transform

        self.examples = []

        #Collects the paths of all files.
        files = list(pathlib.Path(data_dir).iterdir())

        # Shuffle all the files.
        np.random.seed()
        np.random.shuffle(files)
        
        self.examples = files
                
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self,i):
        """Loads and returns a sample from the dataset at a given index."""

        # Cast input data from float64 to complex128 as we require complex dtype.
        data = np.load(self.examples[i], allow_pickle=True).astype(np.complex128)
        # Tranform data and generate observations.
        return self.transform(data)