#Add data set information and configuration here.
import numpy as np
import torch
import pathlib


class MassMappingDataset_Test(torch.utils.data.Dataset):
    """Loads the test data."""
    def __init__(self, data_dir, transform, big_test=False, test_set=False):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
                'to_tensor' is a function that transforms a complex numpy array into a complex PyTorch tensor.
        """
        self.test_set = test_set
        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(data_dir).iterdir())

        # Shuffle list
        np.random.seed()
        np.random.shuffle(files)
        # random.seed()
        # random.shuffle(files)

        self.examples = files
        
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.examples)
    
    def __getitem__(self,i):
        """Loads and returns a sample from the dataset at a given index."""
        # Cast input data from float64 to complex128 as our inputs are complex
        data = np.load(self.examples[i], allow_pickle=True).astype(np.complex128)
        # Tranform data and generate observations
        return self.transform(data)
        

class MassMappingDataset_Val(torch.utils.data.Dataset):
    """Loads the validation data."""

    def __init__(self, data_dir, transform, big_test=False, test_set=False):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.
                'to_tensor' is a function that transforms a complex numpy array into a complex PyTorch tensor.
        """
        self.test_set = test_set
        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(data_dir).iterdir())

        # Shuffle list
        np.random.seed()
        np.random.shuffle(files)
        # random.seed()
        # random.shuffle(files)

        self.examples = files
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        """Loads and returns a sample from the dataset at a given index."""
        # Cast input data from float64 to complex128 as our inputs are complex
        data = np.load(self.examples[i], allow_pickle=True).astype(np.complex128)
        # Tranform data and generate observations
        return self.transform(data)




class MassMappingDataset_Train(torch.utils.data.Dataset):
    """Loads the training data."""

    def __init__(self, data_dir, transform):
        """
        Args:
            data_dir (path): The path to the dataset.
            transform (callable): A callable object that pre-processes the raw data into
                appropriate form. The transform function should take 'kspace', 'target',
                'attributes', 'filename', and 'slice' as inputs. 'target' may be null
                for test data.  
                'to_tensor' is a function that transforms a complex numpy array into a complex PyTorch tensor.
        """
        self.transform = transform

        self.examples = []
        files = list(pathlib.Path(data_dir).iterdir())

        # Shuffle list
        np.random.seed()
        np.random.shuffle(files)
        # random.seed()
        # random.shuffle(files)
        
        self.examples = files
                
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self,i):
        """Loads and returns a sample from the dataset at a given index."""
        # Cast input data from float64 to complex128 as our inputs are complex
        data = np.load(self.examples[i], allow_pickle=True).astype(np.complex128)
        # Tranform data and generate observations
        return self.transform(data)
