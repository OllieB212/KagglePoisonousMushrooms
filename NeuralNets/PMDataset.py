from typing import Tuple
import torch
from torch.utils.data import Dataset
import numpy as np


class PMDataset(Dataset):
    """
    This class implements the torch.utils.data.Dataset class for the dataset.

    Parameters
    ----------
    data_path : string to the path of the data CSV file.
    """

    def __init__(self, data_path: str, targets_path: str):

        self.X = torch.from_numpy(np.load(data_path))
        self.y = torch.from_numpy(np.load(targets_path))

        # shape information of the dataset.
        self.length = self.X.shape[0]

    # Implement __len__ :
    def __len__(self) -> int:
        return self.length

    # Implement __getitem__:
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.X[idx, :], torch.unsqueeze(self.y[idx], 0)


class DatasetPredict(Dataset):
    """
    This class implements the torch.utils.data.Dataset class for the dataset.

    Parameters
    ----------
    data_path : string to the path of the data CSV file.
    """

    def __init__(self, data_path: str):

        self.X = torch.from_numpy(np.load(data_path))

        # shape information of the dataset.
        self.length = self.X.shape[0]

    # Implement __len__ :
    def __len__(self) -> int:
        return self.length

    # Implement __getitem__:
    def __getitem__(self, idx: int):

        return self.X[idx, :]


class DatasetImpute(Dataset):
    """
    This class implements the torch.utils.data.Dataset class for the dataset.

    Parameters
    ----------
    data_path : string to the path of the data CSV file.
    """

    def __init__(self, data_path: str, targets_path: str):

        self.X = torch.from_numpy(np.load(data_path))
        self.y = torch.from_numpy(np.load(targets_path))

        # shape information of the dataset.
        self.length = self.X.shape[0]

    # Implement __len__ :
    def __len__(self) -> int:
        return self.length

    # Implement __getitem__:
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.X[idx, :], self.y[idx, :]
