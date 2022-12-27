import logging as log
import sys

import numpy as np
import torch
import torchvision  # type: ignore
from torch import Tensor, nn


class Dataset:
    """Daraset object holds training or test data in tensor format.

    This is an iterable object where the __item__ accessor returns one batch of data.

    Args:
        name:str            torchvision dataset name
        root:str            root directory to save downloaded data
        train:bool          True for training set, False for test set
        batch_size:int      batch size, or entire dataset is single batch if 0
        device:torch.device move tensors to this device
        dtype:torch.dtype   convert images this data type - scales the values by 1/255 if a floating point type
        start:int           start index (default 0 = start)
        end:int             end index (default 0 = end)

    Attributes:
        data:Tensor         image data in [N,C,H,W] format
        targets:Tensor      labels as 2D array of ints where first dim is 0 for targets and 1 for predictions
        indexes:Tensor      indexes used to shuffle or filter items from the dataset
        classes:list[str]   list of class names
        batch_size:int      current batch size - defaults to len(dataset)
    """

    def __init__(self,
                 name: str,
                 root: str,
                 train: bool = False,
                 batch_size: int = 0,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 start: int = 0,
                 end: int = 0):
        try:
            ds = getattr(torchvision.datasets, name)(root=root, train=train, download=True)
        except AttributeError as err:
            print(f"Error: invalid dataset {name} - {err}")
            sys.exit(1)
        if len(ds.data) != len(ds.targets):
            raise ValueError("expect same number of images and labels")
        if end == 0:
            end = len(ds.targets)
        self.data = to_tensor(ds.data, device, dtype)[start:end]
        self.targets = to_tensor(ds.targets, device, torch.int64)[start:end]
        self.indexes: Tensor | None = None
        self.classes: list[str] = []
        if len(ds.classes) > 0 and ds.classes[0][0].isalpha():
            self.classes = ds.classes
        else:
            self.classes = [str(i) for i in range(len(ds.classes))]
        if batch_size > 0:
            self.batch_size = batch_size
        else:
            self.batch_size = len(self.targets)

    @property
    def nitems(self) -> int:
        """Number of images in dataset"""
        if self.indexes is None:
            return self.data.size()[0]
        else:
            return len(self.indexes)

    @property
    def channels(self) -> int:
        """Number of channels in each image"""
        return self.data.size()[1]

    @property
    def image_shape(self) -> tuple[int, int]:
        """Shape of each image"""
        size = self.data.size()
        return (size[2], size[3])

    def to(self, device="cpu") -> "Dataset":
        """Move tensors to given device and return a new copy of the dataset"""
        if str(device) == str(self.data.device):
            return self
        log.debug(f"move dataset from {self.data.device} to {device}")
        ds = Dataset.__new__(Dataset)
        ds.classes = self.classes
        ds.batch_size = self.batch_size
        ds.data = self.data.to(device)
        ds.targets = self.targets.to(device)
        if self.indexes is not None:
            ds.indexes = self.indexes.to(device)
        else:
            ds.indexes = None
        return ds

    def get(self, i: int, t: Tensor) -> Tensor:
        """Get the ith elemnt from t taking account of any index selection"""
        if self.indexes is None:
            return t[i]
        else:
            return t[self.indexes[i]]

    def indexOf(self, i: int) -> int:
        """Get the index of the ith item"""
        if self.indexes is None:
            return i
        else:
            return int(self.indexes[i])

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        """Get next batch of images and targets, optionally applying index selection"""
        start = i*self.batch_size
        end = min(start+self.batch_size, self.nitems)
        if self.indexes is None:
            return self.data[start:end], self.targets[start:end]
        else:
            ix = self.indexes[start:end]
            return self.data.index_select(0, ix), self.targets.index_select(0, ix)

    def __len__(self) -> int:
        """Get number of batches of data."""
        if self.batch_size > self.nitems:
            return 1
        return self.nitems // self.batch_size

    def __iter__(self):
        """Iterare over batches of data"""
        return DatasetIterator(self)

    def __str__(self) -> str:
        """Print summary of data size and type"""
        return "images: {}{}\nlabels: {}{} nclasses={} batch_size={}".format(
            self.data.dtype, list(self.data.size()), self.targets.dtype, list(self.targets.size()),
            len(self.classes), self.batch_size
        )

    def reset(self) -> "Dataset":
        """Reset filtering to select all items and restore default ordering"""
        self.indexes = None
        return self

    def shuffle(self) -> "Dataset":
        """ Randomly shuffle the indexes"""
        log.debug("shuffle data")
        ix = torch.randperm(self.nitems, device=self.data.device)
        if self.indexes is not None:
            self.indexes = self.indexes.index_select(0, ix)
        else:
            self.indexes = ix
        return self

    def filter(self, predict: Tensor, target_class: int | None = None, errors_only: bool = False) -> "Dataset":
        """Apply filtering by class and/or errors"""
        ix = None
        if errors_only:
            if target_class is None:
                ix = (self.targets != predict).nonzero(as_tuple=True)[0]
            else:
                tgts = torch.where(self.targets != predict, self.targets, -1)
                ix = (tgts == target_class).nonzero(as_tuple=True)[0]
        elif target_class is not None:
            ix = (self.targets == target_class).nonzero(as_tuple=True)[0]
        self.indexes = ix
        return self


class DatasetIterator:
    """DatasetIterator is a helper class to iterate over Dataset batches"""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.batch = 0

    def __next__(self) -> tuple[Tensor, Tensor]:
        if self.batch < len(self.dataset):
            res = self.dataset[self.batch]
            self.batch += 1
            return res
        raise StopIteration


# utils
def to_tensor(data, device, dtype) -> Tensor:
    if isinstance(data, np.ndarray):
        if data.ndim == 4:
            data = data.transpose((0, 3, 1, 2))
        data = torch.from_numpy(data).contiguous()
    elif isinstance(data, list):
        data = Tensor(data)
    data = data.to(device, dtype)
    if data.dtype.is_floating_point:
        data /= 255.0
    size = data.size()
    if len(size) == 3:
        data = data.view(size[0], 1, size[1], size[2])
    return data
