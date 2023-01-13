import logging as log
import queue
from typing import Any

import kornia as K
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision  # type: ignore
from torch import Tensor, nn

from .utils import (InvalidConfigError, format_args, get_module, getargs,
                    init_logger, set_seed)

max_queue_size = 2
queue_poll_timeout = 0.05


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

    def __init__(self, name: str, root: str, train: bool = False, batch_size: int = 0,
                 device: str = "cpu", dtype: torch.dtype = torch.float32,
                 start: int = 0, end: int = 0):
        try:
            ds = getattr(torchvision.datasets, name)(root=root, train=train, download=True)
        except AttributeError as err:
            raise InvalidConfigError(f"invalid dataset: {err}")
        if len(ds.data) != len(ds.targets):
            raise ValueError("expect same number of images and labels")
        if end == 0:
            end = len(ds.targets)
        self.data = to_tensor(ds.data, device, dtype)[start:end]
        self.targets = to_tensor(ds.targets, device, torch.int64)[start:end]
        self.indexes: Tensor | None = None
        if name == "MNIST":
            self.classes = [str(i) for i in range(len(ds.classes))]
        elif name == "CIFAR10":
            self.classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        else:
            self.classes = ds.classes
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
    def image_shape(self) -> torch.Size:
        """Shape of each image"""
        return self.data.size()[1:]

    def clone(self) -> "Dataset":
        """Return a new copy of the dataset - cloning the tensor data"""
        log.debug("clone dataset")
        ds = Dataset.__new__(Dataset)
        ds.classes = self.classes
        ds.batch_size = self.batch_size
        ds.data = self.data.clone()
        ds.targets = self.targets.clone()
        if self.indexes is not None:
            ds.indexes = self.indexes.clone()
        else:
            ds.indexes = None
        return ds

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
        self.batch = -1
        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        self.batch += 1
        if self.batch >= len(self):
            raise StopIteration
        return self[self.batch]

    def __str__(self) -> str:
        """Print summary of data size and type"""
        return "images: {}{} device={}\nlabels: {}{} nclasses={} batch_size={}".format(
            self.data.dtype, list(self.data.size()), self.data.device,
            self.targets.dtype, list(self.targets.size()), len(self.classes), self.batch_size
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

    def filter(self, predict: Tensor, target_class: int = -1, errors_only: bool = False) -> "Dataset":
        """Apply filtering by class and/or errors"""
        log.debug(f"filter images: errors_only={errors_only} class={target_class}")
        ix = None
        if errors_only:
            if target_class >= 0:
                tgts = torch.where(self.targets != predict, self.targets, -1)
                ix = (tgts == target_class).nonzero(as_tuple=True)[0]
            else:
                ix = (self.targets != predict).nonzero(as_tuple=True)[0]
        elif target_class >= 0:
            ix = (self.targets == target_class).nonzero(as_tuple=True)[0]
        self.indexes = ix
        return self


class Transforms(nn.Module):
    """Transforms holds the list of Kornia data augmentation transforms to apply"""

    def __init__(self, config: list[Any]):
        super().__init__()
        self.seq = nn.Sequential()
        self._repr = "transforms("
        for argv in config:
            typ, args, kwargs = getargs(argv)
            self.seq.append(get_module("transform", K.augmentation, typ, args, kwargs))
            self._repr += "\n  " + typ + format_args(args, kwargs)
        self._repr += "\n)"

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)

    def __str__(self):
        return self._repr


class DataLoader:
    """DataLoader loads batches of data from the training set and optionally applies shuffling and augmentation transform

    This will run in a separate thread - if pin_memory is set then CPU tensors are loaded into pinned memory.
    """

    def __init__(self, pin_memory: bool = False):
        self.pin_memory = pin_memory
        self.debug = (log.getLogger("").getEffectiveLevel() == log.DEBUG)

    def start(self, ds: Dataset, shuffle: bool = False, transform: Transforms | None = None, seed: int = 1) -> None:
        self.batches = len(ds)
        self._str = str(ds) + "\n" + str(transform) if transform else str(ds)
        self.queue = mp.Queue(maxsize=max_queue_size)  # type: ignore
        self.done = mp.Event()
        self.loader = Loader(ds, shuffle, transform, self.queue, self.done, seed=seed,
                             pin_memory=self.pin_memory, debug=self.debug)
        self.loader.start()

    def shutdown(self) -> None:
        log.info(f"DataLoader: shutdown")
        self.done.set()
        self.loader.join()

    def __next__(self) -> tuple[Tensor, Tensor]:
        self.batch += 1
        if self.batch >= self.batches:
            raise StopIteration
        i, data, targets = self.queue.get()
        if i != self.batch:
            raise RuntimeError(f"invalid batch {i} returned from loader - expecting {self.batch}")
        return data, targets

    def __len__(self) -> int:
        return self.batches

    def __iter__(self):
        self.batch = -1
        return self

    def __str__(self) -> str:
        return self._str


class Loader(mp.Process):
    """Spawned subprocess which will apply transforms and return batches of data via a queue"""

    def __init__(self, ds: Dataset, shuffle: bool, transform: nn.Module | None, q: mp.Queue, done,
                 seed: int = 1,  pin_memory: bool = False, debug: bool = False):
        super().__init__()
        set_seed(seed)
        self.ds = ds
        self.shuffle = shuffle
        self.transform = transform
        self.queue = q
        self.done = done
        self.pin_memory = pin_memory
        self.debug = debug
        self.daemon = True

    def run(self):
        init_logger(debug=self.debug)
        log.info(f"Loader: batches={len(self.ds)} pin_memory={self.pin_memory}")
        while True:
            if self.shuffle:
                self.ds.shuffle()
            for i, (data, targets) in enumerate(self.ds):
                if self.transform:
                    with torch.no_grad():
                        data = self.transform(data)
                if self.pin_memory:
                    data, targets = data.pin_memory(), targets.pin_memory()
                if self.put(i, data, targets):
                    log.debug("Loader: end run")
                    self.queue.close()
                    return

    def put(self, i, data, targets):
        while not self.done.is_set():
            try:
                self.queue.put((i, data, targets), timeout=queue_poll_timeout)
                return self.done.is_set()
            except queue.Full:
                pass
        return True


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
