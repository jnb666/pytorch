import json
import logging as log
import math
import os
import queue
import sys
import time
from collections import deque
from os import path
from typing import Any

import kornia
import lmdb  # type: ignore
import numpy as np
import torch
import torch.multiprocessing as mp
import torchvision  # type: ignore
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode  # type: ignore
from torchvision.transforms import functional as F  # type: ignore

from .utils import (InvalidConfigError, RunInterrupted, format_args,
                    get_module, getargs, init_logger, set_seed)

max_queue_size = 4
queue_poll_timeout = 0.1


class Transforms(nn.Module):
    """Transforms holds the list of Kornia data augmentation transforms to apply"""

    def __init__(self, config: list[Any]):
        super().__init__()
        self.seq = nn.Sequential()
        self._repr = "transform("
        for argv in config:
            typ, args, kwargs = getargs(argv)
            log.debug(f"transform: {typ} {args} {kwargs}")
            self.seq.append(get_module("transform", kornia.augmentation, typ, args, kwargs))
            self._repr += "\n  " + typ + format_args(args, kwargs)
        self._repr += "\n)"

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            return self.seq(x)

    def __str__(self):
        return self._repr


class Dataset:
    """Dataset base class"""

    def __init__(self, targets: Tensor, classes: list[str], transforms: Transforms | None,
                 size: torch.Size, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.targets = targets
        self.indexes: Tensor | None = None
        self.classes = classes
        self.transform = transforms
        self.size = size
        self.device = device
        self.dtype = dtype

    def image_shape(self) -> torch.Size:
        """Shape of each image"""
        if self.transform is not None:
            shape = self.transform(self.get_data(0)).size()[1:]
        else:
            shape = self.size
        return shape

    def index_of(self, i: int) -> int:
        """Get the index of the ith item"""
        if self.indexes is None:
            return i
        else:
            return int(self.indexes[i])

    def __len__(self) -> int:
        """Number of images in dataset"""
        if self.indexes is None:
            return len(self.targets)
        else:
            return len(self.indexes)

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def get_data(self, ix: int) -> Tensor:
        raise NotImplementedError("Abstract method")

    def get_range(self, start: int, end: int, nofilter: bool = False,
                  data: Tensor | None = None, targets: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if data is None:
            data = torch.empty((end-start,)+self.size, dtype=self.dtype, device=self.device)
        if self.indexes is None or nofilter:
            for i in range(start, end):
                data[i-start] = self.get_data(i)
            targets = select(self.targets, start, end, out=targets)
        else:
            for i in range(start, end):
                data[i-start] = self.get_data(int(self.indexes[i]))
            targets = torch.index_select(self.targets, 0, self.indexes[start:end], out=targets)
        return data, targets

    def get_indexes(self, ix: Tensor, nofilter: bool = False,
                    data: Tensor | None = None, targets: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if data is None:
            data = torch.empty((len(ix),)+self.size, dtype=self.dtype, device=self.device)
        if self.indexes is not None and not nofilter:
            ix = self.indexes.index_select(0, ix)
        for i, id in enumerate(ix.tolist()):
            data[i] = self.get_data(int(id))
        targets = torch.index_select(self.targets, 0, ix, out=targets)
        return data, targets

    def __getitem__(self, key: int | slice | Tensor) -> tuple[Tensor, Tensor]:
        """Get image and target or slice of images and targets, applying index filtering if defined"""
        if isinstance(key, int):
            i = self.index_of(key)
            return self.get_data(i), self.targets[i]
        elif isinstance(key, slice):
            if key.step is not None:
                raise ValueError(f"slice step not supported")
            return self.get_range(key.start, key.stop)
        elif isinstance(key, Tensor):
            return self.get_indexes(key)
        else:
            raise TypeError("invalid index type")

    def reset(self) -> None:
        """Reset filtering to select all items and restore default ordering"""
        self.indexes = None

    def filter(self, predict: Tensor | None, target_class: int = -1, errors_only: bool = False):
        """Apply filtering by class and / or errors """
        log.debug(f"filter images: errors_only={errors_only} class={target_class}")
        if errors_only and predict is not None:
            if target_class >= 0:
                tgts = torch.where(self.targets != predict, self.targets, -1)
                self.indexes = (tgts == target_class).nonzero(as_tuple=True)[0]
            else:
                self.indexes = (self.targets != predict).nonzero(as_tuple=True)[0]
        elif target_class >= 0:
            self.indexes = (self.targets == target_class).nonzero(as_tuple=True)[0]
        else:
            self.indexes = None

    def __str__(self) -> str:
        """Print summary of data size, type and any transforms defined"""
        s = "images: {}{} device={} type={}\nlabels: {}{} nclasses={}".format(
            self.dtype, list((len(self),) + self.size), self.device, type(self).__name__,
            self.targets.dtype, list(self.targets.size()), len(self.classes))
        if self.transform is not None:
            s += "\n" + str(self.transform)
        return s


class TensorDataset(Dataset):
    """TensorDaraset object holds training or test data in tensor format.

    This is an iterable object where the __item__ accessor returns one batch of data.

    Args:
        name: str            torchvision dataset name
        root: str            root directory to save downloaded data
        transforms           optional image transforms
        train: bool          True for training set, False for test set
        device: torch.device move tensors to this device
        dtype: torch.dtype   convert images this data type - scales the values by 1/255 if a floating point type
        start: int           start index(default 0=start)
        end: int             end index(default 0=end)

    Attributes:
        data: Tensor         image data in [N, C, H, W] format
        targets: Tensor      labels as 2D array of ints where first dim is 0 for targets and 1 for predictions
        indexes: Tensor      indexes used to shuffle or filter items from the dataset
        classes: list[str]   list of class names
    """

    def __init__(self, name: str, root: str, transforms: Transforms | None = None, train: bool = False,
                 device: str = "cpu", dtype: torch.dtype = torch.float32, start: int = 0, end: int = 0):
        datadir = path.join(root, name)
        try:
            ds = getattr(torchvision.datasets, name)(root=datadir, train=train, download=True)
        except AttributeError as err:
            raise InvalidConfigError(f"invalid dataset: {err}")
        if len(ds.data) != len(ds.targets):
            raise ValueError("expect same number of images and labels")
        if end == 0:
            end = len(ds.targets)
        self.data = to_tensor(ds.data, device, dtype)[start:end]
        targets = to_tensor(ds.targets, device, torch.int64)[start:end]
        if name == "MNIST":
            classes = [str(i) for i in range(len(ds.classes))]
        elif name == "CIFAR10":
            classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        else:
            classes = ds.classes
        super().__init__(targets, classes, transforms, self.data.shape[1:], device, dtype)

    def get_data(self, ix: int) -> Tensor:
        """Get single image using raw index with no filtering"""
        return self.data[ix]

    def get_range(self, start: int, end: int, nofilter: bool = False,
                  data: Tensor | None = None, targets: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self.indexes is None or nofilter:
            data = select(self.data, start, end, out=data)
            targets = select(self.targets, start, end, out=targets)
        else:
            ix = self.indexes[start:end]
            data = torch.index_select(self.data, 0, ix, out=data)
            targets = torch.index_select(self.targets, 0, ix, out=targets)
        return data, targets

    def get_indexes(self, ix: Tensor, nofilter: bool = False,
                    data: Tensor | None = None, targets: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if self.indexes is not None and not nofilter:
            ix = self.indexes.index_select(0, ix)
        data = torch.index_select(self.data, 0, ix, out=data)
        targets = torch.index_select(self.targets, 0, ix, out=targets)
        return data, targets


class ImagenetDataset(Dataset):
    """Class to load the Imagenet dataset from disk"""

    def __init__(self, root: str, transforms: Transforms | None, resize: int, train: bool = False,
                 device: str = "cpu", dtype: torch.dtype = torch.float32, start: int = 0, end: int = 0):
        self.root = path.join(root, "Imagenet")
        self.train = train
        self._resize = resize
        with open(path.join(self.root, "imagenet_class_index.json")) as f:
            self.class_index = json.load(f)
        self.label_to_cls = {}
        self.labels = []
        classes = []
        for i in range(1000):
            label, name = self.class_index[str(i)]
            classes.append(name.replace("_", " "))
            self.labels.append(label)
            self.label_to_cls[label] = i
        self.file_index: list[str] = []
        if train:
            targets = self._get_train_data()
        else:
            targets = self._get_val_data()
        tgt = torch.tensor(targets, device=device, dtype=torch.int64)
        shape = torch.Size((3, resize, resize))
        super().__init__(tgt, classes, transforms, shape, device, dtype)

    def get_data(self, ix: int) -> Tensor:
        """Get single image using raw index with no filtering"""
        img = Image.open(self.file_index[ix])
        img = F.resize(img, (self._resize, self._resize), interpolation=InterpolationMode.BILINEAR)
        data = F.pil_to_tensor(img)
        channels = data.shape[0]
        if channels == 1:
            # some images are grayscale
            data = data.tile((3, 1, 1))
        elif channels == 4:
            # or rgba
            data = data[:3, :, :]
        data = data.to(self.device, self.dtype)
        if self.dtype.is_floating_point:
            data /= 255.0
        return data

    def _get_train_data(self):
        targets = []
        dir = path.join(self.root, "ILSVRC", "Data", "CLS-LOC", "train")
        file_list = path.join(self.root, "ILSVRC", "ImageSets", "CLS-LOC", "train_cls.txt")
        with open(file_list) as f:
            for line in f:
                file, id = line.removesuffix("\n").split(" ")
                self.file_index.append(path.join(dir, file + ".JPEG"))
                cls = self.label_to_cls[file.split("/")[0]]
                targets.append(cls)
        return targets

    def _get_val_data(self):
        targets = []
        with open(path.join(self.root, "ILSVRC2012_val_labels.json")) as f:
            val_labels = json.load(f)
        dir = path.join(self.root, "ILSVRC", "Data", "CLS-LOC", "val")
        for file in sorted(val_labels.keys()):
            cls = self.label_to_cls[val_labels[file]]
            targets.append(cls)
            self.file_index.append(path.join(dir, file))
        return targets

    def export_to_lmdb(self, root: str):
        """Export dataset to lmdb database"""
        dir = path.join(root, "lmdb", "Imagenet", "train" if self.train else "test", str(self._resize))
        if not path.exists(dir):
            os.makedirs(dir)
        num_images = len(self)
        log.info(f"Imagenet: export {num_images} images to LMDB at {dir}")
        labels = {
            "image_size": self.size,
            "classes": self.classes,
            "targets": self.targets,
        }
        torch.save(labels, path.join(dir, "labels.pt"))
        map_size = 10 * math.prod(self.size) * num_images
        env = lmdb.open(dir, map_size=map_size)
        for i in range(0, num_images, 1000):
            with env.begin(write=True) as txn:
                for j in range(0, min(1000, num_images-i)):
                    sys.stdout.write(f"image {i+j+1} / {num_images}  \r")
                    key = f"{i+j:08}"
                    img = self.get_data(i+j).numpy()
                    txn.put(key.encode("ascii"), img.tobytes())
        env.close()


class LMDBDataset(Dataset):
    """Dataset where elements are stored in a lmdb memory mapped DB with blosc compression"""

    def __init__(self, name: str, root: str, transforms: Transforms | None, resize: int, train: bool = False,
                 device: str = "cpu", dtype: torch.dtype = torch.float32, start: int = 0, end: int = 0):
        if dtype != torch.float32 and dtype != torch.uint8:
            raise ValueError(f"unsupported data type {dtype}")
        self.dir = path.join(root, "lmdb", name, "train" if train else "test", str(resize))
        labels = torch.load(path.join(self.dir, "labels.pt"), map_location=device)
        super().__init__(labels["targets"], labels["classes"], transforms, labels["image_size"], device, dtype)
        self.open()

    def open(self) -> None:
        if hasattr(self, "env"):
            return
        self.env = lmdb.open(self.dir, readonly=True)

    def get_data(self, ix: int) -> Tensor:
        with self.env.begin() as txn:
            buf = txn.get(f"{ix:08}".encode("ascii"))
        if buf is None:
            raise FileNotFoundError(f"LMDBDataset: image {ix} not found")
        img = np.frombuffer(buf, dtype=np.uint8).reshape(self.size)
        if self.dtype == torch.float32:
            data = torch.from_numpy(img.astype(np.float32) / 255.0)
        else:
            data = torch.from_numpy(img.copy())
        return data.to(self.device, self.dtype)

    def close(self) -> None:
        if hasattr(self, "env"):
            self.env.close()
            del self.env


class Loader:
    """Loader base class """

    def __init__(self, batch_size: int, shuffle: bool, pin_memory: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.batches = 0
        self.batch = -1

    @ property
    def nitems(self) -> int:
        return len(self.ds) if self.ds else 0

    def release(self, id: int) -> None:
        pass

    def __len__(self):
        return self.batches

    def __iter__(self):
        raise NotImplementedError("abstract method")

    def __next__(self) -> tuple[Tensor, Tensor, int]:
        raise NotImplementedError("abstract method")

    def start(self, ds: Dataset, max_items: int = 0) -> None:
        self.ds = ds
        nitems = len(ds)
        if max_items:
            nitems = min(nitems, max_items)
        if self.batch_size == 0:
            self.batch_size = nitems
        self.batches = max(1, nitems // self.batch_size)
        log.info("{}: batches={} x {} shuffle={} pin_memory={}".format(
            type(self).__name__, self.batches, self.batch_size, self.shuffle, self.pin_memory))

    def shutdown(self) -> None:
        pass


class SingleProcessLoader(Loader):
    """Iterator to load batches of data and apply shuffle and transforms in same process"""

    def __init__(self, batch_size: int, shuffle: bool, pin_memory: bool = False):
        super().__init__(batch_size, shuffle, pin_memory)
        self.ix = None

    def start(self, ds: Dataset, max_items: int = 0) -> None:
        super().start(ds, max_items)
        ds.open()

    def __iter__(self):
        self.batch = -1
        if self.shuffle:
            log.debug("shuffle data")
            self.ix = torch.randperm(len(self.ds), device=self.ds.targets.device)
        return self

    def __next__(self) -> tuple[Tensor, Tensor, int]:
        self.batch += 1
        if self.batch >= self.batches:
            raise StopIteration
        start = self.batch*self.batch_size
        end = min(start+self.batch_size, len(self.ds))
        if self.ix is None:
            data, targets = self.ds.get_range(start, end)
        else:
            data, targets = self.ds.get_indexes(self.ix[start:end])
        if self.pin_memory:
            data, targets = data.pin_memory(), targets.pin_memory()
        return data, targets, 0


class MultiProcessLoader(Loader):
    """Iterator to load batches of data and apply shuffle and transforms in separate forked subprocesses"""

    def __init__(self, batch_size: int, shuffle: bool, pin_memory: bool = False):
        super().__init__(batch_size, shuffle, pin_memory)
        self.proc: mp.Process | None = None
        self.queue = mp.SimpleQueue()  # type: ignore
        self.rqueue = mp.SimpleQueue()  # type: ignore

    def start(self, ds: Dataset, max_items: int = 0) -> None:
        ds.close()   # else error passing env to subprocess
        super().start(ds, max_items)
        self.proc = mp.Process(target=worker, args=(
            self.queue, self.rqueue, self.ds, self.batches, self.batch_size, self.shuffle, self.pin_memory
        ))
        self.proc.start()

    def release(self, id: int) -> None:
        self.rqueue.put(id)

    def shutdown(self) -> None:
        if self.proc is not None:
            log.debug("MultiProcessLoader: stop worker")
            self.proc.terminate()

    def __iter__(self):
        self.batch = -1
        return self

    def __next__(self) -> tuple[Tensor, Tensor, int]:
        self.batch += 1
        if self.batch >= self.batches:
            raise StopIteration
        data, targets, id = self.queue.get()
        return data, targets, id


def worker(q, rq, ds: Dataset, batches: int, batch_size: int, shuffle: bool, pin_memory: bool):
    ds.open()
    pool = Allocator(max_queue_size, batch_size, ds.size, pin_memory=pin_memory)
    index = None
    while True:
        if shuffle:
            index = torch.randperm(len(ds))
        for i in range(batches):
            id, data, targets = pool.alloc(rq)
            start = i*batch_size
            end = min(start+batch_size, len(ds))
            if index is None:
                ds.get_range(start, end, data=data, targets=targets)
            else:
                ds.get_indexes(index[start:end], data=data, targets=targets)
            if pin_memory:
                data, targets = data.pin_memory(), targets.pin_memory()
            q.put((data, targets, id))


class Allocator():
    """Allocator preallocates a pool of shared memory and returns the unused chunks"""

    def __init__(self, pool_size: int, batch_size: int, image_shape: torch.Size, pin_memory: bool = False):
        self.data, self.targets = [], []
        self.free_list: deque = deque()
        for i in range(pool_size):
            data = torch.empty((batch_size,)+image_shape, dtype=torch.float32).share_memory_().pin_memory()
            targets = torch.empty((batch_size,), dtype=torch.int64).share_memory_().pin_memory()
            if pin_memory:
                data, targets = data.pin_memory(), targets.pin_memory()
            self.data.append(data)
            self.targets.append(targets)
            self.free_list.append(i)

    def alloc(self, rq) -> tuple[int, Tensor, Tensor]:
        """Get next available free tensors from pool - blocks if none are available"""
        if len(self.free_list) == 0:
            self.free_list.append(rq.get())
        id = self.free_list.popleft()
        return id, self.data[id], self.targets[id]


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


def select(t: Tensor, start: int, end: int, out: Tensor | None = None) -> Tensor:
    if out is None:
        return t[start:end]
    else:
        return out.copy_(t[start:end])
