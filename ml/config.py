import logging as log
import os
from glob import glob
from os import path
from typing import Any

import tomli
import torch
import torchvision  # type: ignore
import torchvision.transforms.functional as F  # type: ignore
from torch import Tensor, nn

from .dataset import (Dataset, ImagenetDataset, LMDBDataset, Loader,
                      MultiProcessLoader, SingleProcessLoader, TensorDataset,
                      Transforms)
from .scheduler import StepLRandWeightDecay
from .utils import InvalidConfigError, pformat

default_weight_init = {
    "Linear": ["kaiming_normal", {"nonlinearity": "relu"}],
    "Conv": ["kaiming_normal", {"nonlinearity": "relu"}],
    "BatchNorm": ["constant", 1],
}

default_bias_init = {
    "Linear": ["constant", 0],
    "Conv": ["constant", 0],
    "BatchNorm": ["constant", 0],
}


class Index(tuple):
    """Index is a unique id to reference each layer in the model.

    For a sequential stack of layers it will be a single integer. For more complex modules it will be a tuple.
    e.g. ["Conv2d", "RelU", ["Add", ["Conv2d", "BatchNorm2d", ...], ["Conv2d", "BatchNorm2d"]], "ReLU", ... ]
        Index(2, 0, 1) is the first BatchNorm2d layer in the list
    """

    def __new__(self, ixs):
        return tuple.__new__(Index, ixs)

    def next(self) -> "Index":
        if len(self) == 0:
            return Index((0,))
        elif len(self) == 1:
            return Index((self[0]+1,))
        else:
            return Index((*self[:-1], self[-1]+1))

    def format(self) -> str:
        if len(self) > 0:
            return "  "*(len(self)-1) + f"{self[-1]:2}"
        else:
            return ""

    def __str__(self) -> str:
        return ".".join([str(ix) for ix in self])


class Config():
    """Config object has results of config data parsed from the toml file.

    Args:
        file:str    config file in toml format
        data:str    content of config if loading from string
        name:str    config name if loading from string
        rundir:str  root directory to save run state
        epochs:int  number of epochs to train - if non-zero overrides the config
        seed:int    random number seed - if non-zero overrides the config

    Either file or name and data must be set

    Attributes:
        name:str    basename of config
        version:str config version string or "1" if not defined
        text:str    raw content
        half:bool   set if to use half precision
        dir:str     run directory path
        cfg:dict    parsed config data
        data:dict   dataset config
        train:dict  training config
    """

    def __init__(self, rundir: str, name: str = "", data: str = "", file: str = "",  epochs: int = 0, seed: int = 0):
        if data:
            if not name:
                raise ValueError("Config: name must be set if loading from string")
            self.name = name
            self.text = data
        else:
            log.debug(f"load config from {file}")
            with open(file, mode="r", encoding="utf-8") as f:
                self.text = f.read()
            base = path.basename(file).removesuffix(".toml")
            if base == "config":
                self.name = path.basename(path.split(path.dirname(file))[0])
            else:
                self.name = base
        try:
            self.cfg = tomli.loads(self.text)
        except tomli.TOMLDecodeError as err:
            raise InvalidConfigError(f"error decoding config: {err}")
        self.version = str(self.cfg.get("version", "1"))
        self.half = self.cfg.get("half", False)
        self.dir: str = path.join(rundir, self.name, self.version)
        self.train = self.cfg.get("train", {})
        self.transform = self.cfg.get("transform", {})
        self.weight_init = default_weight_init.copy()
        self.weight_init.update(self.cfg.get("weight_init", {}))
        self.bias_init = default_bias_init.copy()
        self.bias_init.update(self.cfg.get("bias_init", {}))
        if seed != 0:
            self.seed = seed
        else:
            self.seed = int(self.train.get("seed", 1))
        if epochs != 0:
            self.epochs = epochs
        else:
            self.epochs = int(self.train.get("epochs", 100))

    def save(self, clear: bool = False) -> None:
        """Save a copy of the config file to the run directory and optionally clear any data from prior runs."""
        log.info(f"saving config to {self.dir} clear={clear}")
        if clear:
            for file in glob(path.join(self.dir, "*.pt")):
                os.remove(file)
        if not path.exists(self.dir):
            os.makedirs(self.dir)
        with open(path.join(self.dir, "config.toml"), mode="w", encoding="utf-8") as f:
            f.write(self.text)

    def data(self, typ: str) -> dict[str, Any]:
        if typ == "train":
            cfg = self.cfg.get("train_data", {})
        elif typ == "test":
            cfg = self.cfg.get("test_data", {})
        elif typ == "valid":
            cfg = self.cfg.get("valid_data", {})
        else:
            raise InvalidConfigError(f"invalid dataset type: {typ}")
        return cfg

    def dataset(self, root: str, typ: str = "test", device: str = "cpu") -> Dataset:
        """Load a new train, test or valid dataset using config from the [*_data] section"""
        cfg = self.data(typ)
        if len(cfg) == 0:
            raise InvalidConfigError(f"{typ} dataset configuration not found")
        name = cfg.get("dataset", "")
        is_train = cfg.get("train", False)
        transforms = self.transforms(typ)
        start = cfg.get("start", 0)
        end = cfg.get("end", 0)
        resize = cfg.get("resize", 256)
        lmdb_dir = path.join(root, "lmdb", name, "train" if is_train else "test", str(resize))
        if path.exists(lmdb_dir):
            return LMDBDataset(name, root, transforms, resize, train=is_train, device=device, start=start, end=end)
        elif name == "Imagenet":
            return ImagenetDataset(root, transforms, resize, train=is_train, device=device, start=start, end=end)
        else:
            return TensorDataset(name, root, transforms, train=is_train, device=device, start=start, end=end)

    def dataloader(self, typ: str = "test", pin_memory: bool = False) -> Loader:
        """ Get a new data loader based on data config section"""
        cfg = self.data(typ)
        batch_size = cfg.get("batch_size", 0)
        shuffle = cfg.get("shuffle", False)
        if cfg.get("multi_process"):
            return MultiProcessLoader(batch_size, shuffle, pin_memory)
        else:
            return SingleProcessLoader(batch_size, shuffle, pin_memory)

    def transforms(self, typ: str = "train") -> Transforms | None:
        """Get the transforms defined on the image tensor from Kornia data augmentations"""
        try:
            cfg = self.data(typ)["transform"]
            if not isinstance(cfg, list):
                raise InvalidConfigError(f"invalid transform for {typ} - should be a list")
            return Transforms(cfg)
        except KeyError:
            return None

    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create a new optimizer based on config [train] optimizer setting"""
        try:
            typ, kwargs = self.train["optimizer"]
            cls = getattr(torch.optim, typ)
            optimizer = cls(model.parameters(), **kwargs)
        except (AttributeError, KeyError, TypeError) as err:
            raise InvalidConfigError(f"optimizer: {err}")
        return optimizer

    def scheduler(self, optimizer: torch.optim.Optimizer) -> tuple[Any, str]:
        try:
            arglist = self.train["scheduler"]
        except KeyError:
            return None, ""
        try:
            args = arglist.copy()
            kwargs = args.pop() if len(args) > 1 and isinstance(args[-1], dict) else {}
            metric = args[1] if len(args) > 1 else {}
            if args[0] == "StepLRandWeightDecay":
                cls = StepLRandWeightDecay
            else:
                cls = getattr(torch.optim.lr_scheduler, args[0])
            scheduler = cls(optimizer, **kwargs)
        except (AttributeError, KeyError, TypeError) as err:
            raise InvalidConfigError(f"scheduler: {err}")
        return scheduler, metric

    def __str__(self):
        return pformat(self.cfg)
