import logging as log
import os
from glob import glob
from os import path
from typing import Any

import kornia as K
import tomli
import torch
import torchvision  # type: ignore
import torchvision.transforms.functional as F  # type: ignore
from torch import Tensor, nn

from .dataset import Dataset
from .scheduler import StepLRandWeightDecay
from .utils import (DatasetNotFoundError, InvalidConfigError, get_module,
                    pformat)

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
        version:str config version string
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
            try:
                with open(file, mode="r", encoding="utf-8") as f:
                    self.text = f.read()
                self.name = path.basename(file).removesuffix(".toml")
            except FileNotFoundError as err:
                raise InvalidConfigError(f"file not found: {err}")

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
        try:
            self.layers, self.layer_names = self._get_layers()
        except (KeyError, TypeError):
            raise InvalidConfigError("model layer definition missing or invalid")

    def save(self, clear: bool = False) -> None:
        """Save a copy of the config file to the run directory and optionally clear any data from prior runs."""
        log.info(f"saving config to {self.dir} clear={clear}")
        if clear:
            for file in glob(path.join(self.dir, "*.pt")):
                os.remove(file)
        if not path.exists(self.dir):
            os.makedirs(self.dir)
        with open(path.join(self.dir, "config.yaml"), mode="w", encoding="utf-8") as f:
            f.write(self.text)

    def data(self, typ: str) -> dict[str, Any]:
        if typ == "train":
            cfg = self.cfg["train_data"]
        elif typ == "test":
            cfg = self.cfg["test_data"]
        elif typ == "valid":
            cfg = self.cfg.get("valid_data", {})
        else:
            raise InvalidConfigError(f"invalid dataset type: {typ}")
        return cfg

    def dataset(self, root: str, typ: str = "test", device: str = "cpu",
                dtype: torch.dtype = torch.float32) -> Dataset:
        """Load a new train, test or valid dataset and applies normalization if defined.

        Raises DatasetNotFoundError if it does not exist - e.g. for optional valid dataset
        """
        cfg = self.data(typ)
        if len(cfg) == 0:
            raise DatasetNotFoundError(typ)
        train = cfg.get("train", False)
        batch_size = cfg.get("batch_size", 0)
        start = cfg.get("start", 0)
        end = cfg.get("end", 0)
        ds = Dataset(cfg["dataset"], root, train=train, batch_size=batch_size,
                     device=device, dtype=dtype, start=start, end=end)
        try:
            args = cfg["normalize"]
            log.debug(f"normalize: {args}")
            try:
                ds.data = F.normalize(ds.data, args[0], args[1], inplace=True)
            except (TypeError, IndexError) as err:
                raise InvalidConfigError(f"normalize: {err}")
        except KeyError:
            pass
        return ds

    def transforms(self) -> nn.Module | None:
        """Get the transforms defined on the image tensor from Kornia data augmentations"""
        try:
            config = self.cfg["transform"]["transforms"]
        except KeyError:
            return None
        transform = nn.Sequential()
        for args in config:
            transform.append(get_module(K.augmentation, args, desc="transform"))
        return transform

    def _get_layers(self) -> tuple[list[Any], list[str]]:
        cfg = self.cfg["model"]
        names = ["input"]
        for args in cfg["layers"]:
            defn = cfg.get(args[0])
            if defn is None:
                names.append(args[0])
            else:
                names.extend([sub[0] for sub in defn])
        return cfg["layers"], names

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
