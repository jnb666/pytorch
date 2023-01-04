import logging as log
import sys
import time
from glob import glob
from os import path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn

from .config import Config
from .dataset import Dataset
from .model import Model
from .rpc import Database
from .trainer import Stats
from .utils import InvalidConfigError


class Loader:
    """Loader abstract base class"""

    def __init__(self, rundir: str, datadir: str, device: str = "cpu"):
        self.rundir = rundir
        self.datadir = datadir
        self.device = device
        self.name = ""
        self.epoch = -999

    def get_models(self) -> list[str]:
        raise NotImplementedError("abstract method")

    def get_epochs(self) -> list[int]:
        raise NotImplementedError("abstract method")

    def load_config(self, name: str) -> tuple[Config, Dataset, nn.Module | None]:
        raise NotImplementedError("abstract method")

    def load_stats(self, stats: Stats) -> None:
        raise NotImplementedError("abstract method")

    def get_activations(self, layers: list[int], index: int) -> dict[int, Any]:
        raise NotImplementedError("abstract method")

    def get_histograms(self, layers: list[int]) -> dict[int, Any]:
        raise NotImplementedError("abstract method")


class FileLoader(Loader):
    """Util class to load model checkpoint from file"""

    def __init__(self, cfgdir: str, rundir: str, datadir: str, device: str = "cpu"):
        super().__init__(rundir, datadir, device)
        self.cfgdir = cfgdir
        self.checkpoint: dict[str, Any] = {}
        self.model: Model | None = None
        self.data: Dataset | None = None

    def get_models(self) -> list[str]:
        names = [path.basename(f).removesuffix(".toml") for f in glob(path.join(self.cfgdir, "*.toml"))]
        names.sort()
        return names

    def load_config(self, name: str) -> tuple[Config, Dataset, nn.Module | None]:
        file = path.join(self.cfgdir, name + ".toml")
        cfg = Config(file=file, rundir=self.rundir)
        self.data = cfg.dataset(self.datadir, "test", device=self.device)
        transform = cfg.transforms()
        self.model = Model(cfg, device=self.device)
        log.debug(f"== {name} model: ==\n{self.model}")
        file = path.join(cfg.dir, "model.pt")
        self.checkpoint = torch.load(file, map_location=self.device)
        log.info(f"loaded checkpoint from {file}")
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        return cfg, self.data, transform

    def load_stats(self, stats: Stats) -> None:
        stats.load_state_dict(self.checkpoint["stats_state_dict"])

    def get_activations(self, layers: list[int], index: int) -> dict[int, Any]:
        if not self.data or not self.model:
            return {}
        input = self.data.data[index]
        log.debug(f"get_activations: {layers} index={index}")
        return self.model.activations(input.view((1,)+input.size()), layers)

    def get_histograms(self, layers: list[int]) -> dict[int, Any]:
        if not self.data or not self.model:
            return {}
        input = self.data.data[:self.data.batch_size]
        log.debug(f"get_histogram: {layers}")
        return self.model.activations(input, layers, hist_bins=100)


class DBLoader(Loader):
    """Class to load checkpoint info from Redis DB"""

    def __init__(self, db: Database, sender: Callable, rundir: str, datadir: str, device: str = "cpu"):
        super().__init__(rundir, datadir, device)
        self.db = db
        self.send = sender

    def get_models(self) -> list[str]:
        return self.db.get_models()

    def get_epochs(self) -> list[int]:
        return self.db.get_state().epochs

    def load_config(self, name: str) -> tuple[Config, Dataset, nn.Module | None]:
        cfg = Config(name=name, data=self.db.get_config(name), rundir=self.rundir)
        test_data = cfg.dataset(self.datadir, "test", device=self.device)
        transform = cfg.transforms()
        return cfg, test_data, transform

    def load_stats(self, stats: Stats) -> None:
        stats.load_state_dict(self.db.load("ml:stats", device=self.device))

    def get_activations(self, layers: list[int], index: int) -> dict[int, Tensor]:
        missing = self.db.check_exists("ml:activations", [f"{i}:{index}" for i in layers])
        if missing:
            self.send("activations", missing, index)
        res = {}
        for layer in layers:
            res[layer] = self.db.load("ml:activations", f"{layer}:{index}", device=self.device)
        return res

    def get_histograms(self, layers: list[int]) -> dict[int, tuple[Tensor, Tensor]]:
        missing = self.db.check_exists("ml:histograms", [f"{i}" for i in layers])
        if missing:
            self.send("histograms", missing)
        res = {}
        for layer in layers:
            res[layer] = self.db.load("ml:histograms", f"{layer}", device=self.device)
        return res
