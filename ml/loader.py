import logging as log
import sys
import time
from glob import glob
from os import path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn

from .config import Config, Index
from .dataset import Dataset, Transforms
from .model import Model
from .rpc import Database
from .trainer import Stats
from .utils import InvalidConfigError, load_checkpoint


class Loader:
    """Loader abstract base class"""

    def __init__(self, rundir: str, datadir: str, device: str = "cpu"):
        self.rundir = rundir
        self.datadir = datadir
        self.device = device
        self.name = ""
        self.epoch = -999

    def get_models(self) -> list[tuple[str, list[str]]]:
        raise NotImplementedError("abstract method")

    def get_epochs(self) -> list[int]:
        raise NotImplementedError("abstract method")

    def load_model(self, name: str, version: str) -> str:
        raise NotImplementedError("abstract method")

    def load_config(self, name: str, version: str = "") -> tuple[Config, Model, Dataset, Transforms | None]:
        raise NotImplementedError("abstract method")

    def load_stats(self, stats: Stats) -> None:
        raise NotImplementedError("abstract method")

    def get_activations(self, layers: list[Index], index: int) -> dict[Index, Any]:
        raise NotImplementedError("abstract method")

    def get_histograms(self, layers: list[Index]) -> dict[Index, Any]:
        raise NotImplementedError("abstract method")


class FileLoader(Loader):
    """Util class to load model checkpoint from file"""

    def __init__(self, cfgdir: str, rundir: str, datadir: str, device: str = "cpu"):
        super().__init__(rundir, datadir, device)
        self.cfgdir = cfgdir
        self.checkpoint: dict[str, Any] = {}
        self.model: Model | None = None
        self.data: Dataset | None = None

    def get_models(self) -> list[tuple[str, list[str]]]:
        log.debug("get_models")
        models = []
        names = [path.basename(f).removesuffix(".toml") for f in glob(path.join(self.cfgdir, "*.toml"))]
        for name in sorted(names):
            versions = [path.basename(dir) for dir in glob(path.join(self.rundir, name, "*"))]
            models.append((name, versions))
        return models

    def load_model(self, name: str, version: str) -> str:
        log.debug(f"load_model: {name} version={version}")
        cfg = load_config_file(name, version, self.rundir, self.cfgdir)
        return cfg.text

    def load_config(self, name: str, version: str = "") -> tuple[Config, Model, Dataset, Transforms | None]:
        log.debug(f"load_config: {name} version={version}")
        cfg = load_config_file(name, version, self.rundir, self.cfgdir)
        self.data = cfg.dataset(self.datadir, "test", device=self.device)
        transform = cfg.transforms()
        self.model = Model(cfg, self.data.image_shape, device=self.device)
        try:
            self.checkpoint = load_checkpoint(cfg.dir, device=self.device)
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
        except FileNotFoundError:
            self.model.init_weights()
        return cfg, self.model, self.data, transform

    def load_stats(self, stats: Stats) -> None:
        if self.checkpoint.get("stats_state_dict"):
            stats.load_state_dict(self.checkpoint["stats_state_dict"])
        else:
            stats.clear()

    def get_activations(self, layers: list[Index], index: int) -> dict[Index, Any]:
        if not self.data or not self.model:
            return {}
        input = self.data.data[index]
        log.debug(f"get_activations: {layers} index={index}")
        return self.model.activations(input.view((1,)+input.size()), layers)

    def get_histograms(self, layers: list[Index]) -> dict[Index, Any]:
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

    def get_models(self) -> list[tuple[str, list[str]]]:
        return self.db.get_models()

    def get_epochs(self) -> list[int]:
        return self.db.get_state().epochs

    def load_model(self, name: str, version: str) -> str:
        return self.db.get_config(name, version)

    def load_config(self, name: str, version: str = "") -> tuple[Config, Model, Dataset, Transforms | None]:
        log.debug(f"DBloader: load_config {name} version={version}")
        cfg = Config(name=name, data=self.db.get_config(name, version), rundir=self.rundir)
        test_data = cfg.dataset(self.datadir, "test", device=self.device)
        transform = cfg.transforms()
        model = Model(cfg, test_data.image_shape, device=self.device)
        return cfg, model, test_data, transform

    def load_stats(self, stats: Stats) -> None:
        stats.load_state_dict(self.db.load("ml:stats", device=self.device))

    def get_activations(self, layers: list[Index], index: int) -> dict[Index, Tensor]:
        missing = self.db.check_exists("ml:activations", [f"{i}:{index}" for i in layers])
        if missing:
            self.send("activations", missing, index)
        res = {}
        for layer in layers:
            res[layer] = self.db.load("ml:activations", str(layer)+":"+str(index), device=self.device)
        return res

    def get_histograms(self, layers: list[Index]) -> dict[Index, Any]:
        missing = self.db.check_exists("ml:histograms", [f"{i}" for i in layers])
        if missing:
            self.send("histograms", missing)
        res = {}
        for layer in layers:
            res[layer] = self.db.load("ml:histograms", str(layer), device=self.device)
        return res


def load_config_file(name, version, rundir, cfgdir) -> Config:
    if version:
        try:
            file = path.join(rundir, name, version, "config.toml")
            return Config(file=file, rundir=rundir)
        except FileNotFoundError as err:
            pass
    file = path.join(cfgdir, name + ".toml")
    return Config(file=file, rundir=rundir)
