import logging as log
import math
import os
import shutil
import sys
from glob import glob
from os import path
from typing import Any

import kornia as K  # type: ignore
import tomli
import torch
import torchvision  # type: ignore
import torchvision.transforms.functional as F  # type: ignore
from torch import Tensor, nn

from .dataset import Dataset
from .scheduler import StepLRandWeightDecay
from .utils import pformat


class Config():
    """Config object has results of config data parsed from the toml file.

    Args:
        file:str    config file in toml format
        rundir:str  root directory to save run state
        epochs:int  number of epochs to train - if non-zero overrides the config

    Attributes:
        name:str    basename of config
        version:str config version string
        half:bool   set if to use half precision
        dir:str     run directory path
        cfg:dict    parsed config data
        data:dict   dataset config
        train:dict  training config
    """

    def __init__(self, file: str, rundir: str, epochs: int = 0):
        try:
            with open(file, mode="rb") as f:
                self.cfg = tomli.load(f)
        except FileNotFoundError:
            print(f"Error: config file '{file}' not found")
            sys.exit(1)
        except tomli.TOMLDecodeError as err:
            print(f"Error: config file '{file}': {err}")
            sys.exit(1)

        self.file: str = file
        self.name: str = os.path.basename(file).removesuffix(".toml")
        self.version = str(self.cfg.get("version", "1"))
        self.half = self.cfg.get("half", False)
        self.dir: str = path.join(rundir, self.name, self.version)
        self.train = self.cfg.get("train", {})
        if epochs != 0:
            self.train["epochs"] = epochs

    def save(self, clear: bool = False) -> None:
        """Save a copy of the config file to the run directory and optionally clear any data from prior runs."""
        log.info(f"saving config to {self.dir} clear={clear}")
        if clear:
            for file in glob(path.join(self.dir, "*.pt")):
                os.remove(file)
        if not path.exists(self.dir):
            os.makedirs(self.dir)
        shutil.copy(self.file, path.join(self.dir, "config.yaml"))

    def _data(self, typ):
        if typ == "train":
            cfg = self.cfg["train_data"]
        elif typ == "test":
            cfg = self.cfg["test_data"]
        elif typ == "valid":
            cfg = self.cfg.get("valid_data", {})
        else:
            raise ValueError(f"invalid dataset type: {typ}")
        return cfg

    def dataset(self,
                root: str,
                typ: str = "test",
                device: str = "cpu",
                dtype: torch.dtype = torch.float32
                ) -> Dataset | None:
        """Load a new train, test or valid dataset and apply normalization if defined."""
        cfg = self._data(typ)
        if len(cfg) == 0:
            return None
        train = cfg.get("train", False)
        batch_size = cfg.get("batch_size", 0)
        start = cfg.get("start", 0)
        end = cfg.get("end", 0)
        ds = Dataset(cfg["dataset"], root, train=train, batch_size=batch_size,
                     device=device, dtype=dtype, start=start, end=end)
        log.info(f"== {typ} data: ==\n{ds}")
        try:
            args = cfg["normalize"]
            log.info(f"normalize: {args}")
            ds.data = F.normalize(ds.data, args[0], args[1], inplace=True)
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
            transform.append(get_module(K.augmentation, args))
        log.info(f"== transforms: ==\n{transform}")
        return transform

    def model(self, device: str = "cpu", input: torch.Tensor | None = None, init_weights: bool = True) -> nn.Module:
        """Create a new model instance based on the config [model] layers definition."""
        model = nn.Sequential()
        cfg = self.cfg["model"]
        for args in cfg["layers"]:
            defn = cfg.get(args[0])
            argv = args.copy()
            if defn is not None:
                vars = argv.pop() if len(argv) > 1 else {}
                log.debug(f"get_definition: {defn} {argv[0]} {vars}")
                for layer in get_definition(torch.nn, defn, vars):
                    model.append(layer.to(device))
            else:
                layer = get_module(torch.nn, args)
                model.append(layer.to(device))
        if input is not None:
            check_model(model, input)
        log.info(f"== Model: ==\n{model}")
        if init_weights:
            wparams = self.cfg.get("weight_init")
            if wparams:
                init_layers(model, wparams)
            bparams = self.cfg.get("bias_init")
            if bparams:
                init_layers(model, bparams, mode="bias")
        return model

    def optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """Create a new optimizer based on config [train] optimizer setting"""
        typ, kwargs = self.train["optimizer"]
        optimizer = getattr(torch.optim, typ)
        return optimizer(model.parameters(), **kwargs)

    def scheduler(self, opt: torch.optim.Optimizer):
        try:
            typ, kwargs = self.train["scheduler"]
        except KeyError:
            return None
        if typ == "StepLRandWeightDecay":
            scheduler = StepLRandWeightDecay
        else:
            scheduler = getattr(torch.optim.lr_scheduler, typ)
        return scheduler(opt, **kwargs)

    def __str__(self):
        return pformat(self.cfg)


def init_layers(model: nn.Sequential, params: dict[str, list[Any]], mode: str = "weight") -> None:
    log.info(f"{mode} init: {pformat(params)}")
    for i, layer in enumerate(model):
        weights = getattr(layer, mode, None)
        if weights is not None:
            for typ, args in params.items():
                if getattr(torch.nn, typ) == type(layer):
                    if len(args) >= 2 and isinstance(args[-1], dict):
                        kwargs = args.pop().copy()
                    else:
                        kwargs = {}
                    log.debug(f"{i}: init {typ} {mode}: {args} {kwargs}")
                    getattr(torch.nn.init, args[0]+"_")(weights, *args[1:], **kwargs)


def check_model(model: nn.Sequential, input: Tensor):
    """Do a forward pass to check model structure and instantiate lazy layers"""
    x = input
    total_params = 0
    half_on = input.dtype == torch.float16
    log.info(f"half mode = {half_on}")
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=half_on):
        for i, layer in enumerate(model):
            try:
                x = layer(x)
            except RuntimeError as err:
                print(f"Error in layer {i+1}: {layer} definition\n  {err}")
                sys.exit(1)
            name = str(layer)
            name = name[:name.index("(")]
            params = 0
            for val in layer.state_dict().values():
                params += math.prod(val.size())
            if params > 0:
                log.info(f"layer {i:2}: {name:<12} {list(x.size())}  params={params}")
            else:
                log.info(f"layer {i:2}: {name:<12} {list(x.size())}")
            total_params += params
    log.info(f"total params = {total_params}")


def get_definition(pkg, defn, vars):
    layers = []
    for args in defn:
        layer = get_module(pkg, args, vars)
        layers.append(layer)
    return layers


def get_module(pkg, args, vars=None):
    args = list(args).copy()
    kwargs = {}
    if len(args) > 1 and isinstance(args[-1], dict):
        kwargs = args.pop().copy()
    for i, arg in enumerate(args[1:]):
        args[1+i] = getarg(arg, vars)
    for name, arg in kwargs.items():
        kwargs[name] = getarg(arg, vars)
    cname = args[0]
    if cname == "Linear" or cname == "Conv2d" or cname == "BatchNorm2d":
        cname = "Lazy" + cname
    try:
        fn = getattr(pkg, cname)(*args[1:], **kwargs)
    except AttributeError as err:
        print(f"Error: invalid func {args} {kwargs} - {err}")
        sys.exit(1)
    return fn


def getarg(arg, vars):
    if isinstance(arg, list) and len(arg) == 2:
        return (_arg(arg[0], vars), _arg(arg[1], vars))
    if isinstance(arg, list):
        return [_arg(i) for i in arg]
    return _arg(arg, vars)


def _arg(arg, vars):
    if vars and isinstance(arg, str) and arg.startswith("$"):
        try:
            return vars[arg[1:]]
        except KeyError:
            print(f"Error: Variable not defined in config: {arg}")
            sys.exit(1)
    return arg
