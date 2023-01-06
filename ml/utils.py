import logging as log
import os
import pprint
import random
import time
from os import path
from typing import Any

import numpy as np
import torch

torch.set_printoptions(precision=3, threshold=1000, edgeitems=4, linewidth=180)
np.set_printoptions(precision=5, suppress=True, threshold=1000, edgeitems=4, linewidth=160)
pp = pprint.PrettyPrinter(indent=2)


class InvalidConfigError(Exception):
    pass


class DatasetNotFoundError(Exception):
    pass


class RunInterrupted(Exception):
    pass


def init_logger(debug: bool = False, with_timestamp=True) -> None:
    """Initialise logger instance and optionally enable debug messages or log file with millisecond timestamps."""
    level = log.DEBUG if debug else log.INFO
    if with_timestamp:
        log.basicConfig(format="%(asctime)s.%(msecs)03d  %(message)s", datefmt="%H:%M:%S", level=level)
    else:
        log.basicConfig(format="%(message)s", level=level)


def set_logdir(dir: str) -> None:
    """Add logging to directory"""
    logger = log.getLogger("")
    if not path.exists(dir):
        os.makedirs(dir)
    file = path.join(dir, time.strftime("%Y%m%d_%H%M%S") + ".log")
    handler = log.FileHandler(file, encoding="utf-8")
    handler.setLevel(logger.getEffectiveLevel())
    handler.setFormatter(log.Formatter("%(asctime)s.%(msecs)03d  %(message)s", datefmt="%H:%M:%S"))
    for i in range(1, len(logger.handlers)):
        logger.removeHandler(logger.handlers[i])
    log.info(f"writing log to {file}")
    logger.addHandler(handler)


def pformat(data) -> str:
    """Pretty print data with default options. """
    return pp.pformat(data)


def get_device(cpu: bool) -> str:
    """Get cuda or cpu device"""
    if not cpu and torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = "cpu"
    log.info(f"== pytorch {torch.__version__}  device={device} ==")
    return device


def set_seed(seed: int) -> None:
    """Initialise pytorch, numpy and python random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    log.info(f'Set random seed={seed}')


def get_definition(pkg, defn, vars, expand=None):
    layers = []
    for args in defn:
        log.debug(f"add layer: {args} vars={vars}")
        layer = get_module(pkg, args, vars, expand, desc="model")
        layers.append(layer)
    return layers


def get_module(pkg, args, vars=None, expand=None, desc=""):
    args = list(args).copy()
    if expand:
        args[0] = expand(args[0])
    kwargs = {}
    if len(args) > 1 and isinstance(args[-1], dict):
        kwargs = args.pop().copy()
    for i, arg in enumerate(args[1:]):
        args[1+i] = getarg(arg, vars)
    for name, arg in kwargs.items():
        kwargs[name] = getarg(arg, vars)
    try:
        fn = getattr(pkg, args[0])(*args[1:], **kwargs)
    except (AttributeError, TypeError) as err:
        raise InvalidConfigError(f"{desc}: {err}")
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
            raise InvalidConfigError(f"{arg} not defined")
    return arg
