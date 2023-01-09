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


def getargs(argv, vars=None):
    typ, args, kwargs = splitargs(argv)
    for i, arg in enumerate(args):
        args[i] = getarg(arg, vars)
    for name, arg in kwargs.items():
        kwargs[name] = getarg(arg, vars)
    return typ, args, kwargs


def get_module(desc, pkg, typ, args, kwargs):
    try:
        obj = getattr(pkg, typ)(*args, **kwargs)
    except (AttributeError, TypeError) as err:
        raise InvalidConfigError(f"{desc}: {err}")
    return obj


def splitargs(argv):
    if not isinstance(argv, list) or len(argv) == 0 or not isinstance(argv[0], str):
        raise InvalidConfigError(f"invalid arg list: {argv}")
    args = argv.copy()
    if isinstance(args[-1], dict):
        kwargs = args.pop().copy()
    else:
        kwargs = {}
    return args[0], args[1:], kwargs


def format_args(args, kwargs):
    return "(" + ", ".join([str(x) for x in args] + [f"{k}={v}" for k, v in kwargs.items()]) + ")"


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
