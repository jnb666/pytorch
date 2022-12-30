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
np.set_printoptions(precision=3, suppress=True, threshold=1000, edgeitems=4, linewidth=160)
pp = pprint.PrettyPrinter(indent=2)


def init_logger(debug: bool = False, logdir: str = "") -> None:
    """Initialise logger instance and optionally enable debug messages or log file with millisecond timestamps."""
    level = log.DEBUG if debug else log.INFO
    log.basicConfig(format="%(message)s", level=level)
    if not logdir:
        return
    if not path.exists(logdir):
        os.makedirs(logdir)
    file = path.join(logdir, time.strftime("%Y%m%d_%H%M%S") + ".log")
    log.info(f"writing log to {file}")
    handler = log.FileHandler(file, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(log.Formatter(
        "%(asctime)s.%(msecs)03d  %(message)s", datefmt="%H:%M:%S"))
    log.getLogger("").addHandler(handler)


def pformat(data) -> str:
    """Pretty print data with default options. """
    return pp.pformat(data)


def get_device(cpu: bool, seed: int) -> str:
    """Get cuda or cpu device and initialise random number seed"""
    log.info(f"== pytorch {torch.__version__} ==")
    if not cpu and torch.cuda.is_available():
        device = "cuda"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = "cpu"
    random.seed(seed)
    torch.manual_seed(seed)
    log.info(f'Using {device} device  seed={seed}')
    return device


def load_checkpoint(dir: str, epoch: int | None, device: str = "cpu") -> dict[str, Any]:
    """load checkpoint dict from given directory"""
    if epoch is None:
        file = path.join(dir, "model.pt")
    else:
        file = path.join(dir, f"model_{epoch}.pt")
    log.debug(f"load checkpoint from {file} map_location={device}")
    return torch.load(file, map_location=device)
