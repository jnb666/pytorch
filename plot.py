#!/usr/bin/env python
import argparse
import logging as log
import sys
import time

import ml
import numpy as np
import torch
from torch import nn


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="./data", help="data directory root")
    parser.add_argument("--rundir", default="./runs", help="saved run directory root")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--version", default="1", help="config version (default 1)")
    parser.add_argument("--debug", action="store_true", default=False, help="debug printing")
    parser.add_argument("config")
    return parser.parse_args()


def main():
    args = getargs()
    ml.init_logger(debug=args.debug)

    cfg = ml.Config(args.config, args.rundir)
    log.debug(cfg)

    test_data = cfg.dataset(args.datadir, "test")
    transform = cfg.transforms()
    model = cfg.model()

    app = ml.init_gui()
    win = ml.MainWindow(cfg, model, test_data, transform)
    win.update_stats()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
