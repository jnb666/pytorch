#!/usr/bin/env python
# simple plot gui which reads latest data from local disk
import argparse
import logging as log
import sys
from os import path

import ml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="./data", help="data directory root")
    parser.add_argument("--rundir", default="./runs", help="saved run directory root")
    parser.add_argument("--cpu", action="store_true", default=False, help="disable CUDA for evaluation")
    parser.add_argument("--debug", action="store_true", default=False, help="debug printing")
    parser.add_argument("config")
    args = parser.parse_args()

    ml.init_logger(debug=args.debug)
    device = ml.get_device(args.cpu)

    loader = ml.FileLoader(
        cfgdir=path.dirname(args.config),
        rundir=args.rundir,
        datadir=args.datadir,
        device=device
    )
    name = path.basename(args.config).removesuffix(".toml")

    app = ml.init_gui()
    win = ml.MainWindow(loader, model=name)
    win.update_config(name, running=False)
    win.update_stats()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
