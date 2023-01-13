#!/usr/bin/env python
# simple plot gui which reads latest data from local disk
import argparse
import logging as log
import sys
from os import path

import ml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgdir", default="./cfg", help="config file directory")
    parser.add_argument("--datadir", default="./data", help="data directory root")
    parser.add_argument("--rundir", default="./runs", help="saved run directory root")
    parser.add_argument("--cpu", action="store_true", default=False, help="disable CUDA for evaluation")
    parser.add_argument("--debug", action="store_true", default=False, help="debug printing")
    parser.add_argument("--version", default="", help="model version")
    parser.add_argument("config")
    args = parser.parse_args()

    ml.init_logger(debug=args.debug)
    device = ml.get_device(args.cpu)

    loader = ml.FileLoader(
        cfgdir=args.cfgdir,
        rundir=args.rundir,
        datadir=args.datadir,
        device=device
    )
    name = path.basename(args.config).removesuffix(".toml")

    app = ml.init_gui()
    win = ml.MainWindow(loader)
    win.update_config(name, args.version, running=False)
    win.update_stats()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
