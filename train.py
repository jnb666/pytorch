#!/usr/bin/env python
# do command line training run
import argparse
import sys

import ml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="./data", help="data directory root")
    parser.add_argument("--rundir", default="./runs", help="saved run directory root")
    parser.add_argument("--cpu", action="store_true", default=False, help="disable CUDA training")
    parser.add_argument("--seed", type=int, default=0, help="override random seed in config if set")
    parser.add_argument("--epochs", type=int, default=0, help="number of epochs to train")
    parser.add_argument("--resume", type=int, default=0, help="resume training at given epoch")
    parser.add_argument("--clear", action="store_true", default=False, help="clear data from prior runs")
    parser.add_argument("--debug", action="store_true", default=False, help="debug printing")
    parser.add_argument("config")
    args = parser.parse_args()

    ml.init_logger(debug=args.debug)
    device = ml.get_device(args.cpu)

    ctx = ml.CmdContext(args, device)
    try:
        ctx.run()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
