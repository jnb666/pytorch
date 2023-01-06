#!/usr/bin/env python
# run training server which listens for requests on 0MQ port
import argparse
import logging as log
import multiprocessing as mp
import sys
from os import path

import ml
from inotify_simple import INotify, flags  # type: ignore


def watch_config(dir: str):
    db = ml.Database("localhost")
    client = ml.Client("localhost")

    inotify = INotify()
    wd = inotify.add_watch(dir, flags.CREATE | flags.DELETE | flags.MODIFY | flags.DELETE_SELF |
                           flags.MOVED_TO | flags.MOVED_FROM | flags.MOVE_SELF)
    state = db.get_state()
    while True:
        for event in inotify.read():
            log.debug(f"inotify: {event.name}")
            s = db.update_config(path.join(dir, event.name))
            # reload config if changed on disk and run is not in progress
            if s.name == state.name and not s.running and s.checksum != state.checksum:
                status, err = client.send("load", s.name)
                if status == "error":
                    s.error = err
            state.update(s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="./data", help="data directory root")
    parser.add_argument("--rundir", default="./runs", help="saved run directory root")
    parser.add_argument("--configdir", default="./cfg", help="directory for config files")
    parser.add_argument("--cpu", action="store_true", default=False, help="disable CUDA training")
    parser.add_argument("--seed", type=int, default=0, help="override random seed in config if set")
    parser.add_argument("--debug", action="store_true", default=False, help="debug printing")
    args = parser.parse_args()

    ml.init_logger(debug=args.debug)
    device = ml.get_device(args.cpu)

    db = ml.Database("localhost")
    db.init_config(args.configdir)
    p = mp.Process(target=watch_config, args=(args.configdir,))
    p.start()

    ctx = ml.Server(args, device, db)
    try:
        ctx.run()
    except KeyboardInterrupt:
        p.terminate()
        sys.exit(0)

    p.terminate()


if __name__ == "__main__":
    main()
