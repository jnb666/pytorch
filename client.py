#!/usr/bin/env python
import argparse
import logging as log
import sys
import time
from os import path
from typing import Callable

import ml
import redis
import zmq
from ml.gui import MainWindow, init_gui
from PySide6.QtCore import QTimer

timeout = 20


def subscribe(win: MainWindow, db: ml.Database, server: str) -> None:
    r = ml.Database(server, notify=True)
    pubsub = r.db.pubsub()
    log.info("subscribe for redis notifications")
    pubsub.psubscribe("__keyspace@0__:ml:*")
    state = ml.State()

    def get_message():
        msg = pubsub.get_message()
        if msg and msg["channel"].endswith(b"ml:state"):
            s = db.get_state()
            # log.debug(f"notify: {s}")
            if s.error:
                log.info(f"server error: {s.error}")
                win.set_error(s.error)
            elif (s.error != state.error or s.name != state.name or s.version != state.version or
                  s.models != state.models or s.checksum != state.checksum):
                log.info(f"update config: {s.name} version={s.version} running={s.running}")
                win.update_config(s.name, s.version)
            elif (s.epoch != state.epoch or s.running != state.running or s.max_epoch != state.max_epoch
                  or (not s.running and s.epochs != state.epochs)):
                log.info(f"update stats: {s.name} epoch={s.epoch}/{s.max_epoch} running={s.running}")
                win.update_stats()
            state.update(s)

    timer = QTimer(win)
    timer.timeout.connect(get_message)  # type: ignore
    timer.start(timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="localhost", help="server for 0MQ and Redis")
    parser.add_argument("--datadir", default="./data", help="data directory root")
    parser.add_argument("--rundir", default="./runs", help="saved run directory root")
    parser.add_argument("--debug", action="store_true", default=False, help="debug printing")
    parser.add_argument("--force", action="store_true", default=False, help="force load of current model even if running")
    args = parser.parse_args()

    ml.init_logger(debug=args.debug)
    device = ml.get_device("cpu")
    db = ml.Database(args.server)
    state = db.get_state()
    if state.name == "":
        state.name = "mnist_mlp"
    if state.version == "":
        state.version = "1"
    client = ml.Client(args.server)

    loader = ml.DBLoader(
        db=db,
        sender=client.send,
        rundir=args.rundir,
        datadir=args.datadir,
        device=device
    )

    app = init_gui()
    win = MainWindow(loader, client.send)

    subscribe(win, db, args.server)
    log.info(f"load: {state.name} version={state.version}")
    if args.force or not state.running:
        status, err = client.send("load", state.name, state.version)
    else:
        win.update_config(state.name, state.version)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
