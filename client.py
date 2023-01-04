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
from PySide6.QtCore import QTimer

zmq_host = "poppy"
redis_host = "poppy"
timeout = 20


def subscribe(win: ml.MainWindow, db: ml.Database) -> None:
    r = ml.Database(redis_host, notify=True)
    pubsub = r.db.pubsub()
    log.info("subscribe for redis notifications")
    pubsub.psubscribe("__keyspace@0__:ml:*")
    state = ml.State("", [])

    def get_message():
        msg = pubsub.get_message()
        if msg and msg["channel"].endswith(b"ml:state"):
            s = db.get_state()
            log.debug(f"notify: {s}")
            if s.name != state.name or s.models != state.models or s.checksum != state.checksum:
                log.info(f"update config: {s.name} running={s.running}")
                win.update_config(s.name, s.running)
            if (s.name != state.name or s.epoch != state.epoch or s.running != state.running or
                    s.max_epoch != state.max_epoch or (not s.running and s.epochs != state.epochs)):
                log.info(f"update stats: {s.name} epoch={s.epoch}/{s.max_epoch} running={s.running}")
                win.update_stats()
            state.update(s)

    timer = QTimer(win)
    timer.timeout.connect(get_message)  # type: ignore
    timer.start(timeout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="./data", help="data directory root")
    parser.add_argument("--rundir", default="./runs", help="saved run directory root")
    parser.add_argument("--debug", action="store_true", default=False, help="debug printing")
    args = parser.parse_args()

    ml.init_logger(debug=args.debug)
    device = ml.get_device("cpu")
    db = ml.Database(redis_host)
    state = db.get_state()
    client = ml.Client(zmq_host)

    loader = ml.DBLoader(
        db=db,
        sender=client.send,
        rundir=args.rundir,
        datadir=args.datadir,
        device=device
    )

    app = ml.init_gui()
    win = ml.MainWindow(loader, client.send, model=state.name)

    subscribe(win, db)
    if not state.running:
        client.send("load", state.name)

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
