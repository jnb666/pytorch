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
zmq_port = 5555
redis_host = "poppy"
timeout = 20


def client() -> Callable:
    ctx = zmq.Context()
    log.info(f"zmq: connect to server at {zmq_host}:{zmq_port}")
    socket = ctx.socket(zmq.REQ)
    socket.connect(f"tcp://{zmq_host}:{zmq_port}")

    def send(*cmd) -> tuple[str, str]:
        start = time.time()
        socket.send_json(cmd)
        status = socket.recv_json()
        elapsed = time.time() - start
        if not isinstance(status, list) or len(status) != 2:
            raise RuntimeError(f"send: malformed response: {status}")
        if status[0] != "ok":
            log.error(f"{cmd[0]} error: {status[1]}")
            sys.exit(1)
        log.info(f"sent cmd: {cmd} {elapsed:.3f}s")
        return status[0], status[1]

    return send


def subscribe(win: ml.MainWindow, db: ml.Database) -> None:
    r = ml.Database(redis_host, notify=True)
    pubsub = r.db.pubsub()
    log.info("subscribe for redis notifications")
    pubsub.psubscribe("__keyspace@0__:ml:*")
    state = ml.State("", [])

    def get_message():
        msg = pubsub.get_message()
        if not msg:
            return
        key = msg["channel"].removeprefix(b"__keyspace@0__:")
        log.debug(f"pubsub: {msg['channel']} => {key}")
        if key == b"ml:state":
            s = db.get_state()
            if s == state:
                return
            state.__dict__.update(s.__dict__)
            log.info(f"{state.name} epoch={state.epoch}/{state.max_epoch} running={state.running}")
            win.update_stats()
        elif key == b"ml:config":
            log.info(f"update config {state.name} running={state.running}")
            win.update_config(state.name, state.running)

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
    sender = client()

    loader = ml.DBLoader(
        db=db,
        sender=sender,
        rundir=args.rundir,
        datadir=args.datadir,
        device=device
    )

    app = ml.init_gui()
    win = ml.MainWindow(loader, sender, model=state.name)
    subscribe(win, db)
    win.load(state.running)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
