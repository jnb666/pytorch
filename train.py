#!/usr/bin/env python
import argparse
import logging as log
import multiprocessing as mp
import queue
import sys
import time
from multiprocessing import Process, Queue

import ml
import torch
from PySide6.QtCore import QTimer
from torch import nn


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="./data", help="data directory root")
    parser.add_argument("--rundir", default="./runs", help="saved run directory root")
    parser.add_argument("--cpu", action="store_true", default=False, help="disable CUDA training")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--epochs", type=int, default=0, help="number of epochs to train")
    parser.add_argument("--resume", type=int, default=0, help="resume training at given epoch")
    parser.add_argument("--gui", action="store_true", default=False, help="display stats etc. in GUI")
    parser.add_argument("--clear", action="store_true", default=False, help="clear data from prior runs")
    parser.add_argument("--debug", action="store_true", default=False, help="debug printing")
    parser.add_argument("config")
    return parser.parse_args()


def should_quit(cmd, paused):
    while True:
        if paused:
            msg = cmd.get()
            if msg == "quit":
                log.info("quit training run")
                return True
            if msg == "start":
                log.info("resume training run")
                return False
        else:
            try:
                msg = cmd.get(block=False)
                if msg == "quit":
                    log.info("quit training run")
                    return True
                if msg == "stop":
                    log.info("paused training run")
                    paused = True
            except queue.Empty:
                if not paused:
                    return False


def run_gui(cmd, q, cfg, args, timeout=100):
    app = ml.init_gui()
    test_data = cfg.dataset(args.datadir, "test")
    win = ml.MainWindow(cfg, cfg.model(), test_data, cfg.transforms(), cmd)

    def update():
        try:
            epoch = q.get(block=False)
            win.update_stats(epoch)
        except queue.Empty:
            pass

    log.info("start GUI")
    timer = QTimer(win)
    timer.timeout.connect(update)
    timer.start(timeout)

    win.show()
    sys.exit(app.exec())
    log.info("exit GUI")


def run(args, cfg, cmd, q):
    device = ml.get_device(args.cpu, args.seed)
    log.debug(cfg)
    cfg.save(clear=args.clear)

    dtype = torch.float16 if cfg.half else torch.float32
    train_data = cfg.dataset(args.datadir, "train", device=device, dtype=dtype)
    test_data = cfg.dataset(args.datadir, "test", device=device, dtype=dtype)
    valid_data = cfg.dataset(args.datadir, "valid", device=device, dtype=dtype)

    model = cfg.model(device=device, input=train_data[0][0])
    if torch.__version__.startswith("2"):
        log.info("compiling model for torch v2")
        model = torch.compile(model)

    transform = cfg.transforms()
    trainer = ml.Trainer(cfg, model)

    if args.resume > 0:
        stats = trainer.resume_from(args.resume, device)
        log.info(f"  Elapsed time = {stats.elapsed_total()}")
    else:
        stats = ml.Stats()
        stats.xrange = [1, trainer.epochs+1]

    if args.gui and should_quit(cmd, True):
        return

    while not trainer.should_stop(stats):
        if args.gui and should_quit(cmd, False):
            return

        stats.current_epoch += 1
        train_loss = trainer.train(train_data, transform)
        if valid_data is not None:
            valid_loss, valid_accuracy = trainer.test(valid_data)
        else:
            valid_loss, valid_accuracy = None, None
        if trainer.stopper:
            valid_loss_avg = trainer.stopper.average if stats.current_epoch > 1 else valid_loss
        else:
            valid_loss_avg = None
        test_loss, test_accuracy = trainer.test(test_data)

        if stats.current_epoch % trainer.log_every == 0:
            stats.update(trainer.predict, train_loss, test_loss, test_accuracy,
                         valid_loss, valid_accuracy, valid_loss_avg)
            trainer.save(stats)
            info = str(stats)
            if trainer.stopper:
                info += f"  Avg: {valid_loss_avg:6.3}"
            log.info(info)
            if args.gui:
                q.put(stats.current_epoch)
        if stats.current_epoch % (10*trainer.log_every) == 0:
            log.info(f"  Elapsed time = {stats.elapsed_total()}")

    log.info(f"  Elapsed time = {stats.elapsed_total()}")
    if args.gui:
        q.put(-1)


def main():
    mp.set_start_method("spawn")
    args = getargs()
    cfg = ml.Config(args.config, args.rundir, epochs=args.epochs)
    ml.init_logger(debug=args.debug, logdir=cfg.dir)

    cmd = Queue()
    q = Queue()
    if args.gui:
        p = Process(target=run_gui, args=(cmd, q, cfg, args))
        p.start()

    try:
        run(args, cfg, cmd, q)
    except KeyboardInterrupt:
        if args.gui:
            p.kill()
        sys.exit(0)

    if args.gui:
        p.join()


if __name__ == "__main__":
    main()
