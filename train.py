#!/usr/bin/env python
import argparse
import datetime
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

width = 1200
height = 900


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


def run_gui(q, cfg, args, timeout=0.05):
    app, win = ml.init_gui(width, height, f"{cfg.name} v{cfg.version}")
    test_data = cfg.dataset(args.datadir, "test")
    gui = ml.MainWindow(cfg, cfg.model(), test_data, cfg.transforms())
    win.setCentralWidget(gui)

    def update():
        try:
            epoch = q.get(block=False)
            gui.update_stats(epoch)
        except queue.Empty:
            pass

    log.info("start GUI")
    timer = QTimer(gui)
    timer.timeout.connect(update)
    timer.start(timeout)

    win.show()
    sys.exit(app.exec())
    log.info("exit GUI")


def run(args, cfg, q):
    device = ml.get_device(args.cpu, args.seed)
    log.debug(cfg)
    cfg.save(clear=args.clear)

    dtype = torch.float16 if cfg.half else torch.float32
    train_data = cfg.dataset(args.datadir, "train", device=device, dtype=dtype)
    test_data = cfg.dataset(args.datadir, "test", device=device, dtype=dtype)
    valid_data = cfg.dataset(args.datadir, "valid", device=device, dtype=dtype)

    model = cfg.model(device=device, input=train_data[0][0])

    transform = cfg.transforms()
    trainer = ml.Trainer(cfg, model)

    if args.resume > 0:
        stats = trainer.resume_from(args.resume, device)
    else:
        stats = ml.Stats()
        stats.xrange = [1, trainer.epochs+1]

    while not trainer.should_stop(stats):
        start_epoch = time.time()
        stats.current_epoch += 1
        train_loss = trainer.train(train_data, transform)
        if valid_data is not None:
            valid_loss, valid_accuracy = trainer.test(valid_data)
        test_loss, test_accuracy = trainer.test(test_data)
        epoch_time = time.time()-start_epoch

        if stats.current_epoch % trainer.log_every == 0:
            if valid_data is None:
                stats.update(trainer.predict, train_loss, test_loss, test_accuracy)
            else:
                stats.update(trainer.predict, train_loss, test_loss, test_accuracy, valid_loss, valid_accuracy)
            trainer.save(stats)
            info = str(stats)
            if trainer.scheduler:
                lr, = trainer.scheduler.get_last_lr()
                info += f"  Learn rate: {lr:.4}"
            log.info(f"{info}  {epoch_time:.1f}s")
            if args.gui:
                q.put(stats.current_epoch)

    elapsed = datetime.timedelta(seconds=round(stats.elapsed))
    log.info(f"Elapsed time = {elapsed}")


def main():
    mp.set_start_method("spawn")
    args = getargs()
    cfg = ml.Config(args.config, args.rundir, epochs=args.epochs)
    ml.init_logger(debug=args.debug, logdir=cfg.dir)

    q = Queue()
    if args.gui:
        p = Process(target=run_gui, args=(q, cfg, args))
        p.start()

    try:
        run(args, cfg, q)
    except KeyboardInterrupt:
        if args.gui:
            p.kill()
        sys.exit(0)

    if args.gui:
        p.join()


if __name__ == "__main__":
    main()
