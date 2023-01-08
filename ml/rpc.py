import logging as log
import os
import sys
import time
from argparse import Namespace
from glob import glob
from os import path
from typing import Callable

import torch
import zmq
from torch import nn

from .config import Config, Index
from .database import Database, State
from .model import Model
from .trainer import Datasets, Stats, Trainer
from .utils import InvalidConfigError, RunInterrupted, set_logdir, set_seed

zmq_port = 5555


class BaseContext:
    """Context class is used to manage a training run"""

    def __init__(self, args: Namespace, device: str):
        self.args = args
        self.device = device
        self.trainer: Trainer | None = None
        self.ds: Datasets | None = None
        self.cfg: Config | None = None
        self.stats = Stats()
        self.epochs: list[int] = []

    @property
    def epoch(self) -> int:
        return self.stats.current_epoch

    @property
    def log_every(self) -> int:
        if self.trainer:
            return self.trainer.log_every
        return 0

    def load_config(self, cfg: Config) -> None:
        set_logdir(cfg.dir)
        # log.debug(cfg)
        device = "cuda" if self.device == "cuda" else "cpu"
        dtype = torch.float16 if cfg.half else torch.float32
        self.cfg = cfg
        self.ds = Datasets(cfg, self.args.datadir, device=device, dtype=dtype)

    def restart(self, clear_files: bool = False) -> bool:
        if not self.cfg or not self.ds:
            return False
        self.cfg.save(clear=clear_files)
        set_seed(self.cfg.seed)
        model = Model(self.cfg, self.ds.train_data.image_shape, device=self.device, init_weights=True)
        log.info(model)
        self.trainer = Trainer(self.cfg, model, device=self.device)
        self.stats = Stats()
        self.stats.xrange = [1, self.trainer.epochs]
        return True

    def resume(self, epoch: int) -> bool:
        if not self.cfg or not self.ds:
            return False
        self.cfg.save()
        model = Model(self.cfg, self.ds.train_data.image_shape, device=self.device)
        log.info(model)
        self.trainer = Trainer(self.cfg, model, device=self.device)
        self.stats = self.trainer.resume_from(epoch)
        if self.stats.current_epoch != epoch:
            return False
        if self.trainer.stopper:
            self.trainer.stopper.update_stats(self.stats)
        return True

    def do_epoch(self, should_stop: Callable[[], bool] | None = None) -> bool:
        """Returns true if run should stop"""
        if not self.trainer or not self.ds:
            log.error("Error: cannot step to next epoch - run not started")
            return True

        try:
            train_loss = self.trainer.train(self.ds.train_data, self.ds.transform, should_stop=should_stop)
            if self.ds.valid_data is not None:
                valid_loss, valid_accuracy = self.trainer.test(self.ds.valid_data, should_stop=should_stop)
            else:
                valid_loss, valid_accuracy = None, None
            test_loss, test_accuracy = self.trainer.test(self.ds.test_data, should_stop=should_stop)
        except RunInterrupted:
            return True

        self.stats.current_epoch += 1
        self.stats.predict = self.trainer.predict
        self.stats.update(self.trainer.learning_rate(), train_loss, test_loss,
                          test_accuracy, valid_loss, valid_accuracy)
        stop = self.trainer.should_stop(self.stats)
        self.stats.running = not stop

        self.trainer.step(self.stats)

        if stop or (self.epoch % self.log_every == 0):
            self.trainer.save(self.stats)
            log.info(str(self.stats))

        if stop or (self.epoch % (10*self.log_every) == 0):
            log.info(f"  Elapsed time = {self.stats.elapsed_total()}")

        if stop:
            self.epochs = clear_checkpoints(self.epoch, self.trainer.cfg.dir)
        return stop

    def run(self) -> None:
        raise NotImplementedError("abstract method")


class CmdContext(BaseContext):
    """CmdContext class is used to manage a command line training run"""

    def run(self) -> None:
        try:
            cfg = Config(file=self.args.config, rundir=self.args.rundir, epochs=self.args.epochs, seed=self.args.seed)
            self.load_config(cfg)
        except InvalidConfigError as err:
            print(f"Error: config file '{self.args.config}': {err}")
            sys.exit(1)
        try:
            if self.args.resume:
                self.resume(self.args.resume)
            else:
                self.restart(clear_files=self.args.clear)
        except FileNotFoundError as err:
            print(f"Resume file not found: {err}")
            sys.exit(1)

        while not self.do_epoch():
            pass


class Server(BaseContext):
    """Server class is used to manage a command line training run using 0MQ RPC"""

    def __init__(self, args: Namespace, device: str, db: Database):
        super().__init__(args, device)
        self.db = db
        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{zmq_port}")
        log.info(f"listening for commands on port {zmq_port}")
        self.paused = True
        self.started = False

    def run(self) -> None:
        while True:
            self.get_command()
            if self.started and not self.paused:
                if self.do_epoch(should_stop=self.get_command):
                    self.stats.running = False
                    self.paused = True
                self.db.update(self.stats, clear=self.stats.running, epochs=self.epochs)

    def load(self, name: str) -> None:
        log.info(f"cmd: load {name}")
        try:
            data = self.db.get_config(name)
            cfg = Config(name=name, data=data, rundir=self.args.rundir, seed=self.args.seed)
            self.load_config(cfg)
            self.stats = self.db.set_model(name, cfg.dir, cfg.epochs)
            self.ok()
        except InvalidConfigError as err:
            state = State(name=name, error=str(err))
            self.db.set_model_state(state)
            self.error(str(err))
        self.started = False
        self.paused = True

    def start(self, epoch: int, clear: bool) -> None:
        log.info(f"cmd: start {epoch} {clear}")
        try:
            if epoch == 0:
                self.started = self.restart(clear)
            else:
                self.started = self.resume(epoch)
        except (FileNotFoundError, InvalidConfigError) as err:
            self.error(str(err))
            return
        if self.started:
            self.stats.running = True
            self.db.update(self.stats, clear=True)
            self.ok()
            self.paused = False
        else:
            self.error("config not loaded")

    def set_max_epoch(self, epoch: int) -> None:
        log.info(f"cmd: max_epoch {epoch}")
        if self.cfg:
            self.cfg.epochs = epoch
            self.stats.xrange = [0, epoch]
            self.db.update(self.stats)
            self.ok()
        else:
            self.error("config not loaded")

    def pause(self, on: bool) -> None:
        if on or self.started:
            log.info(f"cmd: pause {on}")
            self.paused = on
            if self.started and self.trainer and self.trainer.stopper:
                self.trainer.stopper.update_stats(self.stats)
            self.db.set_running(not on)
            self.ok()
        else:
            state = self.db.get_state()
            self.start(state.epoch, False)

    def get_model(self) -> Model | None:
        if self.started and self.trainer:
            return self.trainer.model
        if not self.cfg or not self.ds or self.stats.current_epoch < 1:
            return None
        model = Model(self.cfg, self.ds.train_data.image_shape, device=self.device)
        trainer = Trainer(self.cfg, model, device=self.device)
        stats = trainer.resume_from(self.stats.current_epoch)
        if stats.current_epoch == self.stats.current_epoch:
            return model
        return None

    def get_activations(self, name: str, layers: list[Index], index: int = 0, hist_bins: int = 0) -> None:
        log.debug(f"cmd: {name} {layers} {index} hist_bins={hist_bins}")
        model = self.get_model()
        if not self.ds or not model:
            self.error("config not loaded")
            return
        test_data = self.ds.test_data
        if name == "ml:activations":
            input = test_data.data[index:index+1]
        else:
            input = test_data.data[:test_data.batch_size]
        layers = [Index(i) for i in layers]
        activations = model.activations(input, layers, hist_bins=hist_bins)
        for layer, val in activations.items():
            key = str(layer)
            if name == "ml:activations":
                key += f":{index}"
            log.debug(f"save {name} {key}")
            self.db.save(val, name, key)
        self.ok()

    def get_command(self) -> bool:
        """Poll for command on 0MQ socket - returns True if run should stop"""
        if self.paused or not self.started:
            cmd = self.socket.recv_json()
        else:
            try:
                cmd = self.socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                return not self.started or self.paused
        match cmd:
            case ["load", str(name)]:
                self.load(name)
            case ["start", int(epoch), bool(clear)]:
                self.start(epoch, clear)
            case ["max_epoch", int(epoch)]:
                self.set_max_epoch(epoch)
            case ["pause", bool(on)]:
                self.pause(on)
            case ["activations", list(layers), int(index)]:
                self.get_activations("ml:activations", layers, index)
            case ["histograms", list(layers)]:
                self.get_activations("ml:histograms", layers, hist_bins=100)
            case _:
                self.error(f"invalid command: {cmd}")
        return not self.started or self.paused

    def ok(self) -> None:
        self.socket.send_json(["ok", ""])

    def error(self, msg: str) -> None:
        log.error(f"error: {msg}")
        self.socket.send_json(["error", msg])


class Client:
    """0MQ client class to send commands"""

    def __init__(self, host: str):
        ctx = zmq.Context()
        log.info(f"zmq: connect to server at {host}:{zmq_port}")
        self.socket = ctx.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{zmq_port}")

    def send(self, *cmd) -> tuple[str, str]:
        """send command to server - returns ("ok"|"error", "errmsg") """
        start = time.time()
        self.socket.send_json(cmd)
        status = self.socket.recv_json()
        elapsed = time.time() - start
        if not isinstance(status, list) or len(status) != 2:
            raise RuntimeError(f"send: malformed response: {status}")
        if status[0] != "ok":
            log.error(f"{cmd[0]} error: {status[1]}")
        cmd_list = []
        for arg in cmd:
            if isinstance(arg, list):
                cmd_list.append(" ".join([str(i) for i in arg]))
            else:
                cmd_list.append(arg)
        log.info(f"sent cmd: {cmd_list} {elapsed:.3f}s")
        return status[0], status[1]


def clear_checkpoints(last_epoch: int, dir: str) -> list[int]:
    """Remove old checkpoint files from previous runs and return current epoch list"""
    epochs = []
    for file in glob(path.join(dir, "model_*.pt")):
        epoch = int(path.basename(file)[6:-3])
        if epoch > last_epoch:
            log.debug(f"remove: {file}")
            os.remove(file)
        else:
            epochs.append(epoch)
    epochs.sort()
    return epochs
