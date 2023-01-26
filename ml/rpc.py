import logging as log
import os
import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from glob import glob
from os import path
from typing import Callable

import torch
import zmq
from torch import nn

from .config import Config, Index
from .database import Database, State
from .dataset import Dataset, Loader, Transforms
from .model import Model, load_model
from .trainer import Stats, Trainer
from .utils import (InvalidConfigError, RunInterrupted, load_checkpoint,
                    set_logdir, set_seed)

zmq_port = 5555


@dataclass
class Data:
    cfg: Config
    model: nn.Module
    train: Dataset
    test: Dataset
    valid: Dataset | None
    train_loader: Loader
    test_loader: Loader
    valid_loader: Loader | None

    def start(self, max_items: int = 0, max_test_items: int = 0):
        self.train_loader.start(self.train, max_items)
        self.test_loader.start(self.test, max_test_items)
        if self.valid is not None and self.valid_loader is not None:
            self.valid_loader.start(self.valid, max_test_items)

    def shutdown(self):
        self.train_loader.shutdown()
        self.test_loader.shutdown()
        if self.valid is not None and self.valid_loader is not None:
            self.valid_loader.shutdown()


class BaseContext:
    """Context class is used to manage a training run"""

    def __init__(self, args: Namespace, device: str):
        self.args = args
        self.device = device
        self.data: Data | None = None
        self.trainer: Trainer | None = None
        self.stats = Stats()
        self.epochs: list[int] = []

    @property
    def cfg(self) -> Config:
        if self.data:
            return self.data.cfg
        raise ValueError("config not loaded")

    @property
    def epoch(self) -> int:
        return self.stats.current_epoch

    def load_config(self, cfg: Config, verbose: bool = False) -> None:
        set_logdir(cfg.dir)
        train_data = cfg.dataset(self.args.datadir, "train")
        train_loader = cfg.dataloader("train")
        log.info(f"== train data: ==\n{train_data}")
        test_data = cfg.dataset(self.args.datadir, "test")
        test_loader = cfg.dataloader("test")
        log.info(f"== test data: ==\n{test_data}")
        if cfg.data("valid"):
            valid_data = cfg.dataset(self.args.datadir, "valid")
            valid_loader = cfg.dataloader("valid")
            log.info(f"== valid data: ==\n{valid_data}")
        else:
            valid_data = None
            valid_loader = None
        model = load_model(cfg, test_data.image_shape(), device=self.device)
        if verbose:
            log.info(str(model))
        self.data = Data(cfg, model, train_data, test_data, valid_data, train_loader, test_loader, valid_loader)

    def get_trainer(self, data: Data) -> Trainer:
        trainer = Trainer(self.cfg, data.model, device=self.device)
        if self.args.capture:
            trainer.capture_graph(data.train_loader, data.train.transform)
        return trainer

    def restart(self, clear_files: bool = False) -> bool:
        if not self.data or not isinstance(self.data.model, Model):
            return False
        self.cfg.save(clear=clear_files)
        set_seed(self.cfg.seed)
        self.data.model.init_weights()
        self.data.start(self.args.max_items, self.args.max_test_items)
        self.trainer = self.get_trainer(self.data)
        self.stats = Stats()
        self.stats.xrange = [1, self.cfg.epochs]
        return True

    def resume(self, epoch: int) -> bool:
        if not self.data or not isinstance(self.data.model, Model):
            return False
        self.cfg.save()
        self.data.start(self.args.max_items, self.args.max_test_items)
        self.trainer = self.get_trainer(self.data)
        self.stats = self.trainer.resume_from(epoch)
        if self.stats.current_epoch != epoch:
            raise RuntimeError(f"invalid checkpoint: epoch is {self.stats.current_epoch} - expected {epoch}")
        return True

    def do_epoch(self, should_stop: Callable | None = None) -> bool:
        """Returns true if run should stop"""
        if not self.trainer or not self.data:
            return True
        try:
            if self.args.capture:
                train_loss, batch_loss = self.trainer.train_capture(
                    self.data.train_loader,  self.data.train.transform, should_stop=should_stop
                )
            else:
                train_loss, batch_loss = self.trainer.train(
                    self.data.train_loader, self.data.train.transform, should_stop=should_stop
                )
            calc_top5 = (len(self.data.test.classes) >= 100)
            if self.data.valid is not None and self.data.valid_loader is not None:
                valid_loss, valid_accuracy, valid_top5_accuracy = self.trainer.test(
                    self.data.valid_loader, self.data.valid.transform, calc_top5=calc_top5, should_stop=should_stop
                )
            test_loss, test_accuracy, test_top5_accuracy = self.trainer.test(
                self.data.test_loader, self.data.test.transform, calc_top5=calc_top5, should_stop=should_stop
            )
        except RunInterrupted:
            self.data.shutdown()
            self.trainer.save(self.stats)
            return True

        self.stats.update_train(self.trainer.learning_rate(), train_loss, batch_loss)
        self.stats.update_test(test_loss, test_accuracy, test_top5_accuracy)
        if self.data.valid is not None and self.data.valid_loader is not None:
            self.stats.update_valid(valid_loss, valid_accuracy, valid_top5_accuracy)
        self.stats.predict = self.trainer.predict

        stop = self.trainer.should_stop(self.stats)
        self.stats.running = not stop
        self.trainer.step(self.stats)

        if stop or (self.epoch % self.trainer.log_every == 0):
            log.info(str(self.stats))
        if stop or (self.epoch % self.trainer.save_every == 0):
            self.trainer.save(self.stats)
        if stop or (self.epoch % (10*self.trainer.log_every) == 0):
            log.info(f"  Elapsed time = {self.stats.elapsed_total()}")
        if stop:
            self.data.shutdown()
            self.epochs = clear_checkpoints(self.epoch, self.trainer.save_every, self.cfg.dir)
        return stop

    def run(self, profile=None) -> None:
        raise NotImplementedError("abstract method")


class CmdContext(BaseContext):
    """CmdContext class is used to manage a command line training run"""

    def run(self, profile=None) -> None:
        try:
            cfg = Config(file=self.args.config, rundir=self.args.rundir, epochs=self.args.epochs, seed=self.args.seed)
            self.load_config(cfg)
        except (InvalidConfigError, FileNotFoundError) as err:
            print(f"error in config file '{self.args.config}': {err}")
            sys.exit(1)
        try:
            if self.args.resume:
                self.resume(self.args.resume)
            else:
                self.restart(self.args.clear)
        except InvalidConfigError as err:
            print(f"error in config file '{self.args.config}': {err}")
            sys.exit(1)
        except (FileNotFoundError, RuntimeError, KeyError) as err:
            log.error(f"error loading state from checkpoint: {err}")
            sys.exit(1)

        while not self.do_epoch():
            if profile:
                profile.step()


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

    def run(self, profile=None) -> None:
        while True:
            self.get_command()
            if self.started and not self.paused:
                if self.do_epoch(should_stop=self.get_command):
                    self.stats.running = False
                    self.paused = True
                self.db.update(self.stats, clear=self.stats.running, epochs=self.epochs)

    def load(self, name: str, version: str) -> None:
        log.info(f"cmd: load {name} version={version}")
        try:
            config = self.db.get_config(name, version)
            cfg = Config(name=name, data=config, rundir=self.args.rundir, seed=self.args.seed)
            self.load_config(cfg)
            self.stats = self.db.set_model(cfg)
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
        except (FileNotFoundError, RuntimeError, KeyError, InvalidConfigError) as err:
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
        if self.data:
            self.cfg.epochs = epoch
            self.stats.xrange = [0, epoch]
            self.db.update(self.stats)
            self.ok()
        else:
            self.error("config not loaded")

    def pause(self, on: bool) -> None:
        if on:
            log.info(f"cmd: pause")
            if self.started:
                self.paused = True
                if self.trainer and self.trainer.stopper:
                    self.trainer.stopper.update_stats(self.stats)
                self.db.set_running(not on)
            self.ok()
        else:
            state = self.db.get_state()
            self.start(state.epoch, False)

    def get_model(self) -> Model | None:
        if self.started and self.trainer and isinstance(self.trainer.model, Model):
            return self.trainer.model
        if not self.data:
            return None
        model = load_model(self.cfg, self.data.test.image_shape(), device=self.device)
        if not isinstance(model, Model):
            return None
        try:
            checkpoint = load_checkpoint(self.cfg.dir, device=self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
        except FileNotFoundError:
            pass
        return model

    def get_activations(self, name: str, layers: list[Index], index: int = 0, hist_bins: int = 0,
                        train: bool = False) -> None:
        log.debug(f"cmd: {name} {layers} {index} hist_bins={hist_bins}")
        model = self.get_model()
        if not self.data or not model:
            self.error("config not loaded")
            return
        ds = self.data.test
        ds.open()
        if name == "ml:activations":
            input = ds.get_data(index).view((1,)+ds.size)
        else:
            batch_size = self.cfg.data("test").get("batch_size", len(ds))
            input = ds.get_range(0, batch_size, nofilter=True)[0]
        input = input.to(self.device)
        if ds.transform is not None:
            input = ds.transform(input)
        layers = [Index(i) for i in layers]
        activations = model.activations(input, layers, hist_bins=hist_bins)
        for layer, val in activations.items():
            key = f"{layer}:{index}" if name == "ml:activations" else f"{layer}"
            log.debug(f"save {name} {key}")
            self.db.save(val, name, key)
        if train and self.started and self.trainer:
            self.trainer.model.train()
        self.ok()

    def get_command(self, train=False) -> bool:
        """Poll for command on 0MQ socket - returns True if run should stop"""
        if self.paused or not self.started:
            cmd = self.socket.recv_json()
        else:
            try:
                cmd = self.socket.recv_json(flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                return not self.started or self.paused
        match cmd:
            case ["load", str(name), str(version)]:
                self.load(name, version)
            case ["start", int(epoch), bool(clear)]:
                self.start(epoch, clear)
            case ["max_epoch", int(epoch)]:
                self.set_max_epoch(epoch)
            case ["pause", bool(on)]:
                self.pause(on)
            case ["activations", list(layers), int(index)]:
                self.get_activations("ml:activations", layers, index, train=train)
            case ["histograms", list(layers)]:
                self.get_activations("ml:histograms", layers, hist_bins=100, train=train)
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


def clear_checkpoints(last_epoch: int, save_every: int, dir: str) -> list[int]:
    """Remove old checkpoint files from previous runs and return current epoch list"""
    epochs = []
    for file in glob(path.join(dir, "model_*.pt")):
        epoch = int(path.basename(file)[6:-3])
        if epoch > last_epoch or (epoch < last_epoch and (epoch % save_every) != 0):
            log.debug(f"remove: {file}")
            os.remove(file)
        else:
            epochs.append(epoch)
    epochs.sort()
    return epochs
