import io
import json
import logging as log
import os
import sys
import time
from argparse import Namespace
from dataclasses import dataclass, field
from glob import glob
from os import path
from typing import Any, Callable

import redis
import torch
import zmq
from torch import nn

from .config import Config
from .model import Model
from .trainer import Datasets, Stats, Trainer
from .utils import (InvalidConfigError, RunInterrupted, get_device, set_logdir,
                    set_seed)

zmq_port = 5555
redis_port = 6379


@dataclass
class State:
    name: str = ""
    epochs: list[int] = field(default_factory=list)
    epoch: int = -1
    max_epoch: int = 0
    running: bool = False

    def dump(self) -> str:
        return json.dumps(self.__dict__)

    def load(self, data: bytes) -> None:
        self.__dict__ = json.loads(data)


class Database:
    """Redis database wrapper class"""

    def __init__(self, host: str, notify: bool = False):
        self.db = redis.Redis(host=host, port=redis_port, db=0)
        password = os.getenv("REDISCLI_AUTH")
        if password:
            self.db.auth(password)  # type: ignore
        self.db.ping()
        log.info(f"connected to redis server at {host}:{redis_port}")
        if notify:
            self.db.config_set("notify-keyspace-events", "KA")

    def init_config(self, dir: str) -> None:
        """Initialise ml.config hash from all .toml files in dir"""
        p = self.db.pipeline(transaction=True)
        p.delete("ml:config")
        for file in glob(path.join(dir, "*.toml")):
            with open(file, encoding="utf-8") as f:
                name = path.basename(file).removesuffix(".toml")
                p.hset("ml:config", name, f.read())
        p.execute()

    def update_config(self, file: str) -> None:
        """Called when config file is created, updated or deleted"""
        if not file.endswith(".toml"):
            return
        name = path.basename(file).removesuffix(".toml")
        try:
            with open(file, encoding="utf-8") as f:
                log.info(f"updated config: {name}")
                self.db.hset("ml:config", name, f.read())

        except FileNotFoundError:
            log.info(f"removed config: {name}")
            self.db.hdel("ml:config", name)

    def get_config(self, name: str) -> str:
        """Get toml config data as a string"""
        data = self.db.hget("ml:config", name)
        if not data:
            raise InvalidConfigError(f"config for {name} not found in DB")
        return data.decode("utf-8")

    def get_models(self) -> list[str]:
        """Get sorted list of defined models"""
        models = [key.decode("utf-8") for key in self.db.hkeys("ml:config")]
        models.sort()
        return models

    def set_model(self, name: str, dir: str, trainer: Trainer | None, device: str = "cpu") -> Stats:
        """Set current model and load epochs and stats from filesystem"""
        log.debug(f"set_model: name={name} dir={dir}")
        state = State(name=name)
        file = path.join(dir, "model.pt")
        stats = Stats()
        try:
            checkpoint = torch.load(file, map_location=device)
            stats.load_state_dict(checkpoint["stats_state_dict"])
            stats.running = False
            state.epoch = stats.current_epoch
            log.debug(f"set_model: loaded checkpoint from {file} - epoch={state.epoch}")
            if trainer:
                trainer.model.load_state_dict(checkpoint["model_state_dict"])
        except FileNotFoundError:
            pass
        for file in glob(path.join(dir, "model_*.pt")):
            filename = path.basename(file)
            state.epochs.append(int(filename[6:-3]))
        state.epochs.sort()
        self.save(stats.state_dict(), f"ml:stats")
        self.db.delete("ml:activations")
        self.db.delete("ml:histograms")
        self.db.set("ml:state", state.dump())
        return stats

    def get_state(self) -> State:
        """Get current state"""
        state = State()
        data = self.db.get("ml:state")
        if data:
            state.load(data)
        return state

    def set_running(self, running: bool) -> None:
        """Update running flag in state and stats key"""
        stats_dict = self.load("ml:stats")
        stats_dict["running"] = running
        self.save(stats_dict, "ml:stats")
        state = self.get_state()
        state.running = running
        self.db.set("ml:state", state.dump())

    def update(self, stats: Stats, clear: bool = False) -> None:
        """Save stats and state key"""
        state = self.get_state()
        state.running = stats.running
        if len(stats.xrange) == 2:
            state.max_epoch = stats.xrange[1]
        if state.running or stats.current_epoch > 0:
            state.epoch = stats.current_epoch
        if stats.current_epoch > 0 and stats.current_epoch not in state.epochs:
            state.epochs.append(stats.current_epoch)
            state.epochs.sort()
        self.save(stats.state_dict(), "ml:stats")
        if clear:
            self.db.delete("ml:activations")
            self.db.delete("ml:histograms")
        self.db.set("ml:state", state.dump())

    def save(self, object: Any, name: str, key: str = "") -> None:
        """Serialise given object and save it to key as a byte string"""
        # log.debug(f"save {name} {key}: {object}")
        buffer = io.BytesIO()
        torch.save(object, buffer)
        if key:
            self.db.hset(name, key, buffer.getvalue())
        else:
            self.db.set(name, buffer.getvalue())

    def load(self, name: str, key: str = "", device: str = "cpu") -> Any:
        """Load state_dict from serialised form saved by save method"""
        if key:
            data = self.db.hget(name, key)
        else:
            data = self.db.get(name)
        if not data:
            raise RuntimeError(f"db load: {name} {key} not found")
        object = torch.load(io.BytesIO(data), map_location=device)
        # log.debug(f"load {name} {key}: {object}")
        return object

    def check_exists(self, name: str, keys: list[str]) -> list[int]:
        """Get list of hash keys which are not present"""
        missing = []
        for key in keys:
            if not self.db.hexists(name, key):
                try:
                    n = key.index(":")
                    missing.append(int(key[:n]))
                except ValueError:
                    missing.append(int(key))
        log.debug(f"check_exists: {name} {keys} -> {missing}")
        return missing


class BaseContext:
    """Context class is used to manage a training run"""

    def __init__(self, args: Namespace, device: str):
        self.args = args
        self.device = device
        self.trainer: Trainer | None = None
        self.ds: Datasets | None = None
        self.stats = Stats()

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
        log.debug(cfg)
        dtype = torch.float16 if cfg.half else torch.float32
        self.ds = Datasets(cfg, self.args.datadir, device=self.device, dtype=dtype)
        model = Model(cfg, device=self.device)
        log.debug(f"== {cfg.name} model: ==\n{model}")
        self.trainer = Trainer(cfg, model)

    def restart(self, clear_files: bool = False) -> bool:
        if not self.trainer or not self.ds:
            return False
        set_seed(self.trainer.cfg.seed)
        self.trainer = Trainer(self.trainer.cfg, self.trainer.model)
        self.trainer.model.init_weights(self.ds.train_data.image_shape)
        self.trainer.cfg.save(clear=clear_files)
        self.stats = Stats()
        self.stats.xrange = [1, self.trainer.epochs]
        return True

    def resume(self, epoch: int) -> bool:
        if not self.trainer or not self.ds:
            return False
        self.trainer.cfg.save()
        self.stats = self.trainer.resume_from(epoch, self.device)
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
        self.stats.update(self.trainer.learning_rate(), train_loss, test_loss, test_accuracy, valid_loss, valid_accuracy)
        stop = self.trainer.should_stop(self.stats)
        self.stats.running = not stop

        if stop or (self.epoch % self.log_every == 0):
            self.trainer.save(self.stats)
            log.info(str(self.stats))

        if stop or (self.epoch % (10*self.log_every) == 0):
            log.info(f"  Elapsed time = {self.stats.elapsed_total()}")
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
                self.db.update(self.stats, clear=not self.paused)

    def load(self, name: str) -> None:
        log.info(f"cmd: load {name}")
        try:
            data = self.db.get_config(name)
            cfg = Config(name=name, data=data, rundir=self.args.rundir, seed=self.args.seed)
            self.load_config(cfg)
            self.stats = self.db.set_model(name, cfg.dir, self.trainer, device=self.device)
            self.ok()
        except InvalidConfigError as err:
            self.error(f"config '{name}': {err}")
        self.started = False
        self.paused = True

    def start(self, epoch: int, clear: bool) -> None:
        log.info(f"cmd: start {epoch} {clear}")
        try:
            if epoch == 0:
                self.started = self.restart(clear)
            else:
                self.started = self.resume(epoch)
        except FileNotFoundError as err:
            self.error(f"error loading checkpoint: {err}")
            return
        if self.started and self.trainer:
            self.stats.running = True
            self.db.update(self.stats, clear=True)
            self.ok()
            self.paused = False
        else:
            self.error("config not loaded")

    def set_max_epoch(self, epoch: int) -> None:
        log.info(f"cmd: max_epoch {epoch}")
        if self.trainer:
            self.trainer.epochs = epoch
            self.stats.xrange = [0, epoch]
            self.db.update(self.stats)
            self.ok()
        else:
            self.error("config not loaded")

    def pause(self, on: bool) -> None:
        if on or self.started:
            log.info(f"cmd: pause {on}")
            self.paused = on
            self.db.set_running(not on)
            self.ok()
        else:
            state = self.db.get_state()
            self.start(state.epoch, False)

    def get_activations(self, layers: list[int], index: int) -> None:
        log.debug(f"cmd: activations {layers} {index}")
        if not self.trainer or not self.ds:
            self.error("config not loaded")
            return
        test_data = self.ds.test_data
        input = test_data.data[index]
        activations = self.trainer.model.activations(input.view((1,)+input.size()), layers)
        for layer in layers:
            self.db.save(activations[layer], "ml:activations", f"{layer}:{index}")
        self.ok()

    def get_histograms(self, layers: list[int]) -> None:
        log.debug(f"cmd: histograms {layers}")
        if not self.trainer or not self.ds:
            self.error("config not loaded")
            return
        test_data = self.ds.test_data
        input = test_data.data[:test_data.batch_size]
        histograms = self.trainer.model.activations(input, layers, hist_bins=100)
        for layer in layers:
            self.db.save(histograms[layer], "ml:histograms", str(layer))
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
                self.get_activations(layers, index)
            case ["histograms", list(layers)]:
                self.get_histograms(layers)
            case _:
                self.error(f"invalid command: {cmd}")
        return not self.started or self.paused

    def ok(self) -> None:
        self.socket.send_json(["ok", ""])

    def error(self, msg: str) -> None:
        log.error(f"error: {msg}")
        self.socket.send_json(["error", msg])