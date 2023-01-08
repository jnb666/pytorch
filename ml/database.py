import hashlib
import io
import json
import logging as log
import os
from dataclasses import dataclass, field
from glob import glob
from os import path
from typing import Any

import redis
import torch

from .config import Index
from .trainer import Stats
from .utils import InvalidConfigError

redis_port = 6379


@dataclass
class State:
    name: str = ""
    epochs: list[int] = field(default_factory=list)
    epoch: int = -1
    max_epoch: int = 0
    running: bool = False
    models: list[str] = field(default_factory=list)
    checksum: str = ""
    error: str = ""

    def dump(self) -> str:
        return json.dumps(self.__dict__)

    def load(self, data: bytes) -> None:
        self.__dict__ = json.loads(data)

    def update(self, s: "State") -> None:
        self.__dict__.update(s.__dict__)


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
        self.set_model_state()

    def update_config(self, file: str) -> State:
        """Called when config file is created, updated or deleted"""
        if file.endswith(".toml"):
            name = path.basename(file).removesuffix(".toml")
            try:
                with open(file, encoding="utf-8") as f:
                    log.info(f"updated config: {name}")
                    self.db.hset("ml:config", name, f.read())
            except FileNotFoundError:
                log.info(f"removed config: {name}")
                self.db.hdel("ml:config", name)
        return self.set_model_state()

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

    def set_model(self, name: str, dir: str, max_epoch: int) -> Stats:
        """Set current model and load epochs and stats from filesystem"""
        log.debug(f"set_model: name={name} dir={dir}")
        state = State(name=name, max_epoch=max_epoch)
        file = path.join(dir, "model.pt")
        stats = Stats()
        try:
            checkpoint = torch.load(file, map_location="cpu")
            stats.load_state_dict(checkpoint["stats_state_dict"])
            log.debug(f"set_model: loaded stats from {file} - epoch={stats.current_epoch}")
        except FileNotFoundError:
            pass
        stats.xrange = [0, max_epoch]
        stats.running = False
        state.epoch = stats.current_epoch
        for file in glob(path.join(dir, "model_*.pt")):
            filename = path.basename(file)
            state.epochs.append(int(filename[6:-3]))
        state.epochs.sort()
        self.save(stats.state_dict(), f"ml:stats")
        self.db.delete("ml:activations")
        self.db.delete("ml:histograms")
        self.set_model_state(state)
        return stats

    def set_model_state(self, state: State | None = None) -> State:
        """Refresh model checksum and list of models in state hash"""
        if state is None:
            state = self.get_state()
        state.models = self.get_models()
        if state.name:
            data = self.db.hget("ml:config", state.name)
            if data:
                m = hashlib.sha256()
                m.update(data)
                state.checksum = m.hexdigest()
        self.db.set("ml:state", state.dump())
        return state

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

    def update(self, stats: Stats, clear: bool = False, epochs: list[int] | None = None) -> None:
        """Save stats and state key"""
        state = self.get_state()
        state.running = stats.running
        if len(stats.xrange) == 2:
            state.max_epoch = stats.xrange[1]
        if state.running or stats.current_epoch > 0:
            state.epoch = stats.current_epoch
        if epochs:
            state.epochs = epochs
        elif stats.current_epoch > 0 and stats.current_epoch not in state.epochs:
            state.epochs.append(stats.current_epoch)
            state.epochs.sort()
        self.save(stats.state_dict(), "ml:stats")
        if clear:
            self.db.delete("ml:activations")
            self.db.delete("ml:histograms")
        self.db.set("ml:state", state.dump())

    def save(self, object: Any, name: str, key: str = "") -> None:
        """Serialise given object and save it to key as a byte string"""
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
        return object

    def check_exists(self, name: str, keys: list[str]) -> list[Index]:
        """Get list of hash keys which are not present"""
        log.debug(f"check_exists: {name} {keys}")
        missing = []
        for key in keys:
            if not self.db.hexists(name, key):
                try:
                    n = key.index(":")
                    missing.append(key_to_index(key[:n]))
                except ValueError:
                    missing.append(key_to_index(key))
        log.debug(f"missing {missing}")
        return missing


def key_to_index(key: str) -> Index:
    ix = Index([int(s) for s in key.split(".")])
    return ix
