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

from .config import Config, Index
from .trainer import Stats
from .utils import InvalidConfigError, load_checkpoint

redis_port = 6379


@dataclass
class State:
    name: str = ""
    version: str = ""
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

    def init_config(self, cfgdir: str, rundir: str) -> None:
        """Initialise ml.config hash from all .toml files in dir"""
        p = self.db.pipeline(transaction=True)
        p.delete("ml:config")
        # previous runs
        for file in glob(path.join(rundir, "*/*/config.toml")):
            cfg = Config(rundir, file=file)
            p.hset("ml:config", cfg.name+":"+cfg.version, cfg.text)
        # and current config files
        for file in glob(path.join(cfgdir, "*.toml")):
            cfg = Config(rundir, file=file)
            p.hset("ml:config", cfg.name+":"+cfg.version, cfg.text)
        p.execute()
        self.set_model_state(self.get_state())

    def update_config(self, file: str, rundir: str) -> State:
        """Called when config file is created, updated or deleted"""
        state = self.get_state()
        if file.endswith(".toml"):
            try:
                cfg = Config(rundir, file=file)
                log.info(f"updated config: {cfg.name}:{cfg.version}")
                self.db.hset("ml:config", cfg.name+":"+cfg.version, cfg.text)
                if state.name == cfg.name:
                    state.version = cfg.version
            except FileNotFoundError:
                name = path.basename(file).removesuffix(".toml")
                log.info(f"removed config: {name}")
                for key in self.db.hkeys("ml:config"):
                    if key.decode().startswith(name+":"):
                        self.db.hdel("ml:config", key)
        return self.set_model_state(state)

    def get_config(self, name: str, version: str) -> str:
        """Get toml config data as a string"""
        data = self.db.hget("ml:config", name+":"+version)
        if not data:
            raise InvalidConfigError(f"config for {name} version {version} not found in DB")
        return data.decode()

    def get_models(self) -> list[tuple[str, list[str]]]:
        """Get sorted list of defined models"""
        names = []
        versions = {}
        for key in self.db.hkeys("ml:config"):
            name, version = key.decode().split(":")
            if name not in names:
                names.append(name)
                versions[name] = [version]
            else:
                versions[name].append(version)
        return [(name, list(sorted(versions[name]))) for name in sorted(names)]

    def set_model(self, cfg: Config) -> Stats:
        """Set current model and load epochs and stats from filesystem"""
        log.debug(f"set_model: name={cfg.name} version={cfg.version} dir={cfg.dir}")
        state = State(name=cfg.name, version=cfg.version, max_epoch=cfg.epochs)
        stats = Stats()
        try:
            checkpoint = load_checkpoint(cfg.dir)
            stats.load_state_dict(checkpoint["stats_state_dict"])
            log.debug(f"set_model: loaded stats - epoch={stats.current_epoch}")
        except FileNotFoundError:
            pass
        stats.xrange = [0, cfg.epochs]
        stats.running = False
        state.epoch = stats.current_epoch
        for file in glob(path.join(cfg.dir, "model_*.pt")):
            filename = path.basename(file)
            state.epochs.append(int(filename[6:-3]))
        state.epochs.sort()
        self.save(stats.state_dict(), f"ml:stats")
        self.db.delete("ml:activations")
        self.db.delete("ml:histograms")
        self.set_model_state(state)
        return stats

    def set_model_state(self, state: State) -> State:
        """Refresh model checksum and list of models in state hash"""
        state.models = [key.decode() for key in self.db.hkeys("ml:config")]
        if state.name:
            data = self.db.hget("ml:config", state.name+":"+state.version)
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
