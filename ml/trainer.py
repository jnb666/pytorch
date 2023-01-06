import datetime
import logging as log
import math
import os
import random
import statistics
import time
from os import path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .config import Config
from .dataset import Dataset
from .model import Model
from .utils import (DatasetNotFoundError, InvalidConfigError, RunInterrupted,
                    pformat)


class Stats():
    """Stats containes the data used to plot loss and accuracy statistics."""

    def __init__(self):
        self.current_epoch = 0
        self.xrange = []
        self.epoch = []
        self.train_loss = []
        self.test_loss = []
        self.test_accuracy = []
        self.valid_loss = []
        self.valid_accuracy = []
        self.valid_accuracy_avg = []
        self.learning_rate = []
        self.predict = torch.tensor([], dtype=torch.int64)
        self.start = [time.time()]
        self.elapsed = [0.0]
        self.running = False

    def __str__(self):
        s = "Epoch {:3d}:  Train loss: {:.3f}  ".format(self.current_epoch, self.train_loss[-1])
        s += "Test loss: {:.3f} accuracy: {:.1%}".format(self.test_loss[-1], self.test_accuracy[-1])
        if len(self.valid_loss):
            s += "  Valid loss: {:.3f}".format(self.valid_loss[-1])
            s += " accuracy: {:.1%}".format(self.valid_accuracy[-1])
            if len(self.valid_accuracy_avg):
                s += " avg: {:.2%}".format(self.valid_accuracy_avg[-1])
        return s

    def elapsed_total(self) -> str:
        """Total elapsed time"""
        total = sum(self.elapsed)
        if total < 60:
            return f"{total:.1f}s"
        else:
            return str(datetime.timedelta(seconds=round(total)))

    def update(self, learning_rate: float, train_loss: float, test_loss: float, test_accuracy: float,
               valid_loss: float | None = None, valid_accuracy: float | None = None):
        """Add new record to stats history"""
        self.elapsed[-1] = time.time() - self.start[-1]
        self.epoch.append(self.current_epoch)
        self.learning_rate.append(learning_rate)
        self.train_loss.append(train_loss)
        self.test_loss.append(test_loss)
        self.test_accuracy.append(test_accuracy)
        if valid_loss is not None:
            self.valid_loss.append(valid_loss)
            self.valid_accuracy.append(valid_accuracy)

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state):
        self.__dict__.update(state)
        self.start.append(time.time())
        self.elapsed.append(0.0)

    def table_columns(self) -> list[str]:
        if len(self.valid_loss):
            return ["train loss", "test loss", "accuracy", "valid loss", "accuracy "]
        else:
            return ["train loss", "test loss", "accuracy"]

    def table_data(self) -> list[tuple[float, ...]]:
        data: list[tuple[float, ...]] = []
        for i in range(len(self.epoch)):
            if len(self.valid_loss) == 0:
                data.append((self.train_loss[i], self.test_loss[i], 100*self.test_accuracy[i]))
            else:
                data.append((self.train_loss[i], self.test_loss[i], 100*self.test_accuracy[i],
                             self.valid_loss[i], 100 * self.valid_accuracy[i]))
        return data


class Stopper:
    """Stopper checks whether the stopping condition is met
    Args:
        epochs      number of epochs for which average loss has not decreased
        extra       number of extra epochs after stopping condition is met - transform is disabled for these
        avg         number of epochs to calc the mean loss
    """

    def __init__(self, epochs: int = 3, extra: int = 0, avg: int = 20):
        log.info(f"init stopper: epochs={epochs} extra={extra} avg={avg}")
        self.epochs = epochs
        self.extra = extra
        self.avg = avg
        self.stopping = -1

    def step(self, stats: Stats) -> bool:
        """Returns True if should stop and updates stats.valid_loss_avg"""
        if len(stats.valid_loss) == 0:
            return False

        stats.valid_accuracy_avg.append(mean(stats.valid_accuracy, len(stats.valid_accuracy)-1, self.avg))
        if len(stats.valid_accuracy_avg) < max(self.avg, self.epochs + 1):
            return False

        if self.stopping >= 0:
            log.debug(f"stopper: stop in {self.stopping} epochs")
            self.stopping -= 1
            return self.stopping <= 0

        val = stats.valid_accuracy_avg[-1]
        prev = np.array(stats.valid_accuracy_avg[-self.epochs-1:-1])
        if val < np.min(prev):
            log.info(f"  valid_avg={val:.3%} {prev*100} - stop in {self.extra} epochs")
            self.stopping = self.extra - 1
            return self.stopping <= 0
        else:
            log.debug(f"valid_avg={val:.3%} {prev*100}")
        return False

    def update_stats(self, stats: Stats):
        """In case where run is restarted - recalc average loss as config may have changed"""
        stats.valid_accuracy_avg.clear()
        for i in range(len(stats.valid_accuracy)):
            stats.valid_accuracy_avg.append(mean(stats.valid_accuracy, i, self.avg))
        log.debug(f"valid accuracy:{np.array(stats.valid_accuracy)}")
        log.debug(f"valid average: {np.array(stats.valid_accuracy_avg)}")


class Datasets:
    """Datasets wraps the training, test, validation datasets and augmentation transform"""

    def __init__(self, cfg: Config, rootdir: str, device: str = "cpu", dtype: torch.dtype = torch.float32):
        try:
            self.train_data = cfg.dataset(rootdir, "train", device, dtype)
            log.info(f"== train data: ==\n{self.train_data}")
            self.test_data = cfg.dataset(rootdir, "test", device, dtype)
            log.info(f"== test data: ==\n{self.test_data}")
        except DatasetNotFoundError as typ:
            raise InvalidConfigError(f"{typ} data set not found")
        self.valid_data: Dataset | None
        try:
            self.valid_data = cfg.dataset(rootdir, "valid", device, dtype)
            log.info(f"== valid data: ==\n{self.valid_data}")
        except DatasetNotFoundError:
            self.valid_data = None
        self.transform = cfg.transforms()
        if self.transform:
            log.info(f"== transforms: ==\n{self.transform}")


class Trainer:
    """Trainer optimises the model weights for a given training dataset and evaluations the loss and accuracy.

    Args:
        cfg:dict             settings from config [train] section
        model:Model          pytorch neural network model
        loss_fn              loss function - defaults ti nn.CrossEntropyLoss

    Attributes:
        model:nn.Module      pytorch network model
        optimizer            torch.optim.Optimizer
        scheduler            torch.optim.lr_scheduler (optional)
        epochs:int           number of epochs to train for
        shuffle:bool         flag set if data is to be shuffled at start of each epoch
        log_every:int        frequency to log stats to stdout
        predict:Tensor       predictions from last test run
        dir:str              directory to save stats and weights
        stopper:Stopper      optional object to check if should stop after this epoch
    """

    def __init__(self, cfg: Config, model: Model, loss_fn=None):
        self.cfg = cfg
        self.dir = cfg.dir
        self.shuffle = cfg.train.get("shuffle", False)
        self.log_every = int(cfg.train.get("log_every", 1))
        self.model = model
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn
        self.predict = torch.tensor([], dtype=torch.int64)
        self.optimizer = cfg.optimizer(self.model)
        self.scheduler = cfg.scheduler(self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.half)
        stop_cfg = self.cfg.train.get("stop")
        self.stopper = Stopper(**stop_cfg) if stop_cfg else None
        log.info(f"== Trainer: ==\n{self}")

    @property
    def epochs(self):
        return self.cfg.epochs

    def __str__(self) -> str:
        return pformat(self.cfg.train)

    def learning_rate(self) -> float:
        lr = 0.0
        for group in self.optimizer.param_groups:
            lr = group['lr']
        return lr

    def save(self, stats: Stats) -> None:
        """Save current model, optimizer, scheduler state and stats to file"""
        checkpoint = {
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": to_list(np.random.get_state()),
            "random_state": random.getstate(),
            "stats_state_dict": stats.state_dict(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.cfg.half:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        file = path.join(self.dir, f"model_{stats.current_epoch}.pt")
        log.debug(f"save checkpoint to {file}")
        torch.save(checkpoint, file)
        link(file, path.join(self.dir, "model.pt"))

    def resume_from(self, epoch: int, device: str = "cpu") -> Stats:
        """Load saved stats, weights and random state from checkpoint file"""
        log.info(f"resume from epoch {epoch}")
        file = path.join(self.dir, f"model_{epoch}.pt")
        log.debug(f"load checkpoint from {file} map_location={device}")
        checkpoint = torch.load(file, map_location=device)

        stats = Stats()
        stats.load_state_dict(checkpoint["stats_state_dict"])
        stats.current_epoch = epoch
        stats.xrange[1] = max(stats.xrange[1], self.epochs)

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if self.cfg.half:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except (RuntimeError, KeyError) as err:
            log.error(f"error loading trainer state from {file}: {err}")
            stats.current_epoch = 0

        if hasattr(checkpoint, "torch_rng_state"):
            torch.set_rng_state(checkpoint["torch_rng_state"].to("cpu"))
        if hasattr(checkpoint, "numpy_rng_state"):
            np.random.set_state(checkpoint["numpy_rng_state"])
        if hasattr(checkpoint, "random_state"):
            random.setstate(checkpoint["random_state"])
        return stats

    def train(self, train_data: Dataset, transform: nn.Module | None = None, half: bool = False,
              should_stop: Callable[[], bool] | None = None) -> float:
        """Train one epoch against training dataset - returns training loss"""
        if self.shuffle:
            train_data.shuffle()
        train_loss = 0.0
        for i, (data, targets) in enumerate(train_data):
            if transform is not None and (self.stopper is None or self.stopper.stopping < 0):
                with torch.no_grad():
                    data = transform(data)
            self.model.train()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.cfg.half):
                pred = self.model(data)
                loss = self.loss_fn(pred, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            train_loss += float(loss) / len(train_data)
            if should_stop and should_stop():
                raise RunInterrupted()
        if self.scheduler:
            self.scheduler.step()
        return train_loss

    def test(self, test_data: Dataset, should_stop: Callable[[], bool] | None = None) -> tuple[float, float]:
        """Calculate loss and accuracy against the test set - returns test loss and accuracy"""
        self.model.eval()
        if len(self.predict) != test_data.nitems:
            self.predict = torch.zeros((test_data.nitems), dtype=torch.int64)
        test_loss = 0.0
        total_correct = 0
        samples = 0
        with torch.no_grad():
            for data, targets in test_data:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.cfg.half):
                    pred = self.model(data)
                    loss = self.loss_fn(pred, targets).item()
                outputs = pred.argmax(1)
                correct = int((outputs == targets).type(torch.float).sum())
                self.predict[samples: samples+len(targets)] = outputs
                test_loss += float(loss) / len(test_data)
                total_correct += correct
                samples += len(targets)
                if should_stop and should_stop():
                    raise RunInterrupted()
        return test_loss, total_correct/samples

    def should_stop(self, stats: Stats) -> bool:
        """Returns True if hit stopping condition"""
        stop = False
        if self.stopper is not None:
            stop = self.stopper.step(stats)
        if stats.current_epoch >= self.epochs:
            return True
        else:
            return stop


def link(file, link):
    try:
        os.remove(link)
    except FileNotFoundError:
        pass
    os.link(file, link)


def to_list(tuple):
    res = []
    for x in tuple:
        if isinstance(x, np.ndarray):
            res.append(x.tolist())
        else:
            res.append(x)
    return res


def mean(vals: list[float], i: int, avg: int) -> float:
    return statistics.mean(vals[max(i+1-avg, 0):i+1])
