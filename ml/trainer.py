import datetime
import json
import logging as log
import os
import statistics
import sys
import time
from os import path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .config import Config
from .dataset import Dataset
from .utils import load_checkpoint, pformat


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
        self.valid_loss_avg = []
        self.predict = []
        self.start = [time.time()]
        self.elapsed = [0.0]

    def __str__(self):
        s = "Epoch {:3d}:  Train loss: {:.3f}  ".format(self.current_epoch, self.train_loss[-1])
        if len(self.valid_loss) != 0:
            s += "Valid loss: {:.3f}  accuracy: {:.1%}  ".format(self.valid_loss[-1], self.valid_accuracy[-1])
        s += "Test loss: {:.3f}  accuracy: {:.1%}".format(self.test_loss[-1], self.test_accuracy[-1])
        return s

    def elapsed_total(self) -> datetime.timedelta:
        """Total elapsed time"""
        total = sum(self.elapsed)
        if total < 10:
            return datetime.timedelta(seconds=total)
        else:
            return datetime.timedelta(seconds=round(total))

    def update(self, predict: Tensor, train_loss: float, test_loss: float, test_accuracy: float,
               valid_loss: float | None = None, valid_accuracy: float | None = None,
               valid_loss_avg: float | None = None):
        """Add new record to stats history"""
        self.elapsed[-1] = time.time() - self.start[-1]
        self.predict = predict
        self.epoch.append(self.current_epoch)
        self.train_loss.append(train_loss)
        self.test_loss.append(test_loss)
        self.test_accuracy.append(test_accuracy)
        if valid_loss is not None:
            self.valid_loss.append(valid_loss)
            self.valid_accuracy.append(valid_accuracy)
        if valid_loss_avg is not None:
            self.valid_loss_avg.append(valid_loss_avg)

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state):
        self.__dict__.update(state)
        self.start.append(time.time())
        self.elapsed.append(0.0)

    def table_columns(self) -> list[str]:
        if len(self.valid_loss):
            return ["train loss", "valid loss", "accuracy ", "test loss", "accuracy"]
        else:
            return ["train loss", "test loss", "accuracy"]

    def table_data(self) -> list[tuple[float, ...]]:
        data: list[tuple[float, ...]] = []
        for i in range(len(self.epoch)):
            if len(self.valid_loss) == 0:
                data.append((self.train_loss[i], self.test_loss[i], 100*self.test_accuracy[i]))
            else:
                data.append((self.train_loss[i], self.valid_loss[i], 100 * self.valid_accuracy[i],
                             self.test_loss[i], 100*self.test_accuracy[i]))
        return data


class Stopper:
    """Stopper checks whether the stopping condition is met"""

    def __init__(self, epochs: int = 3, extra: int = 0, ema: float = 20, eps: float = 1e-4):
        log.info(f"init stopper: epochs={epochs} extra={extra} ema={ema}")
        self.epochs = epochs
        self.extra = extra
        self.k = 2 / (ema + 1)
        self.avg: list[float] = []
        self.eps = eps
        self.stopping = -1

    @property
    def average(self) -> float:
        return self.avg[-1] if len(self.avg) > 0 else 0.0

    def step(self, stats: Stats) -> bool:
        """Returns True if should stop"""
        if self.stopping >= 0:
            log.debug(f"stopper: stop in {self.stopping} epochs")
            self.stopping -= 1
            return self.stopping < 0
        if len(stats.valid_loss) == 0:
            return False
        x = stats.valid_loss[-1]
        if len(self.avg) == 0:
            self.avg.append(x)
        else:
            self.avg.append(x*self.k + self.avg[-1]*(1-self.k))
        if len(self.avg) < self.epochs + 1:
            return False
        val = self.avg[-1]
        stop = (val - max(self.avg[-self.epochs-1:-1])) / val > -self.eps
        prev = np.array(self.avg[-self.epochs-1:-1])
        if stop:
            log.info(f"valid_avg={val:.4} {prev} - stop in {self.extra} epochs")
            self.stopping = self.extra - 1
            return self.stopping <= 0
        else:
            log.debug(f"valid_avg={val:.4} {prev}")
        return False


class Trainer:
    """Trainer optimises the model weights for a given training dataset and evaluations the loss and accuracy.

    Args:
        cfg:dict             settings from config [train] section
        model:nn.Module      pytorch neural network model
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

    def __init__(self, cfg: Config, model: nn.Module, loss_fn=None):
        self.cfg = cfg
        self.model = model
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn
        self.optimizer = cfg.optimizer(model)
        self.scheduler = cfg.scheduler(self.optimizer)
        self.dir = cfg.dir
        self.epochs = int(cfg.train.get("epochs", 10))
        self.shuffle = cfg.train.get("shuffle", False)
        self.log_every = int(cfg.train.get("log_every", 1))
        self.predict: Tensor | None = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.half)
        self.stopper: Stopper | None = None
        try:
            cfg = self.cfg.train["stop"]
            self.stopper = Stopper(**cfg)  # type: ignore
        except KeyError:
            pass
        log.info(f"== Trainer: ==\n{self}")

    def __str__(self) -> str:
        return pformat(self.cfg.train)

    def save(self, stats: Stats) -> None:
        """Save current model, optimizer, scheduler ste and stats to file"""
        checkpoint = {
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": to_list(np.random.get_state()),
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
        try:
            checkpoint = load_checkpoint(self.dir, epoch, device)
        except FileNotFoundError as err:
            print(f"Resume file not found: {err}")
            sys.exit(1)

        stats = Stats()
        stats.load_state_dict(checkpoint["stats_state_dict"])
        stats.current_epoch = epoch
        stats.xrange[1] = max(stats.xrange[1], self.epochs)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if self.cfg.half:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        torch.set_rng_state(checkpoint["torch_rng_state"].to("cpu"))
        np.random.set_state(checkpoint["numpy_rng_state"])
        return stats

    def train(self, train_data: Dataset, transform: nn.Module | None = None, half: bool = False) -> float:
        """Train one epoch against training dataset - returns training loss"""
        self.model.train()
        if self.shuffle:
            train_data.shuffle()
        train_loss = 0
        for i, (data, targets) in enumerate(train_data):
            if transform is not None and (self.stopper is None or self.stopper.stopping < 0):
                if i == 0:
                    log.debug("apply transform")
                with torch.no_grad():
                    data = transform(data)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.cfg.half):
                pred = self.model(data)
                loss = self.loss_fn(pred, targets)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            train_loss += loss.item() / len(train_data)
        if self.scheduler:
            self.scheduler.step()
        return train_loss

    def test(self, test_data: Dataset) -> tuple[float, float]:
        """Calculate loss and accuracy against the test set - returns test loss and accuracy"""
        self.model.eval()
        if self.predict is None or len(self.predict) != test_data.nitems:
            self.predict = torch.zeros((test_data.nitems), dtype=torch.int64)
        test_loss = 0
        total_correct = 0
        samples = 0
        with torch.no_grad():
            for data, targets in test_data:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.cfg.half):
                    pred = self.model(data)
                    loss = self.loss_fn(pred, targets).item()
                outputs = pred.argmax(1)
                correct = int((outputs == targets).type(torch.float).sum())
                self.predict[samples: samples+len(targets)] = outputs
                test_loss += loss / len(test_data)
                total_correct += correct
                samples += len(targets)
        return test_loss, total_correct/samples

    def should_stop(self, stats: Stats) -> bool:
        """Returns True if hit stopping condition"""
        if stats.current_epoch >= self.epochs:
            return True
        if self.stopper is not None:
            return self.stopper.step(stats)
        return False


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
