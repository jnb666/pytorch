import datetime
import logging as log
import math
import os
import random
import statistics
import sys
import time
from os import path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.profiler import record_function

from .config import Config
from .dataset import Dataset, Loader, Transforms
from .model import Model
from .utils import (InvalidConfigError, RunInterrupted, load_checkpoint,
                    pformat, save_checkpoint)


class Stats():
    """Stats containes the data used to plot loss and accuracy statistics."""

    def __init__(self):
        self.current_epoch = 0
        self.xrange = []
        self.epoch = []
        self.batch_loss = []
        self.train_loss = []
        self.test_loss = []
        self.test_accuracy = []
        self.test_top5_accuracy = []
        self.valid_loss = []
        self.valid_accuracy = []
        self.valid_top5_accuracy = []
        self.valid_accuracy_avg = []
        self.learning_rate = []
        self.predict = torch.tensor([], dtype=torch.int64)
        self.start = [time.time()]
        self.elapsed = [0.0]
        self.running = False

    def clear(self) -> None:
        """Clear to default"""
        self.__dict__.update(Stats().__dict__)

    def __str__(self) -> str:
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

    def update_train(self, learning_rate: float, avg_loss: float, batch_loss: Tensor) -> None:
        """Add new record with training set details to stats history"""
        self.current_epoch += 1
        self.elapsed[-1] = time.time() - self.start[-1]
        self.epoch.append(self.current_epoch)
        self.learning_rate.append(learning_rate)
        self.train_loss.append(avg_loss)
        self.batch_loss.append(batch_loss)

    def update_test(self, loss: float, accuracy: float, top5_accuracy: float | None) -> None:
        """Add new record with test set details to stats history"""
        self.test_loss.append(loss)
        self.test_accuracy.append(accuracy)
        if top5_accuracy is not None:
            self.test_top5_accuracy.append(top5_accuracy)

    def update_valid(self, loss: float, accuracy: float, top5_accuracy: float | None) -> None:
        """Add new record with validation set details to stats history"""
        self.valid_loss.append(loss)
        self.valid_accuracy.append(accuracy)
        if top5_accuracy is not None:
            self.valid_top5_accuracy.append(top5_accuracy)

    def state_dict(self) -> dict[str, Any]:
        return self.__dict__

    def load_state_dict(self, state) -> None:
        self.__dict__.update(state)
        self.start.append(time.time())
        self.elapsed.append(0.0)

    def table_columns(self) -> list[str]:
        flds = ["train loss", "test loss", "accuracy"]
        if len(self.test_top5_accuracy):
            flds.append("top5 accuracy")
        if len(self.valid_loss):
            flds.extend(["valid loss", "accuracy "])
        if len(self.valid_top5_accuracy):
            flds.append("top5 accuracy ")
        return flds

    def table_data(self) -> list[tuple[float, ...]]:
        data: list[tuple[float, ...]] = []
        for i in range(len(self.epoch)):
            r = [self.train_loss[i], self.test_loss[i], 100*self.test_accuracy[i]]
            j = i-self.current_epoch
            if len(self.test_top5_accuracy) >= abs(j):
                r.append(100*self.test_top5_accuracy[j])
            elif len(self.test_top5_accuracy):
                r.append(0)
            if len(self.valid_loss):
                r.extend([self.valid_loss[i], 100*self.valid_accuracy[i]])
            if len(self.valid_top5_accuracy) >= abs(j):
                r.append(100*self.valid_top5_accuracy[j])
            elif len(self.valid_top5_accuracy):
                r.append(0)
            data.append(tuple(r))
        return data


class Stopper:
    """Stopper checks whether the stopping condition is met
    Args:
        epochs      number of epochs for which average loss has not decreased
        extra       number of extra epochs after stopping condition is met
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


class Trainer:
    """Trainer optimises the model weights for a given training dataset and evaluations the loss and accuracy.

    Args:
        cfg:dict             settings from config [train] section
        model:Model          pytorch neural network model
        loss_fn              loss function - defaults ti nn.CrossEntropyLoss
        device               device to use for training and test runs ("cpu" | "cuda" | "mps")

    Attributes:
        model:nn.Module      pytorch network model
        optimizer            torch.optim.Optimizer
        scheduler            torch.optim.lr_scheduler (optional)
        metric               optional scheduler metric [used by ReduceLROnPlateau]
        epochs:int           number of epochs to train for
        shuffle:bool         flag set if data is to be shuffled at start of each epoch
        log_every:int        frequency to log stats to stdout
        save_every:int       frequency to save checkpoint file
        predict:Tensor       predictions from last test run
        dir:str              directory to save stats and weights
        stopper:Stopper      optional object to check if should stop after this epoch
    """

    def __init__(self, cfg: Config, model: nn.Module, loss_fn=None, device: str = "cpu"):
        self.cfg = cfg
        self.dir = cfg.dir
        self.device = device
        self.log_every = int(cfg.train.get("log_every", 1))
        self.save_every = int(cfg.train.get("save_every", 10))
        self.model = model
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn
        self.predict = torch.tensor([], dtype=torch.int64)
        self.batch_loss = torch.tensor([], dtype=torch.float32)
        self.optimizer = cfg.optimizer(self.model)
        self.scheduler, self.metric = cfg.scheduler(self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg.half)
        stop_cfg = self.cfg.train.get("stop")
        try:
            self.stopper = Stopper(**stop_cfg) if stop_cfg else None
        except TypeError as err:
            raise InvalidConfigError(f"stop: {err}")
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
            "stats_state_dict": stats.state_dict(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.cfg.half:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        save_checkpoint(self.dir, stats.current_epoch, checkpoint)

    def resume_from(self, epoch: int) -> Stats:
        """Load saved stats, weights and random state from checkpoint file"""
        log.info(f"resume from epoch {epoch}")
        checkpoint = load_checkpoint(self.dir, epoch, device=self.device, set_rng_state=True)
        stats = Stats()
        stats.load_state_dict(checkpoint["stats_state_dict"])
        stats.current_epoch = epoch
        stats.xrange[1] = max(stats.xrange[1], self.epochs)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if self.cfg.half:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except KeyError as err:
            log.warning(f"skip load for {err}")
        if self.stopper:
            self.stopper.update_stats(stats)
        return stats

    def train(self, train_data: Loader, transform: Transforms | None = None,
              should_stop: Callable | None = None) -> tuple[float, Tensor]:
        """Train one epoch against training dataset - returns average training loss and loss per batch"""
        self.model.train()
        train_loss = 0.0
        batches = len(train_data)
        if len(self.batch_loss) != batches:
            self.batch_loss = torch.zeros(batches, dtype=torch.float32)
        dtype = torch.float16 if self.cfg.half else torch.float32
        iterator = iter(train_data)
        for i in range(batches):
            with record_function(f"--training:get_data--"):
                data, targets, id = next(iterator)
                if self.cfg.channels_last:
                    data = data.to(self.device, dtype, memory_format=torch.channels_last, non_blocking=True)
                else:
                    data = data.to(self.device, dtype, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if transform:
                    data = transform(data)
            with record_function(f"--training:forward--"):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.cfg.half):
                    pred = self.model(data)
                    loss = self.loss_fn(pred, targets)
                del data, targets
                train_data.release(id)
            with record_function(f"--training:backward--"):
                self.scaler.scale(loss).backward()
            with record_function(f"--training:update_loss--"):
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                loss_val = float(loss)
                if not math.isnan(loss_val):
                    self.batch_loss[i] = loss_val
                    train_loss += (loss_val - train_loss) / (i+1)
            sys.stdout.write(f"train batch {i+1} / {batches} : loss={train_loss:.3f}  \r")
            if should_stop and should_stop(train=True):
                raise RunInterrupted()
        return train_loss, self.batch_loss

    def test(self, test_data: Loader, transform: Transforms | None = None, calc_top5: bool = False,
             should_stop: Callable | None = None) -> tuple[float, float, float | None]:
        """Calculate loss and accuracy against the test set - returns test loss, accuracy and optionally top5 accuracy"""
        with record_function(f"--testing--"):
            self.model.eval()
            if len(self.predict) != test_data.nitems:
                self.predict = torch.zeros(test_data.nitems, dtype=torch.int64)
            test_loss = 0.0
            accuracy = 0.0
            top5_accuracy = 0.0 if calc_top5 else None
            samples = 0
            batches = len(test_data)
            dtype = torch.float16 if self.cfg.half else torch.float32
            with torch.no_grad():
                for i, (data, targets, id) in enumerate(test_data):
                    data = data.to(self.device, dtype, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                    if transform:
                        data = transform(data)
                    if self.cfg.channels_last:
                        data = data.to(memory_format=torch.channels_last)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.cfg.half):
                        pred = self.model(data)
                        loss = self.loss_fn(pred, targets)
                    nitems = len(targets)
                    outputs = pred.argmax(1)
                    correct = int((outputs == targets).type(torch.float).sum())
                    accuracy += ((correct/nitems) - accuracy) / (i+1)
                    if top5_accuracy is not None:
                        top5_correct = calc_topn_correct(pred, targets, k=5)
                        top5_accuracy += ((top5_correct/nitems) - top5_accuracy) / (i+1)
                    del data, targets
                    test_data.release(id)
                    self.predict[samples: samples+nitems] = outputs
                    loss_val = float(loss)
                    if not math.isnan(loss_val):
                        test_loss += (loss_val - test_loss) / (i+1)
                    samples += nitems
                    sys.stdout.write(f"test batch {i+1} / {batches} : loss={test_loss:.3f}    \r")
                    if should_stop and should_stop():
                        raise RunInterrupted()
            return test_loss, accuracy, top5_accuracy

    def step(self, stats: Stats) -> None:
        """If scheduler is defined then call it's step function"""
        if self.scheduler:
            if self.metric:
                self.scheduler.step(getattr(stats, self.metric)[-1])
            else:
                self.scheduler.step()

    def should_stop(self, stats: Stats) -> bool:
        """Returns True if hit stopping condition"""
        stop = False
        if self.stopper is not None:
            stop = self.stopper.step(stats)
        if stats.current_epoch >= self.epochs:
            return True
        else:
            return stop


def calc_topn_correct(pred: Tensor, targets: Tensor, k: int) -> int:
    """Get number of predictions where one of top k classes matches the target"""
    _, y_pred = pred.topk(k=k, dim=1)
    y_pred = y_pred.t()
    target = targets.view(1, -1).expand_as(y_pred)
    correct = (y_pred == target).reshape(-1).float()
    return int(correct.sum(dim=0, keepdim=True))


def calc_initial_stats(cfg: Config, model: nn.Module, test_data: Dataset, device: str = "cpu") -> Stats:
    """Do test run to calc stats for pre-trained model"""
    log.info(f"calc initial test stats for model  device={device}")
    stats = Stats()
    trainer = Trainer(cfg, model, device=device)
    is_cuda = device.startswith("cuda")
    test_loader = cfg.dataloader("test", pin_memory=is_cuda)
    test_loader.start(test_data)
    calc_top5 = (len(test_data.classes) >= 100)
    test_loss, test_accuracy, top5_accuracy = trainer.test(test_loader, test_data.transform, calc_top5=calc_top5)
    if top5_accuracy is not None:
        log.info(f"Test loss: {test_loss:.3f} accuracy: {test_accuracy:.1%} top5 accuracy: {top5_accuracy:.1%}")
    else:
        log.info(f"Test loss: {test_loss:.3f} accuracy: {test_accuracy:.1%}")
    stats.update_train(0, 0, torch.tensor([]))
    stats.update_test(test_loss, test_accuracy, top5_accuracy)
    stats.predict = trainer.predict
    test_loader.shutdown()
    return stats


def mean(vals: list[float], i: int, avg: int) -> float:
    return statistics.mean(vals[max(i+1-avg, 0):i+1])
