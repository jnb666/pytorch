import importlib.resources
import json
import logging as log
import math
import multiprocessing as mp
import os
import platform
import time
from dataclasses import dataclass, field
from functools import reduce
from os import path
from typing import Any, Callable

import numpy as np
import pyqtgraph as pg  # type: ignore
import PySide6.QtCore as QtCore
import PySide6.QtWidgets as qw
import torch
import torch.nn.functional as F
from PySide6.QtCore import QRect, QSize, Qt, QTimer, QtMsgType
from PySide6.QtGui import QColor, QIntValidator
from torch import Tensor, nn

from .config import Config
from .dataset import Dataset
from .loader import Loader
from .trainer import Stats
from .utils import InvalidConfigError, pformat

window_width = 1067
window_height = 800
window_ypos = 50

font_size = 11 if platform.system() == "Linux" else 14
small_font_size = 10 if platform.system() == "Linux" else 13
tiny_font_size = 9 if platform.system() == "Linux" else 11
bgcolor = "#111"
menu_bgcolor = "#333"
fgcolor = "w"
spacing = 15
margins = 4
layer_list_width = 150
hist_bins = 100
config_file = path.expanduser("~/.pytorch_gui")


def _messageHandler(msg_type, context, message):
    if msg_type == QtMsgType.QtDebugMsg or msg_type == QtMsgType.QtInfoMsg or msg_type == QtMsgType.QtWarningMsg:
        return
    log.debug(f"Qt msg: {msg_type}: {message}")


def init_gui() -> qw.QApplication:
    """Create a new application and set theme"""
    app = qw.QApplication([])
    QtCore.qInstallMessageHandler(_messageHandler)
    fnt = app.font()
    fnt.setPointSize(font_size)
    app.setFont(fnt)
    pg.setConfigOption("background", bgcolor)
    pg.setConfigOption("foreground", fgcolor)
    pg.setConfigOption("antialias", True)
    pg.setConfigOption("imageAxisOrder", "row-major")
    return app


@dataclass
class Options:
    width: int = window_width
    height: int = window_height
    xpos: int = -1
    ypos: int = -1
    image_page: dict[str, int] = field(default_factory=dict)
    image_class: dict[str, int] = field(default_factory=dict)
    image_errors: dict[str, bool] = field(default_factory=dict)
    image_transformed:  dict[str, bool] = field(default_factory=dict)
    activation_index: dict[str, int] = field(default_factory=dict)
    activation_layers: dict[str, list[int]] = field(default_factory=dict)
    histogram_layers: dict[str, list[int]] = field(default_factory=dict)

    def dump(self) -> str:
        return json.dumps(self.__dict__)

    def load(self) -> "Options":
        try:
            with open(config_file, encoding="utf-8") as f:
                self.__dict__.update(json.load(f))
        except FileNotFoundError:
            pass
        return self

    def save(self) -> None:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f)


class MainWindow(qw.QWidget):
    """Main GUI window with select list on the left and content area stacked widget on the right

    Args:
        loader:Loader               loader class which defines data access methods
        sender:Callable             function to send command via 0MQ (optional)
        model:str                   initial model to load (optional)

    Properties:
        cfg:Config                  current config
        model:nn.Sequential         network model
        data:Dataset                test data
        transform:nn.Module         transforms (optional)
        cmd_menu:CommandMenu        top command menu bar
        img_menu:ImageMenu          top image menu bar
        stats:Stats                 latest loaded stats
        epochs:int                  estimated number of epochs
    """

    def __init__(self, loader: Loader, sender: Callable | None = None, model: str = "", running: bool = False):
        super().__init__()
        log.debug(f"new MainWindow: model={model}")
        self.cfg: Config | None = None
        self.data: Dataset | None = None
        self.stats = Stats()
        self.predict = None
        self.opts = Options().load()
        self.loader = loader
        self.cfg_menu = ConfigMenu(self, model, sender)
        self.img_menu = ImageMenu(self.opts)
        self.img_menu.hide()
        self._build()
        rows = qw.QVBoxLayout()
        rows.setSpacing(0)
        rows.setContentsMargins(0, 0, 0, 0)
        rows.addWidget(self.cfg_menu)
        rows.addWidget(self.content)
        rows.addWidget(self.img_menu)
        cols = qw.QHBoxLayout()
        cols.setSpacing(0)
        cols.setContentsMargins(0, 0, 0, 0)
        cols.addWidget(self.menu, 1)
        cols.addLayout(rows, 9)
        self.setLayout(cols)
        self._set_size()

    @property
    def epochs(self):
        if len(self.stats.xrange) == 2:
            return self.stats.xrange[1]
        elif self.cfg:
            return self.cfg.epochs
        else:
            return 100

    def _set_size(self) -> None:
        if self.opts.xpos >= 0 and self.opts.ypos >= 0:
            self.setGeometry(self.opts.xpos, self.opts.ypos, self.opts.width, self.opts.height)
        else:
            self.resize(self.opts.width, self.opts.height)

    def resizeEvent(self, ev):
        self.opts.width = ev.size().width()
        self.opts.height = ev.size().height()
        self.opts.save()

    def moveEvent(self, ev):
        self.opts.xpos = ev.pos().x()
        self.opts.ypos = ev.pos().y()
        self.opts.save()

    def _build(self):
        self.pages = [
            StatsPlots(self),
            Heatmap(self),
            ImageViewer(self),
            Activations(self),
            Histograms(self),
            ConfigLabel(),
        ]
        self.content = qw.QStackedWidget()
        self.menu = qw.QListWidget()
        for i, name in enumerate(["stats", "heatmap", "images", "activations", "histograms", "config"]):
            self.content.addWidget(self.pages[i])
            self.menu.addItem(list_item(name, center=True, min_height=50))
        self.menu.currentItemChanged.connect(self._select_page)
        self.menu.setCurrentRow(0)

    def update_config(self, name: str, running: bool = False) -> None:
        """Load new model definition"""
        log.debug(f"load_config: {name}")
        try:
            self.cfg, self.data, self.transform = self.loader.load_config(name)
        except InvalidConfigError as err:
            self.set_error(str(err))
            return
        self.setWindowTitle(f"{self.cfg.name} v{self.cfg.version}")
        self.img_menu.set_classes(self.data.classes)
        for page in self.pages:
            page.update_config(self.cfg)
        self.data.reset()
        self.img_menu.update_config(self.cfg)
        self.cfg_menu.update_config(self.cfg, self.stats, self.loader.get_models())
        self.update_stats()

    def set_error(self, err: str) -> None:
        log.error(f"set_error: {err}")
        self.cfg_menu.set_error(err)
        name = self.cfg_menu.model_select.currentText()
        data = self.loader.load_model(name)
        self.pages[-1].set_text(data)

    def update_stats(self) -> None:
        """Refresh GUI with updated stats and model weights"""
        if not self.cfg:
            return
        self.loader.load_stats(self.stats)
        self.cfg_menu.update_stats(self.stats)
        if len(self.stats.predict):
            self.predict = torch.clone(self.stats.predict)
        index = self.content.currentIndex()
        self.pages[index].update_stats()
        log.debug(f"updated stats: {self.cfg.name} epoch={self.stats.current_epoch} running={self.stats.running}")

    def _select_page(self, item):
        name = item.text()
        index = self.menu.indexFromItem(item).row()
        log.debug(f"select page {index}: {name}")
        if name == "activations" or name == "images":
            self.img_menu.register(self.pages[index])
        else:
            self.img_menu.register(None)
        self.content.setCurrentIndex(index)
        self.pages[index].update_stats()


class ConfigMenu(qw.QWidget):
    """Main top menu bar"""

    def __init__(self, main: MainWindow, model: str = "", sender: Callable | None = None):
        super().__init__()
        set_background(self, menu_bgcolor)
        self.cmd_menu = CommandMenu(main, sender)
        self.models = main.loader.get_models()
        self.model_select = qw.QComboBox()
        self.model_select.addItems(self.models)
        self.model_select.setCurrentText(model)
        self.model_select.currentIndexChanged.connect(self._select_model)  # type: ignore
        self._error_label = qw.QLabel()
        self._errors = self._error_box(self._error_label)
        self._errors.hide()
        cols = qw.QHBoxLayout()
        cols.setContentsMargins(margins, margins, margins, margins)
        cols.addSpacing(spacing)
        cols.addWidget(self.model_select)
        cols.addSpacing(spacing)
        cols.addWidget(self.cmd_menu)
        cols.addWidget(self._errors)
        cols.addSpacing(spacing)
        self.setLayout(cols)
        self.update_stats(main.stats)

    def set_error(self, err: str) -> None:
        self._error_label.setText(err)
        if err:
            self._errors.show()
            self.cmd_menu.hide()
        else:
            self._errors.hide()
            self.cmd_menu.show()

    def _error_box(self, label):
        # force error label to expand to use all available space without resizing the window
        label.setSizePolicy(qw.QSizePolicy.Expanding, qw.QSizePolicy.Fixed)
        label.setMargin(margins)
        scroll = qw.QScrollArea()
        scroll.setSizePolicy(qw.QSizePolicy.Expanding, qw.QSizePolicy.Fixed)
        scroll.setViewportMargins(0, 0, 0, 0)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(label)
        scroll.setFixedHeight(label.height()+2*margins)
        return scroll

    def update_config(self, cfg: Config, stats: Stats, models: list[str]) -> None:
        """Called when new config file is loaded or config files are changed"""
        if models != self.models:
            log.debug("update model list")
            model = self.model_select.currentText()
            self.model_select.currentIndexChanged.disconnect()  # type: ignore
            self.model_select.clear()
            self.model_select.addItems(models)
            self.model_select.setCurrentText(model)
            self.model_select.currentIndexChanged.connect(self.cmd_menu.select_model)  # type: ignore
        self.cmd_menu.update_config(cfg, stats)
        self.set_error("")

    def update_stats(self, stats: Stats) -> None:
        """Called after loading config and after each epoch"""
        if self.cmd_menu.send:
            self.model_select.setEnabled(not stats.running)
        self.cmd_menu.update_stats(stats)
        self.set_error("")

    def _select_model(self, index):
        name = self.model_select.itemText(index)
        QTimer.singleShot(0, self.cmd_menu.model_loader(name))


class CommandMenu(qw.QWidget):
    """Top menu bar used to control the training loop"""

    def __init__(self, main: MainWindow, sender: Callable | None = None):
        def send(*args):
            status, err = sender(*args)
            if status == "error":
                main.set_error(err)

        super().__init__()
        self.main = main
        self.send = send if sender else None
        self.in_error = False
        self._epoch = qw.QLabel()
        self._elapsed = qw.QLabel()
        self._learn_rate = qw.QLabel()
        if self.send:
            self.start = qw.QPushButton("start")
            self.start.setEnabled(False)
            self.start.clicked.connect(lambda: self.send("start", 0, False))  # type: ignore
            self.stop = qw.QPushButton("stop")
            self.stop.setEnabled(False)
            self.stop.clicked.connect(lambda: self.send("pause", True))  # type: ignore
            self.resume_from = qw.QComboBox()
            self.resume_from.setMinimumWidth(60)
            self.resume = qw.QPushButton("resume")
            self.resume.setEnabled(False)
            self.resume.clicked.connect(self._resume)  # type: ignore
            self.max_epoch = qw.QLineEdit()
            self.max_epoch.setFixedWidth(50)
            self.max_epoch.setValidator(QIntValidator(0, 999))
            self.max_epoch.editingFinished.connect(lambda: self.send("max_epoch", int(self.max_epoch.text())))  # type: ignore
        cols = qw.QHBoxLayout()
        cols.setContentsMargins(margins, margins, margins, margins)
        if self.send:
            cols.addWidget(self.start)
            cols.addSpacing(spacing)
            cols.addWidget(self.stop)
            cols.addSpacing(spacing)
            cols.addWidget(self.resume_from)
            cols.addWidget(self.resume)
            cols.addSpacing(spacing)
            cols.addWidget(qw.QLabel("max epoch"))
            cols.addWidget(self.max_epoch)
        cols.addSpacing(spacing)
        cols.addWidget(self._epoch)
        cols.addSpacing(spacing)
        cols.addWidget(self._elapsed)
        cols.addSpacing(spacing)
        cols.addWidget(self._learn_rate)
        self.setLayout(cols)

    def update_config(self, cfg: Config, stats: Stats) -> None:
        """Called when new config file is loaded or config files are changed"""
        if self.send:
            self._update_epoch_list(stats.current_epoch)
            self.max_epoch.setText(str(cfg.epochs))
        self.enable_start(True)

    def enable_start(self, on: bool) -> None:
        self.start.setEnabled(on)
        self.resume.setEnabled(on)
        self.stop.setEnabled(False)

    def update_stats(self, stats: Stats) -> None:
        """Called after loading config and after each epoch"""
        self._epoch.setText(f"epoch: {stats.current_epoch}")
        self._elapsed.setText(f"elapsed: {stats.elapsed_total()}")
        if len(stats.learning_rate):
            self._learn_rate.setText(f"lr: {stats.learning_rate[-1]:.4}")
        if self.send:
            self.start.setEnabled(not stats.running)
            self.resume.setEnabled(not stats.running)
            self.stop.setEnabled(stats.running)
            if not stats.running:
                self._update_epoch_list(stats.current_epoch)

    def _update_epoch_list(self, epoch: int):
        epochs = self.main.loader.get_epochs()
        self.resume_from.clear()
        self.resume_from.addItems([str(i) for i in epochs])
        self.resume_from.setCurrentText(str(epoch))

    def _resume(self):
        epoch = int(self.resume_from.currentText())
        if epoch == self.main.stats.current_epoch:
            self.send("pause", False)
        else:
            self.send("start", epoch, False)

    def model_loader(self, name):
        if self.send:
            def loader():
                self.send("load", name)
        else:
            def loader():
                self.main.update_config(name)
                self.main.update_stats()
        return loader


class ConfigLabel(qw.QScrollArea):
    """ConfigLabel is a scrollable Qlabel with fixed format text and top alignment"""

    def __init__(self):
        super().__init__()
        set_background(self, bgcolor)
        self.setWidgetResizable(True)
        label = qw.QLabel()
        label.setAlignment(Qt.AlignTop)  # type: ignore
        label.setWordWrap(True)
        fnt = label.font()
        fnt.setPointSize(small_font_size)
        label.setFont(fnt)
        self.content = label
        self.setWidget(self.content)

    def set_text(self, text: str) -> None:
        """Update the label text"""
        self.content.setText(f"<pre>{text}</pre>")

    def update_config(self, cfg: Config) -> None:
        """Called when new config file is loaded"""
        self.set_text(cfg.text)

    def update_stats(self) -> None:
        pass


class StatsPlots(qw.QWidget):
    """StatsPlots widget shows the loss and accuracy plots

    Args:
        main            reference to main window
    """

    def __init__(self, main: MainWindow):
        super().__init__()
        self.main = main
        self.table = pg.TableWidget(editable=False, sortable=False)
        self.plots = pg.GraphicsLayoutWidget(border=True)
        self._draw_plots()
        cols = qw.QHBoxLayout()
        cols.setContentsMargins(0, 0, 0, 0)
        cols.addWidget(self.table, 2)
        cols.addWidget(self.plots, 3)
        self.setLayout(cols)

    def _draw_plots(self) -> None:
        self.plots.clear()
        self.plot1 = self.plots.addPlot(row=0, col=0)
        self.plot1.showGrid(x=True, y=True, alpha=0.75)
        self.plot2 = self.plots.addPlot(row=1, col=0)
        self.plot2.showGrid(x=True, y=True, alpha=0.75)
        init_plot(self.plot1, ylabel="cross entropy loss", legend=(0, 1),
                  xrange=[1, self.main.epochs], yrange=[0, 1], mouse=True)
        init_plot(self.plot2, xlabel="epoch", ylabel="accuracy",
                  xrange=[1, self.main.epochs], yrange=[0, 1], mouse=True)
        self.plot2.getViewBox().setXLink(self.plot1.getViewBox())
        self.line1: list[pg.PlotDataItem] = []
        self.line2: list[pg.PlotDataItem] = []

    def _update_plots(self, stats: Stats) -> None:
        """draw line plots"""
        y1 = [stats.train_loss, stats.test_loss]
        if stats.valid_loss:
            y1.append(stats.valid_loss)
        y2 = [stats.test_accuracy]
        if stats.valid_accuracy:
            y2.append(stats.valid_accuracy)
        if stats.valid_accuracy_avg:
            y2.append(stats.valid_accuracy_avg)
        update_range(self.plot1, stats.current_epoch+1, y1)
        update_range(self.plot2, stats.current_epoch+1, y2)
        if len(self.line1) == len(y1) and len(self.line2) == len(y2):
            log.debug("updating plot lines")
            update_lines(self.line1, stats.epoch, y1)
            update_lines(self.line2, stats.epoch, y2)
        else:
            log.debug("adding lines to plots")
            maxy = max(stats.train_loss[0], stats.test_loss[0])
            self.plot1.setYRange(0, maxy, padding=0)
            self.plot2.setYRange(stats.test_accuracy[0], 1, padding=0)
            self.line1.clear()
            self.line1.append(add_line(self.plot1, stats.epoch, stats.train_loss, color="r", name="training"))
            self.line1.append(add_line(self.plot1, stats.epoch, stats.test_loss, color="g", name="testing"))
            if len(stats.valid_loss):
                self.line1.append(add_line(self.plot1, stats.epoch, stats.valid_loss, color="y", name="validation"))
            self.line2.clear()
            self.line2.append(add_line(self.plot2, stats.epoch, stats.test_accuracy, color="g", name="testing"))
            if len(stats.valid_accuracy):
                self.line2.append(add_line(self.plot2, stats.epoch, stats.valid_accuracy, color="y", name="validation"))
            if len(stats.valid_accuracy_avg):
                self.line2.append(add_line(self.plot2, stats.epoch, stats.valid_accuracy_avg, color="y", dash=True))

    def _update_table(self, stats: Stats) -> None:
        """update stats table"""
        self.table.setFormat("%.3f", column=0)
        self.table.setFormat("%.3f", column=1)
        self.table.setFormat("%.1f", column=2)
        if len(stats.valid_loss):
            self.table.setFormat("%.3f", column=3)
            self.table.setFormat("%.1f", column=4)
        data = stats.table_data()
        data.reverse()
        cols = [(name, float) for name in stats.table_columns()]
        self.table.setData(np.array(data, dtype=cols))
        self.table.setVerticalHeaderLabels([str(i) for i in reversed(stats.epoch)])

    def update_config(self, cfg: Config) -> None:
        """Called when new config file is loaded"""
        log.debug(f"update stats config: {cfg.name} max_epoch={cfg.epochs}")
        self._draw_plots()
        self.plot1.setXRange(0, cfg.epochs, padding=0)
        self.plot2.setXRange(0, cfg.epochs, padding=0)

    def update_stats(self) -> None:
        """Refresh GUI with updated stats"""
        stats = self.main.stats
        log.debug(f"update stats: epoch={stats.current_epoch} of xrange={stats.xrange}")
        self._update_table(stats)
        if stats.current_epoch > 0:
            self._update_plots(stats)
        else:
            self._draw_plots()


class Heatmap(pg.GraphicsLayoutWidget):
    """Heatmap shows a heatmap plot with correlation between the labels and the predictions

    Args:
        main            reference to main window
        cmap: string   color map to use
    """

    def __init__(self, main: MainWindow, cmap="CET-L6"):
        super().__init__(border=True)
        self.main = main
        self.cmap = cmap
        self.plot = self.addPlot()

    def _draw_plot(self, map: np.ndarray, classes: list[str]):
        if not self.main.data:
            return
        nclasses = len(classes)
        init_plot(self.plot, xlabel="target", ylabel="prediction", xrange=[0, nclasses], yrange=[0, nclasses])
        set_ticks(self.plot.getAxis("bottom"), classes)
        set_ticks(self.plot.getAxis("left"), classes)
        img = pg.ImageItem(map)
        img.setColorMap(self.cmap)
        self.plot.addItem(img)
        nitems = len(self.main.data.targets)
        for y in range(nclasses):
            for x in range(nclasses):
                val = map[x, y]
                col = "w" if val < 0.5*nitems/nclasses else "k"
                if val > 0:
                    add_label(self.plot, x, y, val, col)

    def update_config(self, cfg: Config) -> None:
        """Called when new config file is loaded"""
        self.plot.clear()

    def update_stats(self) -> None:
        """Refresh GUI with updated stats"""
        if not self.main.data or self.main.predict is None:
            return
        targets = self.main.data.targets.cpu().numpy()
        predict = np.array(self.main.predict.cpu())
        classes = self.main.data.classes
        map, _, _ = np.histogram2d(targets, predict, bins=len(classes))
        log.debug(f"== heatmap: ==\nclasses={classes}\n{map}")
        self._draw_plot(map, classes)


class ImageMenu(qw.QWidget):
    """ImageMenu is the top menu bar shown on the Images and Activations screen

    Properties:
        label: QLabel             text label to identify image or page number
        info: QLabel              text label with exta info
        target_class: int         show just this class
        errors_only: bool         only show errors
        transformed: bool         apply image transform
    """

    def __init__(self, opts: Options):
        super().__init__()
        set_background(self, menu_bgcolor)
        self._listener = None
        self.opts = opts
        self.name = ""
        self.label = qw.QLabel("")
        self.label.setMinimumWidth(120)
        self.info = qw.QLabel("")
        self.target_class = -1
        self.errors_only = False
        self.transformed = False
        layout = self._build()
        self.setLayout(layout)

    def update_config(self, cfg: Config) -> None:
        self.name = cfg.name
        try:
            self.target_class = self.opts.image_class[cfg.name]
            self._combo.setCurrentIndex(self.target_class+1)
            self.errors_only = self.opts.image_errors[cfg.name]
            self._errors.setChecked(self.errors_only)
            self.transformed = self.opts.image_transformed[cfg.name]
            self._trans.setChecked(self.transformed)
        except KeyError:
            pass

    def _build(self):
        prev = qw.QPushButton("<< prev")
        prev.clicked.connect(self._prev)
        next = qw.QPushButton("next >>")
        next.clicked.connect(self._next)
        self._combo = qw.QComboBox()
        self._combo.addItems(["all classes"])
        self._combo.currentIndexChanged.connect(self._filter_class)
        self._errors = qw.QCheckBox("errors only")
        self._errors.stateChanged.connect(self._filter_errors)
        self._trans = qw.QCheckBox("transform")
        self._trans.stateChanged.connect(self._transform)
        cols = qw.QHBoxLayout()
        cols.setContentsMargins(margins, margins, margins, margins)
        cols.addSpacing(spacing)
        cols.addWidget(self.label)
        cols.addSpacing(spacing)
        cols.addWidget(prev)
        cols.addWidget(next)
        cols.addSpacing(spacing)
        cols.addWidget(qw.QLabel("show"))
        cols.addWidget(self._combo)
        cols.addSpacing(spacing)
        cols.addWidget(self._errors)
        cols.addSpacing(spacing)
        cols.addWidget(self._trans)
        cols.addSpacing(spacing)
        cols.addWidget(self.info)
        cols.addStretch()
        return cols

    def set_classes(self, classes: list[str]) -> None:
        """Assign list of classes"""
        self._combo.clear()
        self._combo.addItems(["all classes"] + classes)
        self.target_class = -1
        self._errors.setChecked(False)
        self.errors_only = False
        self._trans.setChecked(False)

    def register(self, listener) -> None:
        """Called when a new content page is loaded to reset the callbacks"""
        self._listener = listener
        if listener:
            self.show()
        else:
            self.hide()

    def _prev(self):
        if self._listener is not None:
            self._listener.prev()

    def _next(self):
        if self._listener is not None:
            self._listener.next()

    def _filter_class(self, index):
        self.target_class = index-1
        if self._listener is not None:
            self._listener.filter()
        self.opts.image_class[self.name] = self.target_class
        self.opts.save()

    def _filter_errors(self, state):
        self.errors_only = self._errors.isChecked()
        if self._listener is not None:
            self._listener.filter()
        self.opts.image_errors[self.name] = self.errors_only
        self.opts.save()

    def _transform(self, state):
        self.transformed = self._trans.isChecked()
        if self._listener is not None:
            self._listener.set_transformed(self.transformed)
        self.opts.image_transformed[self.name] = self.transformed
        self.opts.save()


class ImageViewer(pg.GraphicsLayoutWidget):
    """ImageViewer displays a grid of images with options to filter by class or errors

    Args:
        main: MainWindow       reference to main window
        rows, cols: int        grid size

    Properties:
        page: int              page number
        images_per_page: int   grid size
        transformed: bool      whether transform is enabled
    """

    def __init__(self, main: MainWindow, rows: int = 5, cols: int = 7):
        super().__init__(border=True)
        self.main = main
        self.menu = main.img_menu
        self.page: int = 0
        self.images_per_page = rows*cols
        self.plots = []
        for row in range(rows):
            for col in range(cols):
                index = row*cols + col
                p = self.addPlot(row, col)
                p.showAxis("top", False)
                p.showAxis("left", False)
                p.showAxis("bottom", False)
                p.getViewBox().setMouseEnabled(x=False, y=False)
                self.plots.append(p)

    @ property
    def pages(self) -> int:
        if self.main.data:
            return 1 + (self.main.data.nitems-1) // self.images_per_page
        else:
            return 0

    def prev(self) -> None:
        """callback from ImageMenu """
        self.page -= 1
        if self.page < 0:
            self.page = max(self.pages-1, 0)
        self._update()

    def next(self) -> None:
        """callback from ImageMenu """
        self.page += 1
        if self.page >= self.pages:
            self.page = 0
        self._update()

    def filter(self) -> None:
        """callback from ImageMenu """
        if self.main.data and self.main.predict is not None:
            self.page = 0
            self.main.data.filter(self.main.predict, self.menu.target_class, self.menu.errors_only)
            self._update()

    def set_transformed(self, on: bool) -> None:
        """callback from ImageMenu """
        log.debug(f"set transformed => {on}")
        self._update()

    def update_config(self, cfg: Config) -> None:
        """Called when new config file is loaded"""
        try:
            self.page = self.main.opts.image_page[cfg.name]
        except KeyError:
            self.page = 0

    def update_stats(self) -> None:
        """Called to update stats data after each epoch"""
        if self.main.data and self.main.predict is not None and self.menu.errors_only:
            self.main.data.filter(self.main.predict, self.menu.target_class, self.menu.errors_only)
        self._update()

    def _update(self):
        if not self.main.data:
            return
        if self.page >= self.pages:
            self.page = 0
        log.debug(f"update images page={self.page+1} / {self.pages}")
        self.menu.label.setText(f"Page {self.page+1} of {self.pages}")
        self.menu.info.setText("")
        start = self.page*self.images_per_page
        for i, p in enumerate(self.plots):
            p.clear()
            p.setTitle(self._title(start+i))
            data = self._image(start + i)
            if data is not None:
                p.setXRange(0, data.shape[1], padding=0)
                p.setYRange(0, data.shape[0], padding=0)
                p.addItem(pg.ImageItem(data))
        self.main.opts.image_page[self.main.cfg.name] = self.page
        self.main.opts.save()

    def _image(self, i: int) -> np.ndarray | None:
        if not self.main.data:
            return None
        ds = self.main.data
        if i >= ds.nitems:
            return None
        img = ds.get(i, ds.data)
        img = img.view(1, *img.shape)
        if self.menu.transformed and self.main.transform:
            img = self.main.transform(img)
        return to_image(img)

    def _title(self, i: int) -> str:
        if not self.main.data:
            return ""
        ds = self.main.data
        s = ""
        if i < ds.nitems:
            s = f"{ds.indexOf(i)}:"
            if self.main.predict is not None:
                pred = int(ds.get(i, self.main.predict))
                s += ds.classes[pred]
        if platform.system() == "Linux":
            return f"<pre><font size=2>{s:<16}</font></pre>"
        else:
            return f"<pre>{s:<16}</pre>"


class LayerList(qw.QListWidget):
    """LayerList is a multiselect list widget with the list of network layers

    Attributes:
        states: list[bool]      flag indicating if each layer is shown
    """

    def __init__(self):
        super().__init__()
        self.setSelectionMode(qw.QAbstractItemView.MultiSelection)  # type: ignore
        self.setFixedWidth(layer_list_width)
        self.states = []

    def set_layers(self, layers: list[str]):
        """Define list of layers"""
        self.clear()
        self.states = [False] * len(layers)
        for i, name in enumerate(layers):
            self.addItem(list_item(f"{i}: {name}", min_height=30))

    def set_selected(self, i: int, on: bool) -> None:
        self.states[i] = on
        self.item(i).setSelected(on)

    def selected(self) -> list[int]:
        return [ix for ix, flag in enumerate(self.states) if flag]


class Histograms(qw.QWidget):
    """The Histograms widgets displays histograms of the activation intensity for the selected layers.

    Args:
        main: MainWindow       reference to main window

    Attributes:
        layers: LayerList            list of layers to select from
        plots: GraphicsLayoutWidget  content widget holding the plots
    """

    def __init__(self, main: MainWindow):
        super().__init__()
        self.main = main
        self.model_name = ""
        self.histograms: dict[int, Any] = {}
        self.plots = pg.GraphicsLayoutWidget(border=True)
        self.layers = LayerList()
        self.layers.itemClicked.connect(self._select_layer)  # type: ignore
        cols = qw.QHBoxLayout()
        cols.setContentsMargins(0, 0, 0, 0)
        cols.addWidget(self.plots)
        cols.addWidget(self.layers)
        self.setLayout(cols)

    def update_config(self, cfg: Config) -> None:
        """Called when new config file is loaded"""
        self.histograms.clear()
        self.model_name = cfg.name
        names = cfg.layer_names
        self.layers.set_layers(names)
        try:
            for i in self.main.opts.histogram_layers[cfg.name]:
                self.layers.set_selected(i, True)
        except KeyError:
            for i, name in enumerate(names):
                if i == 0 or "Conv" in name or "Linear" in name:
                    self.layers.set_selected(i, True)
            self.main.opts.histogram_layers[cfg.name] = self.layers.selected()
            self.main.opts.save()

    def _select_layer(self, item):
        index = self.layers.indexFromItem(item).row()
        self.layers.states[index] = not self.layers.states[index]
        log.debug(f"Histograms: toggle {item.text()} index={index} {self.layers.states[index]}")
        self.get_histograms(self.layers.selected())
        self._update_plots()
        self.main.opts.histogram_layers[self.model_name] = self.layers.selected()
        self.main.opts.save()

    def update_stats(self) -> None:
        """Called to update stats data after each epoch"""
        self.histograms.clear()
        self.get_histograms(self.layers.selected())
        self._update_plots()

    def get_histograms(self, layers: list[int]) -> None:
        """Get activation histogram data from loader"""
        if not self._hist_cached(layers):
            self.histograms.update(self.main.loader.get_histograms(layers))

    def _hist_cached(self, layers: list[int]) -> bool:
        for ix in layers:
            try:
                _ = self.histograms[ix]
            except KeyError:
                return False
        return True

    def _update_plots(self):
        self.plots.clear()
        nplots = len(self.layers.selected())
        if nplots == 0 or not self.histograms:
            return
        rows, cols = calc_grid(nplots)
        n = 0
        for ix in self.layers.selected():
            hist, x0, x1 = self.histograms[ix]
            width = (x1 - x0) / hist_bins
            log.debug(f"layer {ix} hist: orig={x0:.3} width={width:.3}")
            xpos = np.arange(x0, x1, width)
            height = hist.numpy()
            p = self.plots.addPlot(row=n//cols, col=n % cols)
            init_plot(p, xrange=(x0, x1), yrange=(0, np.max(height)))
            p.showAxis("left", False)
            fnt = self.font()
            fnt.setPointSize(tiny_font_size)
            p.getAxis("bottom").setTickFont(fnt)
            p.addItem(pg.BarGraphItem(x0=xpos[:hist_bins], width=width, height=height, brush="g"))
            p.setTitle(self.layers.item(ix).text())
            n += 1


class Activations(qw.QWidget):
    """The Activations widgets displays image plots with the activation intensity for the selected layers.

    Args:
        main: MainWindow             reference to main window
        cmap: str                    color map name

    Attributes:
        layers: LayerList            list of layers to select from
        index: int                   current image index
        plots: PlotGrids             content widget holding the plots
    """

    def __init__(self, main: MainWindow, cmap: str | None = "CET-L6"):
        super().__init__()
        self.main = main
        self.model_name = ""
        self.menu = main.img_menu
        self.cmap = cmap
        self.activations: dict[int, dict[int, Tensor]] = {}
        self.index: int = 0
        self.plots = PlotGrids(cmap)
        self.layers = LayerList()
        self.layers.itemClicked.connect(self._select_layer)  # type: ignore
        cols = qw.QHBoxLayout()
        cols.setContentsMargins(0, 0, 0, 0)
        cols.addWidget(self.plots)
        cols.addWidget(self.layers)
        self.setLayout(cols)

    def update_config(self, cfg: Config) -> None:
        """Called when new config file is loaded"""
        self.activations.clear()
        self.model_name = cfg.name
        log.debug(f"update activation config for {self.model_name}")
        names = cfg.layer_names
        if self.main.data:
            self.plots.set_model(len(names), self.main.data.classes)
        self.layers.set_layers(names)
        try:
            for i in self.main.opts.activation_layers[cfg.name]:
                self.layers.set_selected(i, True)
        except KeyError:
            enabled = 0
            for i, name in enumerate(names):
                if i == 0 or i == len(names)-1 or (enabled <= 2 and ("Conv" in name or "Linear" in name)):
                    self.layers.set_selected(i, True)
                    enabled += 1
            self.main.opts.activation_layers[cfg.name] = self.layers.selected()
            self.main.opts.save()
        self.index = self.main.opts.activation_index.get(cfg.name, 0)

    @property
    def samples(self) -> int:
        if self.main.data:
            return min(self.main.data.nitems, self.main.data.batch_size)
        else:
            return 0

    def get_activations(self, layers: list[int], index: int) -> dict[int, Tensor]:
        """Get activations tensor for given image"""
        if not self._activ_cached(layers, index):
            self.activations[index].update(self.main.loader.get_activations(layers, index))
        return self.activations[index]

    def _activ_cached(self, layers: list[int], index: int) -> bool:
        try:
            _ = self.activations[index]
        except KeyError:
            self.activations[index] = {}
            return False
        for ix in layers:
            try:
                _ = self.activations[index][ix]
            except KeyError:
                return False
        return True

    def _select_layer(self, item):
        index = self.layers.indexFromItem(item).row()
        self.layers.states[index] = not self.layers.states[index]
        log.debug(f"toggle {item.text()} index={index} {self.layers.states[index]}")
        activations = self.get_activations(self.layers.selected(), self.index)
        self.plots.update_plots(activations, self.layers.states)
        self.main.opts.activation_layers[self.model_name] = self.layers.selected()
        self.main.opts.save()

    def prev(self) -> None:
        """callback from ImageMenu """
        self._step(-1)
        self._update_plots()

    def next(self) -> None:
        """callback from ImageMenu """
        self._step(1)
        self._update_plots()

    def filter(self) -> None:
        """callback from ImageMenu """
        if self.main.data and self.main.predict is not None:
            self.main.data.filter(self.main.predict, self.menu.target_class, self.menu.errors_only)
        self._update_plots()

    def set_transformed(self, on: bool) -> None:
        """callback from ImageMenu """
        pass

    def update_stats(self) -> None:
        """Called to update stats data after each epoch"""
        self.activations.clear()
        if self.main.data and self.main.predict is not None and self.menu.errors_only:
            self.main.data.filter(self.main.predict, self.menu.target_class, self.menu.errors_only)
        self._update_plots()

    def _update_plots(self) -> None:
        if not self.main.data:
            return
        log.debug(f"update activation plots for {self.model_name}:{self.index}")
        activations = self.get_activations(self.layers.selected(), self.index)
        self.menu.label.setText(f"Image {self.effective_index+1} of {self.samples}")
        ds = self.main.data
        tgt = ds.targets[self.index]
        self.menu.info.setText(f"{self.index}: target={ds.classes[tgt]}")
        self.plots.update_plots(activations, self.layers.states)

    @property
    def effective_index(self) -> int:
        if self.main.data and self.main.data.indexes is not None:
            return int(torch.searchsorted(self.main.data.indexes, self.index))
        else:
            return self.index

    def _step(self, offset: int) -> None:
        if self.main.data and self.main.data.indexes is not None:
            ix = int(torch.searchsorted(self.main.data.indexes, self.index))
            ix = min(max(0, ix+offset), len(self.main.data.indexes)-1)
            self.index = int(self.main.data.indexes[ix])
        else:
            self.index = min(max(0, self.index + offset), self.samples)
        self.main.opts.activation_index[self.model_name] = self.index
        self.main.opts.save()


class PlotGrids(qw.QWidget):
    """PlotGrids is the containing widget for the layout with the set of activation layer grids

    Args:
        cmap                         colormap name

    Properties:
        layers: dict[int, PlotGrid]  current set of layers
        layer_state: list[bool]      flag indicating which layers are enabled
    """

    def __init__(self, cmap: str | None = None):
        super().__init__()
        set_background(self, bgcolor)
        self.layer_state: list[bool] = []
        self.layers: dict[int, PlotGrid] = {}
        self.cmap = cmap
        self.classes = None
        self._layout = PlotLayout()
        self.setLayout(self._layout)

    def set_model(self, nlayers: int, classes=None):
        """Called when model is loaded"""
        self.layers = {}
        self.layer_state = [False] * nlayers
        self.classes = classes
        self.table = OutputTable(classes)

    def update_plots(self, activations: dict[int, Tensor], layer_state: list[bool]) -> None:
        """update the grids to display the new activation tensors"""
        self.table.set_outputs(activations[len(self.layer_state)-1][0])
        ids = [i for i, flag in enumerate(layer_state) if flag]
        if layer_state == self.layer_state:
            log.debug(f"update plots: index={ids} {activations.keys()}")
            for i, grid in self.layers.items():
                if layer_state[i]:
                    grid.set_data(activations[i][0].numpy())
                    grid.draw()
            self._layout.resize_table()
        else:
            log.debug(f"draw plots: index={ids} {activations.keys()}")
            self._layout.clear()
            for i, enabled in enumerate(layer_state):
                if enabled:
                    self._layout.addWidget(self._grid(i, activations[i][0].numpy()))
                    empty = False
            self._layout.addWidget(self.table)
            self.layer_state = layer_state.copy()

    def _grid(self, i: int, data: np.ndarray) -> "PlotGrid":
        try:
            grid = self.layers[i]
            grid.set_data(data)
        except KeyError:
            labels = self.classes if i == len(self.layer_state)-1 else None
            is_rgb = (i == 0 and data.shape[0] == 3)
            grid = PlotGrid(data, cmap=self.cmap, xlabels=labels, rgbimage=is_rgb)
            self.layers[i] = grid
        return grid


class OutputTable(qw.QTableWidget):
    """OutputTable is a table with the highest probability output values"""

    def __init__(self, classes: list[str]):
        super().__init__()
        self.setColumnCount(2)
        self.setRowCount(1)
        self.setHorizontalHeaderLabels(["class", "prob%"])
        self.classes = classes

    def _update(self, rows):
        self.clearContents()
        self.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.setItem(i, 0, qw.QTableWidgetItem(row[0]))
            self.setItem(i, 1, qw.QTableWidgetItem(row[1]))

    def set_outputs(self, outputs: Tensor):
        log.debug(f"set_outputs: {outputs.shape}")
        prob = F.softmax(outputs.to(device="cpu", dtype=torch.float32), dim=0)
        rows = []
        for i in torch.argsort(prob, descending=True).tolist():
            val = prob[i].item()
            if val < 0.05:
                break
            rows.append((self.classes[i], f"{val:.1%}"))
        self._update(rows)

    def size_hint(self):
        return 215, 25+30*self.rowCount()


class PlotGrid(pg.PlotWidget):
    """PlotGrid widget is a single grid of image plots

    Args:
    data: np.ndarray        tensor with activations
    cmap: str               colormap to use
    xlabels                 if set then use these as list of x axis labels
    rgbimage                if set then consists of R, G, B color image channels

    Properties:
    plots:                  list of plots
    data: np.ndarray        data in [C, H, W] format
    rows, cols: int         grid size
    xlabels                 set if have labels for outputs
    """

    def __init__(self, data: np.ndarray, cmap=None, xlabels=None, rgbimage=False):
        vbox = pg.ViewBox(enableMouse=False, enableMenu=False, defaultPadding=0)
        super().__init__(viewBox=vbox)
        self.rgbimage = rgbimage
        self.xlabels = xlabels
        self.showAxis("left", False)
        if xlabels:
            set_ticks(self.getAxis("bottom"), xlabels)
        else:
            self.showAxis("bottom", False)
        self.set_data(data)
        self.rows, self.cols = calc_grid(self.nplots, square=True)
        self.img = pg.ImageItem(self._reshape())
        if not rgbimage and cmap:
            self.img.setColorMap(cmap)
        self.addItem(self.img)

    def _reshape(self) -> np.ndarray:
        shape = self.data.shape
        if self.rgbimage:
            log.debug(f"PlotGrid transpose rgbimage: data={shape} => {shape[1],shape[2],shape[0]}")
            return self.data.transpose((1, 2, 0))
        else:
            log.debug(f"PlotGrid reshape: data={shape} grid={self.rows}x{self.cols} => {self.height,self.width}")
            reshaped = np.zeros((self.height, self.width))
            w, h = self.img_shape
            for row in range(self.rows):
                for col in range(self.cols):
                    reshaped[row*h:(row+1)*h, col*w:(col+1)*w] = self.data[row*self.cols+col]
            return reshaped

    def draw(self) -> None:
        """Reshape data to rows x cols tiles and update image"""
        self.img.setImage(self._reshape())

    @property
    def nplots(self) -> int:
        return 1 if self.rgbimage else self.data.shape[0]

    @property
    def height(self) -> int:
        return self.rows*self.data.shape[1]

    @property
    def width(self) -> int:
        return self.cols*self.data.shape[2]

    @property
    def img_shape(self) -> tuple[int, int]:
        return self.data.shape[2], self.data.shape[1]

    def set_data(self, data: np.ndarray) -> None:
        """Update data for this set of plots"""
        if data.ndim == 1:
            data = np.reshape(data, (1, 1, data.shape[0]))
        elif data.ndim == 2:
            data = np.flip(data, axis=0)
            data = np.reshape(data, (1, data.shape[0], data.shape[1]))
        elif data.ndim == 3:
            data = np.flip(data, axis=1)
        else:
            raise ValueError(f"Error: layer shape {data.shape} not supported")
        self.data = data

    def set_shape(self, rows: int, cols: int) -> None:
        """Set number of grid rows and columns"""
        self.rows, self.cols = rows, cols

    def expand_width(self, max_factor: float) -> bool:
        """Adjust grid rows and cols expanding width by up to max_factor ratio"""
        updated = False
        rows, cols = self.rows, self.cols
        while rows > 1:
            r, c = calc_grid(self.nplots, rows-1, square=True)
            if c/self.cols <= max_factor:
                rows, cols = r, c
                updated = True
            else:
                break
        if updated:
            self.rows, self.cols = rows, cols
        return updated


class PlotLayout(qw.QLayout):
    """PlotLayout is used to arrange the activation grids in a custom layout

    Args:
        spacing: int       spacing between each grid
        min_size: int      minimum grid height

    Properties:
        x, y, width, height  layout position and size
        items             list of layout items
        min_size: int      minimum grid height
        grid_spacing      spacing between each grid
    """

    def __init__(self, spacing: int = 4, min_size: int = 60):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.min_size = min_size
        self.grid_spacing = spacing
        self._size: list[tuple[int, int]] = []
        self.items: list[qw.QLayoutItem] = []
        self.x: int = 0
        self.y: int = 0
        self.width: int = 800
        self.height: int = 800
        self._changed = False

    def count(self):
        return len(self.items)

    def addItem(self, item):
        self.items.append(item)
        self._changed = True

    def itemAt(self, i):
        if i < len(self.items):
            return self.items[i]
        else:
            return None

    def takeAt(self, i):
        self.changed = True
        return self.items.pop(i)

    def clear(self):
        while self.count() > 0:
            self.takeAt(0).widget().setParent(None)

    def sizeHint(self):
        return QSize(self.width, self.height)

    def widget(self, i):
        return self.items[i].widget()

    def widgets(self):
        return [i.widget() for i in self.items[:-1]]

    @ property
    def nwidgets(self):
        return len(self.items)-1

    def resize_table(self):
        if self.count() == 0:
            return
        table = self.items[-1].widget()
        if not isinstance(table, OutputTable):
            return
        w, h = table.size_hint()
        table.setGeometry(QRect(self.width-w-10, 10, w, h))

    def setGeometry(self, r):
        if self._changed or self.x != r.x() or self.y != r.y() or self.width != r.width() or self.height != r.height():
            self.x = r.x()
            self.y = r.y()
            self.width = r.width()
            self.height = r.height()
            changed = self._recalc_grid()
            if len(changed) > 0:
                self._rescale()
                for i in changed:
                    self.widget(i).draw()
            if self._spacing() > self.grid_spacing + 1:
                self._expand_width()
            self._changed = False

        spacing = self._spacing()
        y = self.y + spacing
        for i, (w, h) in enumerate(self._size):
            x = self.x + (self.width - w) // 2
            self.items[i].setGeometry(QRect(x, y, w, h))
            y += h + spacing
        self.resize_table()

    def _spacing(self) -> int:
        # get vertical spacing between items
        if self.nwidgets < 2:
            return 0
        height = self.height
        for w, h in self._size:
            height -= h
        return int(height / (self.nwidgets+1))

    def _expand_width(self) -> None:
        # expand grid width to fill up remaining space
        nitems = self.nwidgets
        expandable = set()
        fixed_height = self.grid_spacing
        expand_height = 0
        for i, (w, h) in enumerate(self._size):
            if w < self.width and self.widget(i).height > 1:
                expandable.add(i)
                expand_height += h
            else:
                fixed_height += h
            fixed_height += self.grid_spacing
        if len(expandable) == 0:
            return
        scale = (self.height-fixed_height) / expand_height
        for i in expandable:
            w, h = self._size[i]
            s = min(scale, self.width/w)
            self._size[i] = int(w*s), int(h*s)
        log.debug(f"PlotLayout: expand {expandable} => {self._size} scale={scale:.2f}")

    def _recalc_grid(self) -> set[int]:
        # adjust grid layout to make best use of available space - returns ids of updated grids
        nitems = self.nwidgets
        log.debug(f"PlotLayout: width={self.width} height={self.height} nitems={nitems}")
        if nitems <= 0:
            self._size = []
            return set()
        if nitems == 1:
            self._size = [(self.width, self.height)]
            return set()

        prev_grid = [(w.rows, w.cols) for w in self.widgets()]
        log.debug(f"PlotLayout: orig grid={prev_grid}")
        tried = set()
        while True:
            ix = self._rescale()
            if ix < 0 or ix in tried:
                break
            tried.add(ix)
            # expand grid rows while we have space, preserving aspect ratio
            w = self.widget(ix)
            max_factor = self.width / self._size[ix][0]
            if w.expand_width(max_factor):
                log.debug(f"PlotLayout: reshape grid {ix} => {(w.rows,w.cols)}")
            else:
                log.debug(f"PlotLayout: cannot expand {ix} : max_factor={max_factor}")
                break
        grid_changed: set[int] = set()
        for i, w in enumerate(self.widgets()):
            if prev_grid[i] != (w.rows, w.cols):
                grid_changed.add(i)
        return grid_changed

    def _set_min_height(self, ix: int, w: PlotGrid) -> int:
        # expand widget if height < self.min_size
        if w.nplots > 1:
            iw, ih = w.img_shape
            rows, cols = calc_grid(w.nplots, square=True)
            while rows > 1:
                r, c = calc_grid(w.nplots, rows-1, square=True)
                if self.min_size*(c*iw)/(r*ih) > self.width:
                    break
                rows, cols = r, c
            w.set_shape(rows, cols)
        log.debug(f"PlotLayout: set min height for grid {ix} => {(w.rows,w.cols)}")
        return self.min_size

    def _rescale(self) -> int:
        # calc max scale factor to fit in available width and apply min_size
        self._size = []
        scale = 0.0
        for w in self.widgets():
            s = self.width / w.width
            if scale == 0 or (w.height > 1 and s < scale):
                scale = s
        for ix, w in enumerate(self.widgets()):
            height = int(w.height*scale)
            if height < self.min_size:
                height = self._set_min_height(ix, w)
            if w.xlabels:
                height += 20
            width = min(int(height*w.width/w.height), self.width)
            self._size.append((width, height))

        # shrink items taller than min_size if total height exceeds layout height
        nitems = self.nwidgets
        total_height = 0
        avail_height = self.height - self.grid_spacing*(nitems+1)
        flex = set()
        for i, (w, h) in enumerate(self._size):
            if h > self.min_size:
                flex.add(i)
                total_height += h
            else:
                avail_height -= self.min_size

        if total_height > avail_height:
            scale = avail_height / total_height
            for i in flex:
                self._size[i] = (int(self._size[i][0]*scale), int(self._size[i][1]*scale))
        log.debug(f"PlotLayout: size={self._size}")

        # get narrowest multirow grid to reshape
        grid_to_resize = -1
        max_scale = 0.0
        for i, (w, h) in enumerate(self._size):
            scale = self.width / w
            if scale > 1.1 and scale > max_scale and self.widget(i).rows > 1 and self._size[i][1] > self.min_size:
                max_scale = scale
                grid_to_resize = i
        return grid_to_resize


def set_background(w, col):
    p = w.palette()
    p.setColor(w.backgroundRole(), QColor(col))
    w.setPalette(p)
    w.setAutoFillBackground(True)


def list_item(name, center=False, min_height=None):
    item = qw.QListWidgetItem(name)
    if center:
        item.setTextAlignment(Qt.AlignCenter)
    if min_height:
        item.setSizeHint(QSize(0, min_height))
    return item


def set_ticks(ax, classes):
    ax.setTicks([[(i+0.5, str(val)) for i, val in enumerate(classes)]])


def init_plot(p, xlabel=None, ylabel=None, xrange=None, yrange=None, legend=None, mouse=False):
    p.clear()
    if xlabel:
        p.setLabel("bottom", xlabel)
    if ylabel:
        p.setLabel("left", ylabel)
    if xrange:
        p.setXRange(xrange[0], xrange[1], padding=0)
    if yrange:
        p.setYRange(yrange[0], yrange[1], padding=0)
    if legend:
        p.addLegend(offset=legend, labelTextSize=str(font_size)+"pt")
    if mouse:
        p.getViewBox().setMouseMode(pg.ViewBox.PanMode)
    else:
        p.getViewBox().setMouseEnabled(x=False, y=False)


def add_line(p, xs, ys, color="w", dash=False, name=None):
    length = min(len(xs), len(ys))
    if dash:
        pen = pg.mkPen(color=color, width=2, style=Qt.DashLine)
    else:
        pen = pg.mkPen(color=color, width=2)
    return p.plot(xs[:length], ys[:length], pen=pen, name=name)


def update_lines(lines, xs, ys):
    for i, line in enumerate(lines):
        length = min(len(xs), len(ys[i]))
        line.setData(xs, ys[i])


def update_range(p, max_epoch, yvals):
    (x0, x1), (y0, y1) = p.viewRange()
    if x1 < max_epoch:
        p.setXRange(x0, max_epoch, padding=0)
    ys = [y[-1] for y in yvals if len(y) > 0]
    miny, maxy = min(ys), max(ys)
    if miny < y0 or maxy > y1:
        p.setYRange(min(miny, y0), max(maxy, y1))


def add_label(plot, x, y, val, col):
    text = pg.TextItem(f"{val:.0f}", color=col)
    plot.addItem(text)
    text.setPos(x+0.3, y+0.7)


def calc_grid(items: int, rows=0, square=False) -> tuple[int, int]:
    if items <= 1:
        return 1, 1
    if rows < 1:
        rows = round(math.sqrt(items))
    while rows >= 1:
        cols = 1 + (items-1) // rows
        if not square or rows*cols == items:
            return rows, cols
        rows -= 1
    return rows, cols


def to_image(data: Tensor) -> np.ndarray:
    """convert to image in [H, W] or [H, W, C] format"""
    n, c, h, w = data.shape
    img = np.flip(data.cpu().numpy(), axis=2)
    if c == 1:
        return img[0][0]
    if c == 3 or c == 4:
        return img[0].transpose((1, 2, 0))
    raise ValueError(f"Invalid data shape for image: {data.shape}")
