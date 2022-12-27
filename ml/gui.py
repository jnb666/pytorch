import logging as log
import math
from functools import reduce
from typing import Any

import numpy as np
import pyqtgraph as pg  # type: ignore
import PySide6.QtWidgets as qw
import torch
from PySide6.QtCore import QRect, QSize, Qt
from PySide6.QtGui import QColor
from torch import Tensor, nn

from .config import Config
from .dataset import Dataset
from .trainer import Stats
from .utils import load_checkpoint

font_size = 11
bgcolor = "#111"
fgcolor = "w"
spacing = 15
margins = 2
ypos = 50
layer_list_width = 150


def init_gui(width: int, height: int, title: str) -> tuple[qw.QApplication, qw.QMainWindow]:
    """Set default fonts, colors etc."""
    app = qw.QApplication([])
    fnt = app.font()
    fnt.setPointSize(font_size)
    app.setFont(fnt)
    pg.setConfigOption("background", bgcolor)
    pg.setConfigOption("foreground", fgcolor)
    pg.setConfigOption("antialias", True)
    pg.setConfigOption("imageAxisOrder", "row-major")

    win = qw.QMainWindow()
    win.setWindowTitle(title)
    sz = app.primaryScreen().size()
    win.setGeometry((sz.width()-width)//2, ypos, width, height)
    return app, win


class MainWindow(qw.QWidget):
    """Main GUI window with select list on the left and content area stacked widget on the right

    Args:
        cfg:Config           parsed data from config toml file
        model:nn.Module      torch network model
        test_data:Dataset    test images and labels
        transform:nn.Module  augmentationtransform applied to training data (optional)

    Properties:
        dir:str              current run directory
        topmenu:ImageMenu    top menu bar
        checkpoint:dict      loaded checkpoint data
        stats:Stats|None     latest stats from checkpoint
    """

    def __init__(self,
                 cfg: Config,
                 model: nn.Sequential,
                 test_data: Dataset,
                 transform: nn.Module | None):
        super().__init__()
        self.checkpoint: dict[str, Any] = {}
        self.stats: Stats | None = None
        self.topmenu = ImageMenu(test_data.classes)
        self.topmenu.hide()
        self.dir = cfg.dir
        epochs = int(cfg.train.get("epochs", 10))
        self._pages = [
            StatsPlots(epochs),
            Heatmap(test_data),
            Activations(self.topmenu, model, test_data, transform, cfg.dir),
            ImageViewer(self.topmenu, test_data, transform),
            ConfigLabel(str(cfg)),
        ]
        menu = self._build()
        rows = qw.QVBoxLayout()
        rows.addWidget(self.topmenu)
        rows.addWidget(self._content)
        cols = qw.QHBoxLayout()
        cols.addWidget(menu, 1)
        cols.addLayout(rows, 9)
        self.setLayout(cols)

    def _build(self):
        self._content = qw.QStackedWidget()
        menu = qw.QListWidget()
        self._page_ids = {}
        for i, name in enumerate(["stats", "heatmap", "activations", "images", "config"]):
            self._page_ids[name] = i
            self._content.addWidget(self._pages[i])
            menu.addItem(list_item(name, center=True, min_height=50))
        menu.currentItemChanged.connect(self._select_page)
        menu.setCurrentRow(0)
        return menu

    def _select_page(self, item):
        name = item.text()
        index = self._page_ids[name]
        log.debug(f"select page {index} {name}")
        if name == "activations" or name == "images":
            self.topmenu.register(self._pages[index])
            self.topmenu.show()
        else:
            self.topmenu.register(None)
            self.topmenu.hide()
        self._content.setCurrentIndex(index)
        self._pages[index].update_stats(self.stats, self.checkpoint)

    def update_stats(self, epoch: int | None = None) -> None:
        """Refresh GUI with updated stats and model weights"""
        try:
            self.checkpoint = load_checkpoint(self.dir, epoch)
        except FileNotFoundError:
            log.warning(f"no checkpoint found in {self.dir} - skip")
            return
        self.stats = Stats()
        self.stats.load_state_dict(self.checkpoint["stats_state_dict"])
        index = self._content.currentIndex()
        log.debug(f"update content index={index}")
        self._pages[index].update_stats(self.stats, self.checkpoint)


class StatsPlots(qw.QWidget):
    """StatsPlots widget shows the loss and accuracy plots

    Args:
    epochs:int  max number of epochs
    """

    def __init__(self, epochs: int):
        super().__init__()
        log.debug(f"init StatsPlots: epochs={epochs}")
        self._table = pg.TableWidget(editable=False, sortable=False)
        layout = qw.QHBoxLayout()
        layout.addWidget(self._table, 1)
        layout.addWidget(self._make_plots(epochs), 2)
        self.setLayout(layout)

    def _make_plots(self, epochs):
        w = pg.GraphicsLayoutWidget()
        self._plot1 = w.addPlot(row=0, col=0)
        self._plot1.showGrid(x=True, y=True, alpha=0.75)
        self._plot2 = w.addPlot(row=1, col=0)
        self._plot2.showGrid(x=True, y=True, alpha=0.75)
        init_plot(self._plot1, ylabel="cross entropy loss",
                  xrange=[1, epochs], yrange=[0, 1], legend=(0, 1))
        init_plot(self._plot2, xlabel="epoch", ylabel="accuracy",
                  xrange=[1, epochs], yrange=[0, 1])
        return w

    def _update_plots(self, stats: Stats) -> None:
        """draw line plots"""
        maxy = max(max(stats.train_loss), max(stats.test_loss))
        maxy = math.ceil(maxy*5) / 5 + 0.0001
        init_plot(self._plot1, ylabel="cross entropy loss",
                  xrange=stats.xrange, yrange=[0, maxy], legend=(0, 1))
        add_line(self._plot1, stats.epoch, stats.train_loss, color="r", name="training")
        add_line(self._plot1, stats.epoch, stats.test_loss, color="g", name="testing")
        if len(stats.valid_loss):
            add_line(self._plot1, stats.epoch, stats.valid_loss, color="y", name="validation")

        miny = min(stats.test_accuracy)
        miny = math.floor(miny*10) / 10
        init_plot(self._plot2, xlabel="epoch", ylabel="accuracy",
                  xrange=stats.xrange, yrange=[miny, 1.0001])
        add_line(self._plot2, stats.epoch, stats.test_accuracy, color="g", name="testing")
        if len(stats.valid_accuracy):
            add_line(self._plot2, stats.epoch, stats.valid_accuracy, color="y", name="validation")

    def _update_table(self, stats: Stats) -> None:
        """update stats table"""
        self._table.setFormat("%.3f", column=0)
        self._table.setFormat("%.3f", column=1)
        self._table.setFormat("%.1f", column=2)
        if len(stats.valid_loss):
            self._table.setFormat("%.3f", column=3)
            self._table.setFormat("%.1f", column=4)
        data = stats.table_data()
        data.reverse()
        cols = [(name, float) for name in stats.table_columns()]
        self._table.setData(np.array(data, dtype=cols))
        self._table.setVerticalHeaderLabels([str(i) for i in reversed(stats.epoch)])

    def update_stats(self, stats: Stats | None, checkpoint: dict[str, Any]) -> None:
        """Refresh GUI with updated stats"""
        if stats:
            log.debug(f"update stats: epoch={stats.current_epoch} of xrange={stats.xrange}")
            self._update_table(stats)
            self._update_plots(stats)


class Heatmap(pg.GraphicsLayoutWidget):
    """Heatmap shows a heatmap plot with correlation between the labels and the predictions

    Args:
        test_data:Dataset  - test data set
        cmap:string       - color map to use
    """

    def __init__(self, test_data: Dataset, cmap="CET-L6"):
        super().__init__()
        self.data = test_data
        self._targets = test_data.targets.to("cpu").numpy()
        self._predict = np.zeros((test_data.nitems))
        self._classes = len(test_data.classes)
        log.debug(f"init Heatmap: classes={self._classes} cmap={cmap}")
        self._plot, self._img = self._init_plot(self._classes, cmap)

    def _init_plot(self, classes, cmap):
        p = self.addPlot()
        init_plot(p, xlabel="target", ylabel="prediction",
                  xrange=[0, classes], yrange=[0, classes])
        set_ticks(p.getAxis("bottom"), self.data.classes)
        set_ticks(p.getAxis("left"), self.data.classes)
        img = pg.ImageItem(np.zeros((classes, classes)))
        img.setColorMap(cmap)
        p.setTitle("Epoch 0")
        p.addItem(img)
        return p, img

    def _add_label(self, x, y, val, col):
        if val > 0:
            text = pg.TextItem(f"{val:.0f}", color=col)
            self._plot.addItem(text)
            text.setPos(x+0.3, y+0.7)

    def update_stats(self, stats: Stats | None, checkpoint: dict[str, Any]) -> None:
        """Refresh GUI with updated stats"""
        if not stats or len(stats.predict) == 0:
            return
        predict = np.array(stats.predict)
        map, _, _ = np.histogram2d(self._targets, predict, bins=self._classes)
        log.debug(f"== heatmap: ==\n{map}")
        self._plot.clear()
        self._img.setImage(map)
        self._plot.addItem(self._img)
        for y in range(self._classes):
            for x in range(self._classes):
                val = map[x, y]
                col = "w" if val < 0.5 * \
                    len(self._targets)/self._classes else "k"
                self._add_label(x, y, val, col)
        self._plot.setTitle(f"Epoch {stats.current_epoch}")


class ImageMenu(qw.QWidget):
    """ImageMenu is the top menu bar shown on the Images and Activations screen

    Args:
        classes:list             list of data classes

    Properties:
        label:QLabel             text label to identify image or page number
        info:QLabel              text label with exta info
        target_class:int         show just this class
        errors_only:bool         only show errors
    """

    def __init__(self, classes: list):
        super().__init__()
        log.debug("init ImageMenu")
        self._listener = None
        self.label = qw.QLabel("")
        self.label.setMinimumWidth(120)
        self.info = qw.QLabel("")
        self._target_class = None
        self._errors_only = False
        self._transformed = False
        layout = self._build(classes)
        self.setLayout(layout)

    def _build(self, classes):
        prev = qw.QPushButton("<< prev")
        prev.clicked.connect(self._prev)
        next = qw.QPushButton("next >>")
        next.clicked.connect(self._next)
        filter_class = qw.QComboBox()
        filter_class.addItems(["all classes"] + classes)
        filter_class.currentIndexChanged.connect(self._filter_class)
        filter_errors = qw.QCheckBox("errors only")
        filter_errors.stateChanged.connect(self._filter_errors)
        transform = qw.QCheckBox("transform")
        transform.stateChanged.connect(self._transform)
        cols = qw.QHBoxLayout()
        cols.setContentsMargins(margins, margins, margins, margins)
        cols.addWidget(self.label)
        cols.addSpacing(spacing)
        cols.addWidget(prev)
        cols.addWidget(next)
        cols.addSpacing(spacing)
        cols.addWidget(qw.QLabel("show"))
        cols.addWidget(filter_class)
        cols.addSpacing(spacing)
        cols.addWidget(filter_errors)
        cols.addSpacing(spacing)
        cols.addWidget(transform)
        cols.addSpacing(spacing)
        cols.addWidget(self.info)
        cols.addStretch()
        return cols

    def register(self, listener) -> None:
        """Called when a new content page is loaded to reset the callbacks"""
        self._listener = listener

    def _prev(self):
        if self._listener is not None:
            self._listener.prev()

    def _next(self):
        if self._listener is not None:
            self._listener.next()

    def _filter_class(self, index):
        self._target_class = None if index == 0 else index-1
        if self._listener is not None:
            self._listener.filter(self._target_class, self._errors_only)

    def _filter_errors(self, state):
        self._errors_only = not self._errors_only
        if self._listener is not None:
            self._listener.filter(self._target_class, self._errors_only)

    def _transform(self, state):
        self._transformed = not self._transformed
        if self._listener is not None:
            self._listener.set_transformed(self._transformed)


class ImageViewer(pg.GraphicsLayoutWidget):
    """ImageViewer displays a grid of images with options to filter by class or errors

    Args:
        menu:ImageMenu        reference to top menu
        data:Dataset          image data set
        transform:nn.Module   set of transforms to apply
        rows, cols: int       grid size

    Properties:
        page:int              page number
        images_per_page:int   grid size
        image_size:tuple      size of each image
        data:Dataset          test dataset
        predict:Tensor|None   current predicted classes
        transform:nn.Module   set of transforms to apply
        transformed:bool      whether transform is enabled
    """

    def __init__(self, menu: ImageMenu, data: Dataset, transform: nn.Module | None, rows: int = 5, cols: int = 8):
        super().__init__()
        self.page: int = 0
        self.images_per_page = rows*cols
        self.image_size = data.image_shape
        self.data = data
        self.transform = transform
        self.transformed = False
        self.predict: Tensor | None = None
        self._menu = menu
        log.debug(f"init ImageViewer: image_size={self.image_size}")
        self._plots = self._build(rows, cols)

    @property
    def pages(self) -> int:
        return 1 + (self.data.nitems-1) // self.images_per_page

    def _build(self, rows, cols):
        plots = []
        for row in range(rows):
            for col in range(cols):
                index = row*cols + col
                p = self.addPlot(row, col)
                p.showAxis("top", False)
                p.showAxis("left", False)
                p.showAxis("bottom", False)
                p.setXRange(0, self.image_size[1], padding=0)
                p.setYRange(0, self.image_size[0], padding=0)
                p.setTitle(self._title(index))
                img = pg.ImageItem(self._image(index))
                p.addItem(img)
                plots.append((p, img))
        return plots

    def prev(self) -> None:
        """callback from ImageMenu """
        self.page -= 1
        if self.page < 0:
            self.page = self.pages-1
        self._update()

    def next(self) -> None:
        """callback from ImageMenu """
        self.page += 1
        if self.page >= self.pages:
            self.page = 0
        self._update()

    def filter(self, target_class: int | None, errors_only: bool) -> None:
        """callback from ImageMenu """
        if self.predict is not None:
            self.page = 0
            log.debug(
                f"filter images errors_only={errors_only} class={target_class}")
            self.data.filter(
                self.predict, target_class=target_class, errors_only=errors_only)
            self._update()

    def set_transformed(self, on: bool) -> None:
        """callback from ImageMenu """
        log.debug(f"set transformed => {on}")
        self.transformed = on
        self._update()

    def update_stats(self, stats: Stats | None, checkpoint: dict[str, Any]) -> None:
        """Called to update stats data after each epoch"""
        if stats and len(stats.predict):
            self.predict = torch.clone(stats.predict)
            self._update()

    def _update(self):
        if self.page >= self.pages:
            self.page = 0
        log.debug(f"update images page={self.page+1} / {self.pages}")
        self._menu.label.setText(f"Page {self.page+1} of {self.pages}")
        self._menu.info.setText("")
        start = self.page*self.images_per_page
        for i, (p, img) in enumerate(self._plots):
            img.setImage(self._image(start+i))
            p.setTitle(self._title(start+i))

    def _image(self, i: int) -> np.ndarray:
        if i >= self.data.nitems:
            return np.zeros(self.image_size)
        data = self.data.get(i, self.data.data)
        data = data.view(1, *data.shape)
        if self.transformed and self.transform is not None:
            data = self.transform(data)
        return to_image(data)

    def _title(self, i: int) -> str:
        s = ""
        if i < self.data.nitems:
            s = f"{self.data.indexOf(i)}:"
            if self.predict is not None:
                pred = int(self.data.get(i, self.predict))
                s += self.data.classes[pred]
        return f"<pre><font size=2>{s:<16}</font></pre>"


class Activations(qw.QWidget):
    """The Activations widgets displays image plots with the activation intensity for the selected layers.

    Args:
        menu:ImageMenu              top menu bar
        model:nn.Module             neural net model
        data:Dataset                test data - will load first batch of data
        transform:nn.Module         set of transforms to apply when enriching images
        test_transform:nn.Module    set of transforms to apply by default
        dir:str                     run directory with saved weights etc.
        cmap:str                    color map name

    Attributes:
        model:nn.Sequential         neural net model
        data:Dataset                test data - will load first batch of data
        layer_name:list[str]        short name of each layer type
        layer_state:list[bool]      flag indicating if each layer is shown
        layer_index:dict[str,int]   map from layer name to index
        index:int                   current image index
        activations:list[Tensor]    activations for each layer
        plots:PlotGrids             content widget holding the plots
        predict:Tensor|None         current predicted classes
    """

    def __init__(self,
                 menu: ImageMenu,
                 model: nn.Sequential,
                 data: Dataset,
                 transform: nn.Module | None,
                 dir: str,
                 cmap: str | None = "CET-L6"):
        super().__init__()
        self.data = data
        self.transform = transform
        self.transformed = False
        self.model = model
        self.menu = menu
        self.dir = dir
        self.cmap = cmap
        log.debug(f"init Activations: dir={dir} cmap={cmap}")
        self.index: int = 0
        self.activations: list[Tensor] = []
        self.predict: Tensor | None = None
        self.plots = PlotGrids(len(model)+1, cmap, data.classes)
        self.layer_state: list[bool] = []
        self.layer_name: list[str] = []
        self.layer_index: dict[str, int] = {}
        cols = qw.QHBoxLayout()
        cols.setContentsMargins(0, 0, 0, 0)
        cols.addWidget(self.plots)
        rows = qw.QVBoxLayout()
        rows.addWidget(self._layer_list())
        cols.addLayout(rows)
        self.setLayout(cols)

    @property
    def samples(self) -> int:
        return min(self.data.nitems, self.data.batch_size)

    def _layer_list(self):
        list = qw.QListWidget()
        list.setSelectionMode(qw.QAbstractItemView.MultiSelection)
        self._add_layer(list, 0, "input")
        enabled = 0
        for i, layer in enumerate(self.model):
            self._add_layer(list, i+1, str(layer))
        for i, name in enumerate(self.layer_name):
            if i == 0 or i == len(self.layer_name)-1 or (enabled <= 2 and ("Conv" in name or "Linear" in name)):
                self.layer_state[i] = True
                list.item(i).setSelected(True)
                enabled += 1
        list.itemClicked.connect(self._select_layer)
        list.setFixedWidth(layer_list_width)
        return list

    def _add_layer(self, list, index, name):
        try:
            name = name[:name.index("(")]
        except ValueError:
            pass
        if name.startswith("Lazy"):
            name = name[4:]
        self.layer_name.append(name)
        label = f"{index}: {name}"
        self.layer_state.append(False)
        self.layer_index[label] = index
        list.addItem(list_item(label, min_height=30))

    def _select_layer(self, item):
        name = item.text()
        index = self.layer_index[name]
        self.layer_state[index] = not self.layer_state[index]
        log.debug(f"toggle {name} index={index} {self.layer_state[index]}")
        self.plots.update_plots(self.index, self.activations, self.layer_state)

    def prev(self) -> None:
        """callback from ImageMenu """
        self.index -= 1
        if self.index < 0:
            self.index = self.samples-1
        self._update_plots()

    def next(self) -> None:
        """callback from ImageMenu """
        self.index += 1
        if self.index >= self.samples:
            self.index = 0
        self._update_plots()

    def filter(self, target_class: int | None, errors_only: bool) -> None:
        """callback from ImageMenu """
        if self.predict is not None:
            self.index = 0
            log.debug(
                f"Activations: filter images errors_only={errors_only} class={target_class}")
            self.data.filter(
                self.predict, target_class=target_class, errors_only=errors_only)
            self._recalc()
            self._update_plots()

    def set_transformed(self, on: bool) -> None:
        """callback from ImageMenu """
        log.debug(f"set transformed => {on}")
        self.transformed = on
        self._recalc()
        self._update_plots()

    def update_stats(self, stats: Stats | None, checkpoint: dict[str, Any]) -> None:
        """Called to update stats data after each epoch"""
        if checkpoint and stats and len(stats.predict):
            log.debug(f"Activations: update_stats - epoch={stats.current_epoch}")
            self.predict = torch.clone(stats.predict)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self._recalc()
            self._update_plots()

    @torch.no_grad()
    def _recalc(self) -> None:
        self.model.eval()
        x = self.data[0][0]
        if self.transformed and self.transform is not None:
            x = self.transform(x)
        log.debug(f"update activations: input={list(x.size())}")
        self.activations = [x]
        for i, layer in enumerate(self.model):
            x = layer(x)
            self.activations.append(x)
        self.outputs = x

    def _update_plots(self) -> None:
        if self.index >= self.samples:
            self.index = 0
        self.menu.label.setText(f"Image {self.index+1} of {self.samples}")
        tgt = int(self.data.get(self.index, self.data.targets))
        self.menu.info.setText(
            f"{self.data.indexOf(self.index)}: target={self.data.classes[tgt]}")
        self.plots.update_plots(self.index, self.activations, self.layer_state)


class PlotGrids(qw.QWidget):
    """PlotGrids is the containing widget for the layout with the set of activation layer grids

    Args:
        nlayers:int    total number of layers
        cmap           colormap name
        classes        list of output classes

    Properties:
        layers:dict[int,PlotGrid]  current set of layers
        layer_state:list[bool]     flag indicating which layers are enabled
    """

    def __init__(self, nlayers: int, cmap: str | None = None, classes=None):
        super().__init__()
        set_background(self)
        self.layer_state: list[bool] = [False] * nlayers
        self.layers: dict[int, PlotGrid] = {}
        self.cmap = cmap
        self.classes = classes
        self.table = OutputTable(classes)
        self._layout = PlotLayout()
        self._layout.addWidget(PlotGrid(None))
        self.setLayout(self._layout)

    def update_plots(self, index: int, activations: list[Tensor], layer_state: list[bool]) -> None:
        """update the grids to display the new activation tensors"""
        self.table.set_outputs(activations[-1][index])
        ids = [i for i, flag in enumerate(layer_state) if flag]
        if layer_state == self.layer_state:
            log.debug(f"update plots: index={index} {ids}")
            for i, grid in self.layers.items():
                grid.set_data(activations[i][index])
            self._layout.resize_table()
        else:
            log.debug(f"draw plots: index={index} {ids}")
            self._layout.clear()
            empty = True
            for i, enabled in enumerate(layer_state):
                if enabled:
                    self._layout.addWidget(
                        self._grid(i, activations[i][index]))
                    empty = False
            if empty:
                self._layout.addWidget(PlotGrid(None))
            self._layout.addWidget(self.table)
            self.layer_state = layer_state.copy()

    def _grid(self, i: int, data: Tensor) -> "PlotGrid":
        try:
            grid = self.layers[i]
            grid.set_data(data)
        except KeyError:
            labels = self.classes if i == len(self.layer_state)-1 else None
            is_rgb = (i == 0 and data.size()[0] == 3)
            grid = PlotGrid(data, cmap=self.cmap,
                            xlabels=labels, rgbimage=is_rgb)
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
        prob = torch.nn.functional.softmax(outputs, dim=0)
        ix = torch.argsort(prob, dim=0, descending=True)
        rows = []
        probs = prob.tolist()
        for i in ix.tolist():
            if probs[i] < 0.05:
                break
            rows.append((self.classes[i], "{:.1f}".format(100*probs[i])))
        log.debug(f"update table -> {rows}")
        self._update(rows)

    def size_hint(self):
        return 215, 25+30*self.rowCount()


class PlotGrid(pg.GraphicsLayoutWidget):
    """PlotGrid widget is a single grid of image plots

    Args:
    data:Tensor|None        tensor with activations
    cmap:str|None           colormap to use
    spacing:int             spacing between each plot
    xlabels                 if set then use these as list of x axis labels
    rgbimage                if set then consists of R,G,B color image channels

    Properties:
    plots:list[pg.PlotWidget]  list of plots
    data:np.ndarray            data in [N,C,H,W] format
    rows,cols:int              grid size
    xlabels                    set if have labels for outputs
    """

    def __init__(self, data: Tensor | None, cmap: str | None = None, spacing: int = 0, xlabels=None, rgbimage=False):
        super().__init__()
        self.cmap = cmap
        self.xlabels = xlabels
        self.rgbimage = rgbimage
        self.centralWidget.layout.setSpacing(spacing)
        self.centralWidget.layout.setContentsMargins(0, 0, 0, 0)
        self.plots: list[tuple[pg.PlotItem, pg.ImageItem]] = []
        if data is None:
            self.data = np.zeros((1, 10, 10))
        else:
            self.set_data(data)
        if rgbimage:
            self._rows, self._cols = 1, 1
        else:
            self._rows, self._cols = calc_grid(self.nplots)
        self._draw()

    @property
    def nplots(self) -> int:
        return self.data.shape[0]

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    @property
    def height(self) -> int:
        return self._rows*self.data.shape[1]

    @property
    def width(self) -> int:
        return self._cols*self.data.shape[2]

    def set_data(self, tdata: Tensor) -> None:
        """Update data for this set of plots"""
        data = tdata.to("cpu").numpy()
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
        if len(self.plots) == 0:
            return
        if self.rgbimage:
            self.plots[0][1].setImage(self.data.transpose((1, 2, 0)))
        else:
            for i, (plot, img) in enumerate(self.plots):
                img.setImage(self.data[i])

    def redraw(self) -> None:
        """Redraw the plots updaing the layout - call this when rows and cols have changed"""
        self.clear()
        for i, (plot, img) in enumerate(self.plots):
            self.addItem(plot, row=i//self.cols, col=i % self.cols)

    def expand_width(self, max_factor: float) -> bool:
        """Adjust grid rows and cols expanding width by up to max_factor ratio"""
        updated = False
        rows, cols = self._rows, self._cols
        while rows > 1:
            r, c = calc_grid(self.nplots, rows-1)
            if c/self.cols <= max_factor:
                rows, cols = r, c
                updated = True
            else:
                break
        if updated:
            self._rows, self._cols = rows, cols
        return updated

    def _draw(self) -> None:
        if self.rgbimage:
            if self.nplots != 3 and self.nplots != 4:
                raise ValueError(
                    "expecting 3 or 4 color channels for RGB image")
            data = self.data.transpose((1, 2, 0))
            plot, img = self._plot(data)
            self.addItem(plot, row=0, col=0)
            self.plots.append((plot, img))
        else:
            for i in range(self.nplots):
                plot, img = self._plot(self.data[i], self.cmap)
                self.addItem(plot, row=i//self._cols, col=i % self._cols)
                self.plots.append((plot, img))

    def _plot(self, data: np.ndarray, cmap=None) -> tuple[pg.PlotItem, pg.ImageItem]:
        """plot data from numpy array in [H,W] or[H,W,C] format"""
        p = pg.PlotItem()
        p.showAxis("left", False)
        if self.xlabels:
            set_ticks(p.getAxis("bottom"), self.xlabels)
        else:
            p.showAxis("bottom", False)
        p.setXRange(0, data.shape[1], padding=0)
        p.setYRange(0, data.shape[0], padding=0)
        img = pg.ImageItem(data)
        if cmap is not None:
            img.setColorMap(cmap)
        p.addItem(img)
        return p, img


class PlotLayout(qw.QLayout):
    """PlotLayout is used to arrange the activation grids in a custom layout

    Args:
        spacing:int       spacing between each grid
        min_size:int      minimum grid height

    Properties:
        x,y,width,height  layout position and size
        items             list of layout items
        min_size:int      minimum grid height
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

    @property
    def nwidgets(self):
        return len(self.items)-1

    def resize_table(self):
        table = self.items[-1].widget()
        if not isinstance(table, OutputTable):
            return
        w, h = table.size_hint()
        # log.debug(f"table resize => {w},{h}")
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
                    self.widget(i).redraw()
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
        # log.debug(f"PlotLayout: expand {expandable} => {self._size} scale={scale:.2f}")

    def _recalc_grid(self) -> set[int]:
        # adjust grid layout to make best use of available space - returns ids of updated grids
        nitems = self.nwidgets
        log.debug(
            f"PlotLayout: width={self.width} height={self.height} nitems={nitems}")
        if nitems <= 0:
            self._size = []
            return set()
        if nitems == 1:
            self._size = [(self.width, self.height)]
            return set()

        prev_grid = [(w.rows, w.cols) for w in self.widgets()]
        # log.debug(f"PlotLayout: orig grid={prev_grid}")
        while True:
            ix = self._rescale()
            if ix < 0:
                break
            # expand grid rows while we have space
            w = self.widget(ix)
            if w.expand_width(self.width / self._size[ix][0]):
                log.debug(
                    f"PlotLayout: reshape grid {ix} => {(w.rows,w.cols)}")
            else:
                break
        grid_changed: set[int] = set()
        for i, w in enumerate(self.widgets()):
            if prev_grid[i] != (w.rows, w.cols):
                grid_changed.add(i)
        return grid_changed

    def _rescale(self) -> int:
        # calc max scale factor to fit in available width and apply min_size
        self._size = []
        scale = 0.0
        for w in self.widgets():
            s = self.width / w.width
            if scale == 0 or (w.height > 1 and s < scale):
                scale = s
        for w in self.widgets():
            height = int(w.height*scale)
            if w.nplots == 1:
                height = max(height, self.min_size)
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
                self._size[i] = (int(self._size[i][0]*scale),
                                 int(self._size[i][1]*scale))

        # get narrowest multirow grid to reshape
        grid_to_resize = -1
        max_scale = 0.0
        for i, (w, h) in enumerate(self._size):
            scale = self.width / w
            if scale > 1.1 and scale > max_scale and self.widget(i).rows > 1:
                max_scale = scale
                grid_to_resize = i
        return grid_to_resize


class ConfigLabel(qw.QScrollArea):
    """ConfigLabel is a scrollable Qlabel with fixed format text and top alignment"""

    def __init__(self, text):
        super().__init__()
        self.setWidgetResizable(True)
        content = qw.QLabel(text)
        content.setAlignment(Qt.AlignTop)
        content.setWordWrap(True)
        content.setStyleSheet(
            f'background-color: "{bgcolor}"; border: 10px solid "{bgcolor}"; font-family: "Monospace"; font-size: 10pt;'
        )
        self.setWidget(content)

    def update_stats(self, stats: Stats | None, checkpoint: dict[str, Any]) -> None:
        pass


def set_background(w):
    p = w.palette()
    p.setColor(w.backgroundRole(), QColor(bgcolor))
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


def init_plot(p, xlabel=None, ylabel=None, xrange=None, yrange=None, legend=None):
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


def add_line(p, xs, ys, color="w", name=None):
    pen = pg.mkPen(color=color, width=2)
    return p.plot(xs, ys, pen=pen, name=name)


def _calc_grid(items, rows=None) -> tuple[int, int]:
    if rows is None:
        rows = round(math.sqrt(items))
    cols = 1 + (items-1) // rows
    return rows, cols


def calc_grid(items: int, rows: int | None = None) -> tuple[int, int]:
    if items <= 1:
        return 1, 1
    rows, cols = _calc_grid(items, rows)
    r, c = rows, cols
    i = 0
    while r > 2 and i < 2:
        r, c = _calc_grid(items, rows-i-1)
        if r*c == items:
            return r, c
        i += 1
    return rows, cols


def to_image(data: Tensor) -> np.ndarray:
    """convert to image in [H,W] or [H,W,C] format"""
    n, c, h, w = data.shape
    img = np.flip(data.numpy(), axis=2)
    if c == 1:
        return img[0][0]
    elif c == 3 or c == 4:
        return img[0].transpose((1, 2, 0))
    raise ValueError(f"Invalid data shape for image: {data.shape}")
    raise ValueError(f"Invalid data shape for image: {data.shape}")
