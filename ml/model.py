import logging as log
import math
from typing import Any

import torch
from torch import Tensor, nn

from .config import Config
from .utils import InvalidConfigError, format_args, get_module, splitargs


class Index(tuple):
    """Index is a unique id to reference each layer in the model.

    For a sequential stack of layers it will be a single integer. For more complex modules it will be a tuple.
    e.g. ["Conv2d", "RelU", ["Add", ["Conv2d", "BatchNorm2d", ...], ["Conv2d", "BatchNorm2d"]], "ReLU", ... ]
        Index(2, 0, 1) is the first BatchNorm2d layer in the list
    """

    def __new__(self, *ix):
        return tuple.__new__(Index, ix)

    def next(self) -> "Index":
        if len(self) == 0:
            return Index(0)
        elif len(self) == 1:
            return Index(self[0]+1)
        else:
            return Index(*self[:-1], self[-1]+1)

    def __str__(self):
        return ".".join([str(ix) for ix in self])


class Module:
    """Base class for all of the network submodules"""

    def __init__(self, index: Index, typ: str):
        super().__init__()
        self.index = index
        self.typ = typ
        self.out_shape = torch.Size()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("virtual method")


class Layer(Module):
    """Layer wraps a single torch.nn layer.

    Args:
        index:Index             unique layer index
        args:list               argument list from parsed config file
        vars:dict               variables in config to resolve [optional]

    Attributes:
        typ:str                 type name as defined in config file
        layer:nn.Module         wrapped module - uses Lazy version of torch.nn class where available
        out_shape:torch.Size    output shape - set when containing Model is initialised
    """

    def __init__(self, index: Index, args: list[Any], vars=None):
        super().__init__(index, args[0])
        log.debug(f"{index} layer: {args}")
        self.layer = get_module(torch.nn, args, vars=vars, desc="model")

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer(x)
        self.out_shape = x.size()[1:]
        return x

    def num_params(self) -> int:
        n = 0
        for p in self.layer.parameters():
            n += math.prod(p.size())
        return n

    def __str__(self) -> str:
        s = f"{str(self.index):>2}: {self.typ:12}{str(list(self.out_shape)):15}"
        n = self.num_params()
        s += f"  params={n}" if n else ""
        return s


class Model(nn.Sequential):
    """Model class is a torch neural network based on nn.Sequential, but with an explicit weight initialisation step

    The model is instantiated with float32 datatype on the CPU, it is ony moved to the device after weight initialisation.

    Args:
       config:Config            parsed toml config file
       input_shape:torch.Size   shape of input image (C,H,W)

    Attributes:
        config:Config           config info
        input_shape:torch.Size  input image shape
        indexes:list            indexes of registered layers
        layers:dict             registered layers
    """

    def __init__(self, config: Config, input_shape: torch.Size, device: str = "cpu", init_weights: bool = False):
        super().__init__()
        self.config = config
        self.device = device
        self.input_shape = input_shape

        self.indexes: list[Index] = []
        self.layers: dict[Index, Layer] = {}
        index = Index(0)
        for args in config.layers:
            index = self.add(index, args)

        x = torch.zeros((1,) + self.input_shape)
        for index in self.indexes:
            try:
                x = self.layers[index].forward(x)
            except RuntimeError as err:
                raise InvalidConfigError(f"error building model: {err}")

        self.weight_info: dict[str, str] = {}
        if init_weights:
            self.apply(self._weight_init)
        self.to(device)

    def add(self, index: Index, args: list[Any], vars=None) -> Index:
        """Append a new layer to the network at index and return the next index"""
        if not isinstance(args, list) or len(args) == 0:
            raise InvalidConfigError(f"invalid layer args: {args}")
        cfg = self.config.cfg.get("model", {})
        defn = cfg.get(args[0])
        if defn and isinstance(defn, list):
            _, _, vars = splitargs(args)
            for item in defn:
                index = self.add(index, item, vars)
        else:
            m = Layer(index, args, vars)
            self.append(m.layer)
            self.indexes.append(index)
            self.layers[index] = m
            index = index.next()
        return index

    def activations(self, input: Tensor, layers: list[int], hist_bins: int = 0) -> dict[int, Any]:
        """Evaluate model and returns activations or histograms of activations for each layer given"""
        self.eval()
        res = {}

        def add(i, x):
            if hist_bins:
                vals = x.to(dtype=torch.float32).flatten()
                x0, x1 = torch.min(vals).item(), torch.max(vals).item()
                hist = torch.histc(vals, hist_bins, x0, x1)
                log.debug(f"model histogram: layer={i} range={x0:.2f}:{x1:.2f}")
                res[i] = hist.cpu(), x0, x1
            else:
                log.debug(f"model activations: layer={i} shape={list(x.shape)}")
                res[i] = x.cpu()

        with torch.no_grad():
            x = input
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.config.half):
                for i, layer in enumerate(self):
                    if i in layers:
                        add(i, x)
                    x = layer(x)
                add(len(self), x)
        return res

    def __str__(self):
        s = "== model =="
        for index in self.indexes:
            s += "\n" + str(self.layers[index])
        for key in sorted(self.weight_info.keys()):
            s += f"\n{key:20}: {self.weight_info[key]}"
        return s

    @torch.no_grad()
    def _weight_init(self, layer: nn.Module):
        typ = type(layer).__name__
        for name, param in layer.named_parameters():
            init_config = self.config.weight_init if name == "weight" else self.config.bias_init
            init_parameter(self.weight_info, name, typ, param, init_config)


def init_parameter(weight_info, weight_type, layer_type, param, config) -> None:

    def set_weights(param, argv):
        log.debug(f"init {weight_type} for {layer_type}: {argv}")
        typ, args, kwargs = splitargs(argv)
        try:
            getattr(torch.nn.init, typ+"_")(param, *args, **kwargs)
            weight_info[layer_type + " " + weight_type] = typ + format_args(args, kwargs)
        except (AttributeError, TypeError) as err:
            raise InvalidConfigError(f"error in {typ} init: {err}")

    # first try exact then substring match
    for key, argv in config.items():
        if key == layer_type:
            set_weights(param, argv)
            return
    for key, argv in config.items():
        if key in layer_type:
            set_weights(param, argv)
            return
