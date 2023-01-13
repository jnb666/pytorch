import logging as log
import math
from typing import Any

import torch
from torch import Tensor, nn

from .config import Config, Index
from .utils import (InvalidConfigError, format_args, get_module, getarg,
                    getargs, splitargs)


class Layer(nn.Module):
    """Layer wraps a single torch.nn layer.

    Args:
        index:Index             unique layer index
        args:list               argument list from parsed config file
        vars:dict               variables in config to resolve [optional]

    Attributes:
        typ:str                 type name as defined in config file
        module:nn.Module        wrapped module - uses Lazy version of torch.nn class where available
        out_shape:torch.Size    output shape - set when containing Model is initialised
    """

    def __init__(self, index: Index, argv: list[Any], vars=None, device: str = "cpu"):
        super().__init__()
        self.index = index
        self.out_shape = torch.Size()
        self.typ, self.args, self.kwargs = getargs(argv, vars)
        if hasattr(torch.nn, "Lazy" + self.typ):
            typ = "Lazy" + self.typ
            kwargs = self.kwargs.copy()
            kwargs["device"] = device
        else:
            typ = self.typ
            kwargs = self.kwargs
        self.module = get_module("model", torch.nn, typ, self.args, kwargs)
        log.debug(f"  {index} {repr(self.module)}")

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)

    def initialise(self, input: Tensor) -> Tensor:
        # log.debug(f"{self.index} {self.typ} initialise {list(input.shape)}")
        x = self(input)
        self.out_shape = x.size()[1:]
        return x

    def layer_names(self) -> list[tuple[Index, str]]:
        return [(self.index, self.typ)]

    def get_activations(self, input: Tensor, layers: list[Index], res: dict[Index, Any], hist_bins: int = 0) -> Tensor:
        x = self(input)
        if self.index in layers:
            add_activations(res, self.index, x, hist_bins)
        return x

    def __str__(self) -> str:
        n = num_params(self)
        return "{desc:70}{shape:13}{params}".format(
            desc=self.index.format() + ": " + self.typ + format_args(self.args, self.kwargs),
            shape=str(list(self.out_shape)),
            params=f"  {n:,} params" if n else ""
        )


class AddBlock(nn.Module):
    """Component of a ResNet network, Returns block1(x) + block2(x).

    Args:
        model:Model             parent model
        index:Index             unique layer index
        argv:list               argument list from parsed config file = [block1, block2]
        vars:dict               variables in config to resolve [optional]

    Attributes:
        layers:dict             registered layers from Model
        module1:nn.Sequential   block1 module list
        module2:nn.Sequential   block2 module list
        block1:list             indexes of block1 layers
        block2:list             indexes of block2 layers
    """

    def __init__(self, model: "Model", index: Index, args: list[Any], vars: dict[str, Any] | None):
        super().__init__()
        self.layers = model.layers
        self.index = index
        self.typ = "AddBlock"
        if len(args) != 2:
            raise InvalidConfigError(f"expecting 2 args for AddBlock - got {args}")
        log.debug(f"  {index} AddBlock {args}")

        self.module1 = nn.Sequential()
        self.block1: list[Index] = []
        model.add(Index(index + (0, 0)), self.block1, self.module1, [args[0]], vars=vars)

        self.module2 = nn.Sequential()
        self.block2: list[Index] = []
        model.add(Index(index + (1, 0)), self.block2, self.module2, [args[1]], vars=vars)

    def forward(self, x: Tensor) -> Tensor:
        return self.module1(x) + self.module2(x)

    def initialise(self, input: Tensor) -> Tensor:
        # log.debug(f"{self.index} {self.typ} initialise {list(input.shape)}")
        x = input
        for ix in self.block1:
            x = self.layers[ix].initialise(x)
        y = input
        for ix in self.block2:
            y = self.layers[ix].initialise(y)
        return x + y

    def layer_names(self) -> list[tuple[Index, str]]:
        names = []
        for ix in self.block1 + self.block2:
            names.extend(self.layers[ix].layer_names())
        return names

    def get_activations(self, input: Tensor, layers: list[Index], res: dict[Index, Any], hist_bins: int = 0) -> Tensor:
        x = input
        for ix in self.block1:
            x = self.layers[ix].get_activations(x, layers, res, hist_bins)
        y = input
        for ix in self.block2:
            y = self.layers[ix].get_activations(y, layers, res, hist_bins)
        return x + y

    def _format(self, block: list[Index]) -> str:
        if len(block) == 0:
            return "[]"
        layers = [str(self.layers[ix]) for ix in block]
        return "[\n" + "\n".join(layers) + "\n    ]"

    def __str__(self) -> str:
        return self.index.format() + ": AddBlock(" + self._format(self.block1) + "," + self._format(self.block2) + ")"


class Model(nn.Sequential):
    """Model class is a torch neural network based on nn.Sequential, but with an explicit weight initialisation step

    The model is instantiated with float32 datatype on the CPU, it is ony moved to the device after weight initialisation.

    Args:
        config:Config           parsed toml config file
        input_shape:torch.Size  shape of input image (C,H,W)
        device:str              device for layer weights
        init_weights:bool       if set then will call init_weights

    Attributes:
        config:Config           config info
        input_shape:torch.Size  input image shape
        indexes:list            indexes of registered layers
        layers:dict             registered layers
    """

    def __init__(self, config: Config, input_shape: torch.Size, device: str = "cpu", init_weights: bool = False):
        super().__init__()
        self.config = config
        self.model_cfg = config.cfg.get("model", {})
        self.device = device
        self.input_shape = input_shape
        self.layers: dict[Index, Any] = {}
        self.indexes: list[Index] = []
        self.add(Index((1,)), self.indexes, self, ["layers"])

        x = torch.zeros((1,) + self.input_shape, device=device)
        for index in self.indexes:
            try:
                x = self.layers[index].initialise(x)
            except RuntimeError as err:
                raise InvalidConfigError(f"error building model: {err}")

        self._weight_info: dict[str, str] = {}
        if init_weights:
            self.init_weights()
        self.to(device)

    def init_weights(self) -> None:
        """Initialise model weights using init function defined in the config"""

        def weight_init(layer: nn.Module):
            if (isinstance(layer, Model) or isinstance(layer, Layer) or isinstance(layer, AddBlock)
                    or isinstance(layer, nn.Sequential)):
                return
            typ = type(layer).__name__
            for name, param in layer.named_parameters():
                init_config = self.config.weight_init if name == "weight" else self.config.bias_init
                init_parameter(self._weight_info, name, typ, param, init_config)

        self._weight_info.clear()
        with torch.no_grad():
            self.apply(weight_init)
        log.info("== init weights: ==\n" + self.weight_info())

    def add(self, index: Index, indexes: list[Index], seq: nn.Sequential, args: list[Any], vars=None) -> Index:
        """Append a new layer or group to the network at index and return the next index """
        if not isinstance(args, list) or len(args) == 0:
            raise InvalidConfigError(f"invalid layer args: {args}")
        defn = self.model_cfg.get(args[0])
        if defn is not None or args[0] == "AddBlock":
            typ, block_args, block_vars = splitargs(args)
            if block_vars:
                if vars is None:
                    vars = {}
                for name, var in block_vars.items():
                    vars[name] = getarg(var, vars)
            if args[0] == "AddBlock":
                adder = AddBlock(self, index, block_args, vars)
                seq.append(adder)
                self.layers[index] = adder
            elif isinstance(defn, list):
                log.debug(f"[{typ}] {vars}")
                for item in defn:
                    index = self.add(index, indexes, seq, item, vars)
                return index
            else:
                raise InvalidConfigError(f"block definition should be a list - got {defn}")
        else:
            layer = Layer(index, args, vars, device=self.device)
            seq.append(layer)
            self.layers[index] = layer
        indexes.append(index)
        return index.next()

    def layer_names(self) -> list[tuple[Index, str]]:
        """Flattened list of layer index and name for all layers in the network"""
        names = [(Index((0,)), "input")]
        for ix in self.indexes:
            names.extend(self.layers[ix].layer_names())
        return names

    def activations(self, input: Tensor, layers: list[Index], hist_bins: int = 0) -> dict[Index, Any]:
        """Evaluate model and returns activations or histograms of activations for each layer given"""
        self.eval()
        res: dict[Index, Any] = {}
        with torch.no_grad():
            x = input
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.config.half):
                if Index((0,)) in layers:
                    add_activations(res, Index((0,)), x, hist_bins)
                for ix in self.indexes:
                    x = self.layers[ix].get_activations(x, layers, res, hist_bins)
        return res

    def weight_info(self) -> str:
        return "\n".join([f"{key:20}: {self._weight_info[key]}" for key in sorted(self._weight_info.keys())])

    def __str__(self):
        s = "== model: =="
        s += f"\n{Index((0,)).format()}: {'Input':66}{str(list(self.input_shape))}"
        total_params = 0
        for ix in self.indexes:
            layer = self.layers[ix]
            s += "\n" + str(layer)
            total_params += num_params(layer)
        return s + f"\ntotal parameters: {total_params:,}"


def init_parameter(weight_info, weight_type, layer_type, param, config) -> None:

    def set_weights(param, argv):
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
    raise InvalidConfigError(f"no {weight_type} init found for {layer_type}")


def num_params(m: nn.Module) -> int:
    n = 0
    for p in m.parameters():
        n += math.prod(p.size())
    return n


def add_activations(res, index, x, hist_bins=0):
    if hist_bins:
        vals = x.to(dtype=torch.float32).flatten()
        x0, x1 = torch.min(vals).item(), torch.max(vals).item()
        hist = torch.histc(vals, hist_bins, x0, x1)
        log.debug(f"model histogram: layer={index} range={x0:.2f}:{x1:.2f}")
        res[index] = hist.cpu(), x0, x1
    else:
        log.debug(f"model activations: layer={index} shape={list(x.shape)}")
        res[index] = x.cpu()
