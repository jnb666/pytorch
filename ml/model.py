import logging as log
import math
from typing import Any

import torch
from torch import Tensor, nn

from .config import Config
from .utils import InvalidConfigError, get_definition, get_module


def expand_name(name):
    if "Linear" in name or "Conv" in name or "BatchNorm" in name:
        return "Lazy" + name
    else:
        return name


class Model(nn.Sequential):
    """Model class is a torch neural network based on nn.Sequential, but with an explicit weight initialisation step

    Args:
       config:Config        parsed toml config file
       device:str           cuda or cpu device

    Attributes:
        cfg:Config          reference to config
        device:str          device for parameters
        layers:list         list of modules
    """

    def __init__(self, config: Config, device: str = "cpu"):
        super().__init__()
        self.device = device
        self.cfg = config
        cfg = config.cfg["model"]
        for args in cfg["layers"]:
            defn = cfg.get(args[0])
            argv = args.copy()
            if defn is not None:
                vars = argv.pop() if len(argv) > 1 else {}
                log.debug(f"get_definition: {defn} {argv[0]} {vars}")
                for layer in get_definition(torch.nn, defn, vars, expand=expand_name):
                    self.append(layer.to(device))
            else:
                log.debug(f"add layer: {args}")
                layer = get_module(torch.nn, args, expand=expand_name)
                self.append(layer.to(device))

    def init_weights(self, input_shape: tuple[int, int, int]) -> None:
        """Initialise the weights for each layer in the model and instantiate lazy layers"""
        dtype = torch.float16 if self.cfg.half else torch.float32
        log.info(f"== Init weights: ==   {dtype}")
        x = torch.zeros((1,)+input_shape, device=self.device, dtype=dtype)
        total_params = 0
        weight_init = {}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.cfg.half):
            for i, layer in enumerate(self):
                try:
                    x = layer(x)
                except RuntimeError as err:
                    raise InvalidConfigError(f"error in layer {i}: {layer} definition\n  {err}")
                name = get_name(layer)
                nparams = num_params(layer)
                params = f"params={nparams}" if nparams else ""
                winit = init_layer_weights("weight", layer, self.cfg.weight_init)
                if winit:
                    weight_init[name + " weight init"] = winit
                winit = init_layer_weights("bias", layer, self.cfg.bias_init)
                if winit:
                    weight_init[name + " bias init"] = winit
                log.info(f"layer {i:2}: {name:<12} {list(x.size()[1:])} {params}")
                total_params += nparams
        log.info(f"total params = {total_params}")
        for typ in sorted(weight_init.keys()):
            log.info(f"{typ:30}: {weight_init[typ]}")

    def activations(self, input: Tensor, layers: list[int], hist_bins: int = 0) -> dict[int, Any]:
        """Evaluate model and returns activations or histograms of activations for each layer given"""
        self.eval()
        res = {}

        def add(i, x):
            if hist_bins:
                data = x.to(device="cpu", dtype=torch.float32).flatten()
                height, xpos = torch.histogram(data, hist_bins)
                log.debug(f"model histogram: layer={i} range={xpos[0]:.2f}:{xpos[-1]:.2f}")
                res[i] = height, xpos
            else:
                log.debug(f"model activations: layer={i} shape={list(x.shape)}")
                res[i] = x.cpu()

        with torch.no_grad():
            x = input
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.cfg.half):
                for i, layer in enumerate(self):
                    if i in layers:
                        add(i, x)
                    x = layer(x)
                add(len(self), x)
        return res


def init_layer_weights(typ: str, layer: nn.Module, params: dict[str, Any]) -> str:
    try:
        weights = getattr(layer, typ)
    except AttributeError:
        return ""
    if weights is None:
        return ""
    name = get_name(layer)
    # first try exact then substring match
    for layer_type, args in params.items():
        if layer_type == name:
            return set_weights(weights, args.copy())
    for layer_type, args in params.items():
        if layer_type in name:
            return set_weights(weights, args.copy())
    raise InvalidConfigError(f"missing {typ} init for {name}")


def set_weights(weights, args):
    if len(args) >= 2 and isinstance(args[-1], dict):
        kwargs = args.pop()
    else:
        kwargs = {}
    try:
        fn = getattr(torch.nn.init, args[0]+"_")
    except AttributeError as err:
        raise InvalidConfigError(f"invalid init function {args} {kwargs} - {err}")
    fn(weights, *args[1:], **kwargs)
    plist = (",".join([str(x) for x in args[1:]]) +
             ",".join([f"{k}={v}" for k, v in kwargs.items()]))
    return args[0] + "(" + plist + ")"


def get_name(layer: nn.Module) -> str:
    name = str(layer)
    return name[:name.index("(")]


def num_params(layer: nn.Module) -> int:
    params = 0
    for val in layer.state_dict().values():
        params += math.prod(val.size())
    return params
