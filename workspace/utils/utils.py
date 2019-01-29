import os
import torch
import importlib

def get_instance(config, *args, **kwargs):
    if isinstance(config, str):
        splits = config.split('.')
    else:
        splits = config.type.split('.')
        if hasattr(config, 'kwargs'):
            kwargs.update(config.kwargs.to_dict())

    module = importlib.import_module('.'.join(splits[:-1]))
    return getattr(module, splits[-1])(*args, **kwargs)

##### config utils #####

_device_flag = ['cpu']

def to_device(item):
    """ convert a module / tensor to cuda mode.
    """
    if _device_flag[0] != 'cuda':
        return item
    if isinstance(item, torch.nn.Module):
        item.cuda()
    elif isinstance(item, dict):
        return {key: to_device(value) for key, value in item.items()}
    elif isinstance(item, torch.Tensor):
        return item.cuda()

class Config():
    """ use in config files. expand a dict as cls.__dict__
    """
    def __init__(self, config_dict=None):
        if config_dict is None:
            return

        for key, value in config_dict.items():
            if isinstance(value, dict): 
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def get_optimizer(config, params):
        import torch.optim as opt
        if hasattr(opt, config.type):
            return getattr(opt, config.type)(params, **config.kwargs.to_dict())

    def get_lr_scheduler(config, optimizer):
        import torch.optim.lr_scheduler as lrs
        if hasattr(lrs, config.type):
            return getattr(lrs, config.type)(
                optimizer, **config.kwargs.to_dict())

    def to_dict(self):
        ret = {}
        for key, value in self.__dict__.items(): 
            if isinstance(value, Config):
                ret[key] = value.to_dict()
            else:
                ret[key] = value
        return ret

def get_config(parser=None):
    """ add config,cuda args to a parser.
        config: the config module path (e.g, config.config, without .py suffix)
        cuda: set cuda mode. 
    """
    if parser is None:
        from argparse import ArgumentParser
        _parser = ArgumentParser()
    else:
        _parser = parser

    _parser.add_argument('--config')
    _parser.add_argument('--cuda', action='store_true')
    args = _parser.parse_args()

    config = importlib.import_module(args.config).config

    if args.cuda:
        _device_flag[0] = 'cuda'

    if parser is None:
        return config
    else:
        return config, args
