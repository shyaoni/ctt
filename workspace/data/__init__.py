from argparse import ArgumentParser

from .dataset import *
from utils import get_instance

def data_parser(parser=None):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument('--dts', type=str)
    return parser

def get_dataset(name, *args, **kwargs):
    return get_instance('data.' + name, *args, **kwargs)
    
