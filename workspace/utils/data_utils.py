import numpy as np
import torch

from nltk.tokenize import wordpunct_tokenize

from .to_pack_cls import *
from .field_cls import *
from .debug_utils import prolog, is_test

_tokenize_pipeline = []

def tokenize_pipeline(fn=None):
    if fn is None:
        return _tokenize_fn
    if not isinstance(fn, list):
        fn = [fn]
    _tokenize_pipeline.clear()
    _tokenize_pipeline.extend(fn)

def tokenize(example, ppln=None):
    if ppln is None:
        ppln = _tokenize_pipeline
    for fn in ppln:
        example = fn(example)
    return example

def tokens2str(tokens, field):
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.tolist()
    string = [field.itos[token] for token in tokens]
    string = string[1:tokens.index(field.stoi[field.eos])]
    return ' '.join(string)

def numericalize(tokens, field):
    unk_ids = field.stoi[field.unk]
    return [field.stoi[field.bos]] + \
        [field.stoi.get(token, unk_ids) for token in tokens] + \
        [field.stoi[field.eos]]

def preprocess(examples, field, log=None, in_pack=False):
    if isinstance(examples, to_pack):
        if isinstance(examples.lst, str):
            return to_pack(numericalize(tokenize(examples.lst), field))
        elif isinstance(examples.lst[0], str):
            return to_pack([numericalize(tokenize(example), field) 
                for example in examples])
        else:
            return examples
    elif isinstance(examples, tuple):
        return (preprocess(example, field) for example in examples)
    elif isinstance(examples, list):
        if log is not None:
            examples = prolog(
                examples, name=(log if isinstance(log, str) else ''))
        return [preprocess(example, field) for example in examples]
    elif isinstance(examples, dict):
        return {
            key: preprocess(value, field) for key, value in examples.items()}
    return examples

def str2data(strings, field):
    return pack2data(preprocess({'pack': to_pack(strings)}, field)['pack'])

#### initialization ####

tokenize_pipeline([wordpunct_tokenize])
