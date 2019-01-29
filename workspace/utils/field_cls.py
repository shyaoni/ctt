import numpy as np
import torch
from .debug_utils import is_test, prolog
from .cache_utils import cacher

class Field():
    """ A field class wraps a vocab with special tokens.
    """
    def __init__(self, vocab, 
                 bos='<bos>', eos='<eos>', pad='<pad>', unk='<unk>'):
        self.itos = [pad, bos, eos, unk] + vocab
        self.stoi = {x:i+4 for i, x in enumerate(vocab)}
        self.bos = bos
        self.eos = eos
        self.pad = pad
        self.unk = unk
        self.stoi[pad] = 0
        self.stoi[bos] = 1
        self.stoi[eos] = 2
        self.stoi[unk] = 3
        self.vocab = vocab
    
    def get_embedding_from_glove(self, config):
        cache = cacher('embedding', self.vocab, config.dim, config.vocab_size)

        if cache.cached:
            return cache.data

        embedding = [None, ] * len(self.itos)

        if is_test('data'):
            for i, x in enumerate(embedding):
                if i == 0:
                    embedding[i] = np.zeros(config.dim)
                elif x is None:
                    embedding[i] = np.random.uniform(-1, 1, config.dim).tolist()
            return torch.FloatTensor(embedding)    
        
        with open(config.path, 'r') as f:
            for line in prolog(f):
                word, vec_str = line.split(' ', maxsplit=1)
                vec = [float(val) for val in vec_str.split(' ')]

            if config.dim != len(vec):
                raise ValueError('dim not matched in glove settings.')

            if word in self.itos:
                embedding[self.itos[word]] = vec

        for i, x in enumerate(embedding):
            if i == 0:
                embedding[i] = np.zeros(config.dim)
            elif x is None:
                embedding[i] = np.random.uniform(-1, 1, config.dim).tolist()

        return cache.cache(torch.FloatTensor(embedding))
