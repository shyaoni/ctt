import os, sys
import torch
import hashlib

from .debug_utils import is_test

_cache_path = ['/tmp']

def cache_path(path=None):
    if path is None:
        return _cache_path[0]
    _cache_path[0] = path

class cacher():
    def __init__(self, *args):
        self.cached = False
        self.name = args[0]

        to_hash = ','.join([arg.__str__() for arg in args])
        hash_value = hashlib.md5(to_hash.encode()).hexdigest()

        file_path = os.path.join(cache_path(), '{}.pt'.format(hash_value))
        if os.path.exists(file_path):
            self.load(file_path)
        else:
            print('cache {} does not exists, to be built...'.format(self.name))

        self.file_path = file_path
        self.to_hash = to_hash
        self.hash_value = hash_value

    def load(self, path):
        print('load cache {} from {}.'.format(self.name, path))

        self.data = torch.load(path)
        self.cached = True

    def cache(self, data):
        if is_test('cache') or self.cached:
            return data

        print('save cache {} to {}.'.format(self.name, self.file_path))

        torch.save(data, self.file_path)
        return data
        
