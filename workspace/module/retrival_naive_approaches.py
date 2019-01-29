import os
import numpy as np
from utils.train_utils import valid_logger, random_tmp_pt

class Predictor():
    def __init__(self):
        self._valid_logger = valid_logger()

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def retrieve(self, *args, **kwargs):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def train_log_valid(self, value):
        self._valid_logger.append(value)

        if self._valid_logger.last_is_best():
            self.save(random_tmp_pt())

    def train_back_to_best(self):
        self.load(random_tmp_pt()) 
        os.remove(random_tmp_pt())

class RandomPredictor(Predictor):
    def __init__(self):
        from random import shuffle
        self.random_shuffle = shuffle
        pass

    def train(self, *args, **kwargs):
        pass

    def predict(self, context, utterances):
        return self.random_shuffle(list(range(len(utterances)))), None 

if __name__ == '__main__':
    pass
