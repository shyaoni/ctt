import torch
import numpy as np
from utils import prolog

def intersect_rate(a, b, i):
    return sum(1 for x in b if x in a) / (len(b) if i else len(a))

class Evaluations():
    def __init__(self, predictor, data):
        self.predictor = predictor
        self.data = data 
        self.res = [predictor.predict(item['context'], item['utterance'])
                    for item in prolog(data)]

    def recall(self, k=1):
        recall_cnt = 0
        for (pred, _), item in zip(self.res, self.data):
            recall_cnt += intersect_rate(data['label'], pred[:k], 0)
        return recall_cnt / len(self.res)

    def precision(self, k=1):
        precision_cnt = 0
        for (pred, _), item in zip(self.res, self.data):
            precision_cnt += intersect_rate(data['label'], pred[:k], 1)
        return precision_cnt / len(self.res)

def torch_acc(logits, label, k=1):
    _, indices = torch.topk(logits, k, dim=-1)
    cnt = 0
    for a, b in zip(label, indices):
        cnt += a.item() in b.tolist()
    return cnt / len(logits)
