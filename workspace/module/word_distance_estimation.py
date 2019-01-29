import numpy as np
import collections

from utils import prolog

class WordDistanceEstimator():
    def __init__(self):
        pass

    def ngd(self, a, b):
        if self.counter_map[(a, b)] == 0:
            return 1e10
        loga = np.math.log(self.counter[a])
        logb = np.math.log(self.counter[b])
        logab = np.math.log(self.counter_map[(a, b)])
        return (max(loga, logb) - logab) / (self.logM - min(loga, logb))

    def build(self, dts):
        counter = collections.defaultdict(int)

        counter_map = collections.defaultdict(int)
        for a, b in prolog(dts.pairs, name='count from pairs'):
            a_lst, b_lst = [], []
            for i in a:
                a_lst += dts.examples[i]['keywords']
            b_lst = dts.examples[b]['keywords']

            for x in set(a_lst):
                for y in set(b_lst):
                    counter_map[(x, y)] += 1 
                    #if x != y:
                    #    counter_map[(y, x)] += 1

            for x in set(a_lst + b_lst):
                counter[x] += 1

        self.counter = counter
        self.counter_map = counter_map
        self.logM = np.math.log(len(dts.pairs))

        from IPython import embed
        embed()

    def get_ranks(self, a):
        res = {}
        for b in self.counter.keys():
            res[b] = self.ngd(a, b)

        r = sorted(list(res.items()), key=lambda x: x[1])
        r = [(x, y, self.counter[x], self.counter_map[(a, x)]) for x, y in r]
        return [a for a in r if a[-1] > 1 and a[-2] >= 20]
