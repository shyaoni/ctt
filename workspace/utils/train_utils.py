import sys 

class valid_logger():
    def __init__(self):
        self.values = []

    def append(self, value):
        self.values.append(value)

    def last_is_best(self):
        if len(self.values) == 1:
            return True
        return self.values[-1] >= max(self.values[:-1])

def random_tmp_pt():
    return '/tmp/{}.pt'.format(hex(sys.maxsize + 1 + hash('train')))
