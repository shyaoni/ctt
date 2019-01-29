import torch
import collections

class to_pack():
    """ Use this class in dataset to denotes a variant-length tensor to be collated. 
        1/2D list are supported, corresponding to utterance / dialog.
    """
    def __init__(self, lst):
        self.lst = lst

    def __iter__(self):
        for item in self.lst:
            yield item

    def tolist():
        return self.lst

def to_pack_consensus(packs):
    """compute the tensor from packs, padding with zero. 
    """
    lsts = [p.lst for p in packs]
    dim = []

    if isinstance(lsts[0][0], collections.Sequence):
        dim.append(max(len(lst) for lst in lsts))
        dim.append(max(len(x) for lst in lsts for x in lst))
    else:
        dim.append(max(len(lst) for lst in lsts)) 

    if len(dim) == 1:
        return [{
            'length': len(p.lst),
            'text_ids': torch.LongTensor(p.lst + [0,]*(dim[0]-len(p.lst)))}
            for p in packs]
    else:
        return [{
            'length': torch.LongTensor(
                [len(lst) for lst in p.lst] + [1,]*(dim[0]-len(p.lst))),
            'utterance_length': len(p.lst),
            'text_ids': torch.LongTensor(
                [lst + [0,]*(dim[1]-len(lst)) for lst in p.lst] + \
                [[0,]*dim[1]]*(dim[0]-len(p.lst)))}
            for p in packs]

def collate(batch):
    """collate_fn supports to_pack class. 
       Each sample must be a dict, with all to_pack items placed in first-level.
    """ 
    expanded_batch = [{} for _ in range(len(batch))]
    for key, value in batch[0].items():
        if isinstance(value, to_pack):
            for new, item in zip(
                expanded_batch, to_pack_consensus([b[key] for b in batch])):
                new[key] = item
        else:
            for new, old in zip(expanded_batch, batch):
                new[key] = old[key]
            
    return torch.utils.data.dataloader.default_collate(expanded_batch) 

def pack2data(pack):
    """convert a single pack to data.
    """
    return collate([{'pack': pack}])['pack']


