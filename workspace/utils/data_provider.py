import torch
import torch.utils.data

from .to_pack_cls import collate

class Dataset(torch.utils.data.Dataset):
    def __init__(self, examples, field):
        self.examples = examples
        self.field = field

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

class DataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collate_fn = collate
