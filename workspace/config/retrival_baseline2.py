from torch import device
from utils import Config

vocab_size = 10000
embedding_dim = 200

config = Config({
    "predictor_type": "model.retrival_baseline.Predictor_DualRNNEncoder",
    "num_epoch": 12,
    "vocab_size": vocab_size,
    "save_path": './save/retrival_baseline2.pt'
})
config.train_set = Config({
    "type": "utterance_with_cands",
    "kwargs": {
        "train": True,
    },
    "loader": {
        "batch_size": 30,
        "shuffle": True,
        "num_workers": 8,
    }
})
config.test_set = Config({
    "type": "utterance_with_cands",
    "kwargs": {
        "train": False, 
    },
    "loader": {
        "batch_size": 1,
        "shuffle": False,
    }
})
config.corpus_set = Config({
    "loader": {
        "batch_size": 30,
        "shuffle": False,
        "num_workers": 8
    }
})

config.model = Config({
    "embedding": {
        "vocab_size": vocab_size,
        "path": "./data/glove/glove.twitter.27B.{}d.txt".format(embedding_dim),
        "dim": embedding_dim,
    },
    "context_encoder": {
        "type": "nn.GRU",
        "kwargs": {
            "input_size": embedding_dim,
            "hidden_size": 300,
            "bidirectional": True,
            "batch_first": True
        },
        "flatten_code_size": 600
    },
    "candidates_encoder": {
        "type": "nn.GRU",
        "kwargs": {
            "input_size": embedding_dim,
            "hidden_size": 300,
            "bidirectional": True,
            "batch_first": True
        },
        "flatten_code_size": 600
    }})

config.optimizer = Config({
    "type": "Adam",
    "kwargs": {
        "lr": 0.001
    }
})
config.lr_scheduler = Config({
    "type": "CosineAnnealingLR",
    "kwargs": {
        "T_max": 8,
        "eta_min": 0.0001,
    }
})
