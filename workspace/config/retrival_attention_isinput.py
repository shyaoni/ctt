from torch import device
from utils import Config

embedding_dim = 200
vocab_size = 10000

config = Config({
    "predictor_type": "model.retrival_attention.Predictor_AttentionTriple",
    "num_epoch": 17,
    "vocab_size": vocab_size,
    "save_path": './save/retrival_attention_isinput.pt'
})

config.loader = Config({
    "batch_size": 30,
    "shuffle": True,
    "num_workers": 8,
})
config.test_loader = Config({
    "batch_size": 1,
    "shuffle": False
})

config.model = Config({
    "embedding": {
        "path": "./data/glove/glove.twitter.27B.{}d.txt".format(embedding_dim),
        "dim": embedding_dim,
        "vocab_size": vocab_size
    },
    "attention_mlp": {
        "hidden_size": 200,
        "is_input": True,
        "output_size": embedding_dim
    },
    "context_encoder": {
        "type": "nn.GRU",
        "kwargs": {
            "input_size": embedding_dim,
            "hidden_size": 300,
            "bidirectional": True,
            "batch_first": True
        },
        "flatten_code_size": 600,
        "flatten_output_size": 600
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
        "lr" : 0.0003
    }
})
config.lr_scheduler = Config({
    "type": "CosineAnnealingLR",
    "kwargs": {
        "T_max": 20,
        "eta_min": 0.00001,
    }
})
