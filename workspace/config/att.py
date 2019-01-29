from utils import Config

vocab_size = 10000
embedding_dim = 50

context_rnn_minor_hsize = 50
context_rnn_minor_isbi = True

context_rnn_major_hsize = 100
context_rnn_major_isbi = True

candidates_rnn_hsize = 50
candidates_rnn_isbi = True

config = Config({
    "save_path": './save/att.pt',
    "num_epoch": 12,
})
config.data = Config({
    "vocab_size": vocab_size,
    "train_set": {
        "type": "utterance_with_cands_from_negative_sampling",
        "kwargs": {
            "train": True,
            "num_negs": 19,
            "num_turns": 3
        },
        "loader": {
            "batch_size": 30,
            "shuffle": True,
            "num_workers": 8,
        }
    },
    "test_set": {
        "type": "utterance_with_cands",
        "kwargs": {
            "train": False, 
            "num_turns": 3
        },
        "loader": {
            "batch_size": 1,
            "shuffle": False,
        }
    },
    "corpus_set": {
        "type": "utterance",
        "kwargs": {
            "train": True
        },
        "loader": {
            "batch_size": 30,
            "shuffle": False,
            "num_workers": 8
        }
    }
})
config.model = Config({
    "type": "model.baseline.Baseline",
    "kwargs": {
        "embedding": {
            "vocab_size": vocab_size,
            "path": "./data/glove/glove.twitter.27B.{}d.txt".format(embedding_dim),
            "dim": embedding_dim,
        },
        "context_encoder": {
            "type": "model.hierarchical_encoder.HierarchicalRNN",
            "kwargs": {
                "minor_encoder": {
                    "type": "torch.nn.GRU",
                    "kwargs": {
                        "input_size": embedding_dim,
                        "hidden_size": context_rnn_minor_hsize,
                        "bidirectional": context_rnn_minor_isbi,
                        "batch_first": True
                    }
                },
                "major_encoder": {
                    "type": "torch.nn.GRU",
                    "kwargs": {
                        "input_size": context_rnn_minor_hsize * \
                            (context_rnn_minor_isbi + 1),
                        "hidden_size": context_rnn_major_hsize,
                        "bidirectional": context_rnn_major_isbi,
                        "batch_first": True
                    }
                }
            },
            "code_size": context_rnn_major_hsize * \
                (context_rnn_major_isbi + 1)
        },
        "candidates_encoder": {
            "type": "torch.nn.GRU",
            "kwargs": {
                "input_size": embedding_dim,
                "hidden_size": candidates_rnn_hsize,
                "bidirectional": candidates_rnn_isbi,
                "batch_first": True
            },
            "code_size": candidates_rnn_hsize * (candidates_rnn_isbi + 1)
        },
        "with_attention": True
    }
})

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
