import torch

import utils
from data import data_parser, get_dataset

import module.tokenizer
module.tokenizer.tokenize_pipeline()

import module.word_distance_estimation

if __name__ == '__main__':
    config, args = utils.get_config(data_parser())
    dataset = get_dataset(args.dts)

    vocab = dataset.get_vocab(config.vocab_size)
    test_set = dataset.utterance_with_cands_from_negative_sampling(
        vocab, train=True)

    wde = module.word_distance_estimation.WordDistanceEstimator()
    wde.build(test_set)

    from IPython import embed
    embed()
