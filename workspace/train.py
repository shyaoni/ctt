import torch

import utils
from data import data_parser, get_dataset

import module.tokenizer
module.tokenizer.tokenize_pipeline()

if __name__ == '__main__':
    config, args = utils.get_config(data_parser())
    dataset = get_dataset(args.dts)

    vocab = dataset.get_vocab(config.data.vocab_size)
    train_set = dataset.get_set(config.data.train_set, vocab)
    test_set = dataset.get_set(config.data.test_set, vocab)

    predictor = utils.get_instance(config.model, train_set.field)

    predictor.train(train_set, config, dts_valid=test_set)
    predictor.save(config.save_path)
