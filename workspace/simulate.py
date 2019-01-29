import torch

import utils
from data import data_parser, get_dataset

import module.tokenizer
module.tokenizer.tokenize_pipeline()

import module.chat as chat

if __name__ == '__main__':
    config, args = utils.get_config(data_parser())
    dataset = get_dataset(args.dts)

    vocab = dataset.get_vocab(config.vocab_size)
    predictor = utils.get_instance(config.predictor_type, 
        None, config.model)
    predictor.load(config.save_path)

    train_set = dataset.utterance(vocab, train=True)

    chatbot = chat.Chatbot(chat.InferGraphAgent(
        predictor).corpus(train_set, config).build(width=100))

    from IPython import embed
    embed()
