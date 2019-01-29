import torch

import utils
from data import data_parser, get_dataset

import module.tokenizer
module.tokenizer.tokenize_pipeline()

torch.multiprocessing.set_sharing_strategy('file_system')

from module.chat import Sess, Chatbot, InferGraphAgent

if __name__ == '__main__':
    config, args = utils.get_config(data_parser())
    dataset = get_dataset(args.dts)

    vocab = dataset.get_vocab(config.data.vocab_size)
    corpus_set = dataset.get_set(config.data.corpus_set, vocab)

    predictor = utils.get_instance(config.model)
    predictor.load(config.save_path)

    chatbot = Chatbot(InferGraphAgent(predictor, id='agent').corpus(
        corpus_set, config.data.corpus_set).build())

    from IPython import embed
    embed()

