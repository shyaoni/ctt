import torch

import utils
from data import data_parser, get_dataset

import module.tokenizer
module.tokenizer.tokenize_pipeline()

torch.multiprocessing.set_sharing_strategy('file_system')

from module.chat import Sess, Chatbot, InferGraphAgent

import importlib

def sess(chatbot):
    return Sess(chatbot)

def chatbot(config_path):
    config = importlib.import_module(config_path).config
    dataset = get_dataset('dts_ConvAI2')
    vocab = dataset.get_vocab(config.data.vocab_size)
    corpus_set = dataset.get_set(config.data.corpus_set, vocab)
    predictor = utils.get_instance(config.model)
    predictor.load(config.save_path)
    chatbot = Chatbot(InferGraphAgent(predictor, id='agent').corpus(
        corpus_set, config.data.corpus_set).build())
    chatbot.send_target('work')

    return chatbot


