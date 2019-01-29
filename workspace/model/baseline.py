import numpy as np
import torch
import torch.nn as nn

from utils import Config, DataLoader, is_test, get_instance, to_device, prolog, \
    sigmoid_cross_entropy_with_logits, dynamic_rnn
import module.evaluation as evaluation

from module.retrival_naive_approaches import Predictor

from scipy.stats import truncnorm

class DualEncoder(nn.Module):
    def __init__(self, field, config):
        super().__init__()

        embedding = field.get_embedding_from_glove(config.embedding)
        self.embedding = nn.Embedding(
            num_embeddings=config.embedding.vocab_size + 4,
            embedding_dim=config.embedding.dim,
            padding_idx=0,
            _weight=embedding)

        self.context_encoder = get_instance(config.context_encoder)
        self.candidates_encoder = get_instance(config.candidates_encoder)

        transform_mat = torch.Tensor(truncnorm.rvs(-2, 2, size=(
            config.candidates_encoder.code_size,
            config.context_encoder.code_size)))
        transform_bias = torch.zeros(1)

        self.transform_mat = torch.nn.Parameter(transform_mat)
        self.transform_bias = torch.nn.Parameter(transform_bias)
        self.config = config

    def code_context(self, context):
        _, codes = self.context_encoder(
            self.embedding(context['text_ids']),
            context['length'],
            context['utterance_length'])
        return codes

    def code_candidates(self, candidates):
        inputs = candidates['text_ids']
        length = candidates['length']

        shape = inputs.shape[:2]
        inputs = self.embedding(inputs)

        _, codes = dynamic_rnn(
            self.candidates_encoder,
            inputs.view(-1, *inputs.shape[2:]), 
            length.view(-1, *length.shape[2:])) 
        return torch.matmul(
            codes.view(*shape, *codes.shape[1:]), self.transform_mat)

    def get_logits(self, context_code, candidates_code, batched=True):
        logits = (context_code.unsqueeze(1) * candidates_code).sum(dim=-1) + \
            self.transform_bias
        return logits

    def forward(self, context, candidates):
        context_code = self.code_context(context)
        candidates_code = self.code_candidates(candidates)
        logits = self.get_logits(context_code, candidates_code)
        return logits

class Baseline(Predictor):
    def __init__(self, field=None, **kwargs):
        super().__init__()
        self.config = Config(kwargs)
        self.field = field

        if field is not None:
            self.build()

    def build(self):
        self.model = DualEncoder(self.field, self.config)
        to_device(self.model)

    def get_logits(self, data):
        return self.model(
            data['context'], data['candidates'])

    def default_loss(self, logits, label, num_candidates):
        return sigmoid_cross_entropy_with_logits(logits, label, num_candidates)

    def train_epoch(self, epoch_id, loader, optimizer):
        loss_cnt, acc_cnt = [], []
        for batch_id, data in enumerate(loader):
            data = to_device(data)
            logits = self.get_logits(data)
            loss = self.default_loss(logits, data['label'], 
                data['num_candidates'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = evaluation.torch_acc(logits, data['label'])

            loss_cnt.append(loss.item())
            acc_cnt.append(acc)

            print('{}: loss={:.5f}, acc={:5f}   '.format(batch_id, 
                np.mean(loss_cnt), np.mean(acc_cnt)), end='\r') 

        print('epoch {}: loss={:.5f}, acc={:5f}   '.format(
            epoch_id, np.mean(loss_cnt), np.mean(acc_cnt)))

    def test(self, dataset, config):
        if dataset is None:
            return -1

        loader = DataLoader(dataset, **config.data.test_set.loader.to_dict())
        loss_cnt, acc_cnt = [], []
        with torch.no_grad():
            for batch_id, data in enumerate(prolog(loader)):
                data = to_device(data)
                logits = self.get_logits(data)
                
                loss = self.default_loss(logits, data['label'],
                    data['num_candidates'])
                acc = evaluation.torch_acc(logits, data['label'])

                loss_cnt.append(loss.item())
                acc_cnt.append(acc)

        print('test loss={:.5f}, acc={:5f}, '.format(
            np.mean(loss_cnt), np.mean(acc_cnt)))
        return np.mean(acc_cnt)

    def train(self, dataset, config, dts_valid=None):
        loader = DataLoader(dataset, **config.data.train_set.loader.to_dict())
        optimizer = config.optimizer.get_optimizer(self.model.parameters())
        lr_scheduler = config.lr_scheduler.get_lr_scheduler(optimizer)

        for epoch_id in range(config.num_epoch):
            self.train_epoch(epoch_id, loader, optimizer)
            self.train_log_valid(self.test(dts_valid, config))

            lr_scheduler.step()

        self.train_back_to_best()   

    def get_code(self, data, cands=False):
        if not cands:
            return self.model.code_context(data)

        return self.model.code_candidates({
            'text_ids': data['text_ids'].unsqueeze(1),
            'length': data['length'].unsqueeze(1)
        }).squeeze(1)

    def build_corpus(self, dataset, config):
        loader = DataLoader(dataset, **config.loader.to_dict())
        corpus = []
        with torch.no_grad():
            for batch_id, data in enumerate(prolog(loader, 'build corpus')):
                data = to_device(data)
                codes = self.get_code(data['uttr'], cands=True)

                for code, text_ids in zip(codes, data['uttr']['text_ids']):
                    corpus.append((text_ids, code))

        text_ids, codes = zip(*corpus)
        codes = torch.stack(codes, dim=0)
        return text_ids, codes

    def retrieve(self, data, corpus, k=1):
        data = to_device(data)
        with torch.no_grad():
            context_code = self.get_code(data)
            candidates_code = corpus[1]
            probs = self.model.get_logits(
                context_code, candidates_code).squeeze(0)

        probs, ids = torch.topk(probs, k=k)
        probs = torch.sigmoid(probs)
        return probs, ids

    def save(self, path):
        torch.save({
            'field': self.field,
            'state_dict': self.model.state_dict()
        }, path)

    def load(self, path):
        meta = torch.load(path)
        self.field = meta['field']
        self.build()
        self.model.load_state_dict(meta['state_dict'])

if __name__ == '__main__':
    pass
