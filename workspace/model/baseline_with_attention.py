import numpy as np
import torch
import torch.nn as nn

from utils import get_torch_instance, pad_packed_sequence, \
    pack_padded_sequence, pack_order_inv, pack_order, to_device, prolog, \
    masked, DataLoader, sigmoid_cross_entropy_with_logits, \
    debug_inflags, loss_check
import module.evaluation as evaluation

from module.retrival_naive_approaches import Predictor

from scipy.stats import truncnorm

def truncnorm_mat_params(size):
    return torch.nn.Parameter(
        torch.Tensor(truncnorm.rvs(-2, 2, size=size)))

class Retrival_AttentionTriple_AttentionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w = truncnorm_mat_params((
            config.context_encoder.flatten_output_size,
            config.attention_mlp.hidden_size))
        self.u = truncnorm_mat_params((
            config.context_encoder.flatten_code_size,
            config.attention_mlp.hidden_size))
        self.v = truncnorm_mat_params(
            (1, 1, config.attention_mlp.hidden_size))

        self.is_input = config.attention_mlp.is_input

    def get_attention_vec(self, code_context, output):
        a = (torch.tanh(
            torch.matmul(code_context, self.u).unsqueeze(1) + \
            torch.matmul(output, self.w)) * self.v).sum(dim=-1)
        a = a - a.max(dim=-1, keepdim=True)[0]
        e = nn.functional.softmax(a, dim=-1).unsqueeze(-1)
        return e

    def forward(self, code_context, output, input, length=None):
        e = self.get_attention_vec(code_context, output)
        if length is not None:
            output = masked(output, length)
            input = masked(input, length)

        if self.is_input:
            return (e * input).sum(dim=1)
        else:
            return (e * output).sum(dim=1)

class Retrival_AttentionTriple(nn.Module):
    def __init__(self, field, config):
        super().__init__()

        embedding = None
        if field is not None:
            embedding = field.get_embedding_from_glove(config.embedding)

        self.embedding = nn.Embedding(
            num_embeddings=config.embedding.vocab_size + 4,
            embedding_dim=config.embedding.dim,
            padding_idx=0,
            _weight=embedding) 

        self.context_encoder = get_torch_instance(config.context_encoder)
        self.candidates_encoder = get_torch_instance(config.candidates_encoder)

        self.attention_mlp = Retrival_AttentionTriple_AttentionMLP(config)

        self.transform_mat = truncnorm_mat_params((
            config.context_encoder.flatten_code_size + \
            config.attention_mlp.output_size,
            config.candidates_encoder.flatten_code_size)) 
        self.transform_bias = truncnorm_mat_params((1, )) 

        self.config = config

    def code_context(self, inputs, sequence_length):
        inputs = self.embedding(inputs) 
        output, codes = self.context_encoder(
            pack_padded_sequence(
                inputs, sequence_length, 
                batch_first=True))
        output, length = pad_packed_sequence(output, batch_first=True)
        return codes.permute(1, 0, 2)[pack_order_inv()].view(
            len(pack_order_inv()), -1), output[pack_order_inv()]

    def code_candidates(self, inputs, sequence_length, attention):
        inputs = self.embedding(inputs)

        packed_input = pack_padded_sequence(
            inputs.view((-1, ) + inputs.shape[2:]),
            sequence_length.view(-1),
            batch_first=True)

        _, codes = self.candidates_encoder(packed_input)
        codes = codes.permute(1, 0, 2)[pack_order_inv()].view(
            len(pack_order_inv()), -1)
        return codes.view(inputs.shape[:2] + codes.shape[1:])

    def forward(self, context_text_ids, context_length, 
                candidates_text_ids, candidates_length):
        context_code, context_output = self.code_context(
            context_text_ids, context_length)
        attention_code = self.attention_mlp(
            context_code, context_output, 
            self.embedding(context_text_ids), context_length)

        context_code = torch.cat([context_code, attention_code], dim=-1)

        context_code_transformed = \
            torch.matmul(context_code, self.transform_mat)

        candidates_code = self.code_candidates(
            candidates_text_ids, candidates_length, attention_code)

        logits = (candidates_code * \
            context_code_transformed.unsqueeze(dim=1)).sum(dim=-1) + \
            self.transform_bias
        return logits 

class Predictor_AttentionTriple(Predictor):
    def __init__(self, field=None, config=None):
        self.config = config
        self.field = field

        if field is not None:
            self.build()

    def build(self):
        self.model = Retrival_AttentionTriple(self.field, self.config)
        to_device(self.model)

    def default_loss(self, logits, label):
        return sigmoid_cross_entropy_with_logits(logits, label)

    def get_logits(self, data):
        return self.model(
            context_text_ids=data['context']['text_ids'],
            context_length=data['context']['length'],
            candidates_text_ids=data['candidates']['text_ids'],
            candidates_length=data['candidates']['length'])

    def train(self, dataset, config):
        loader = DataLoader(dataset, **config.loader.to_dict())
        optimizer = config.optimizer.get_optimizer(self.model.parameters())
        lr_scheduler = config.lr_scheduler.get_lr_scheduler(optimizer)

        for epoch_id in range(config.num_epoch):
            loss_cnt, acc_cnt = [], []
            for batch_id, data in enumerate(loader): 
                data = to_device(data)
                logits = self.get_logits(data)
                loss = self.default_loss(logits, data['label'])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                acc = evaluation.torch_acc(logits, data['label'])

                loss_cnt.append(loss.item())
                acc_cnt.append(acc)

                print('{}: loss={:.5f}, acc={:5f}   '.format(batch_id, 
                    np.mean(loss_cnt), np.mean(acc_cnt)), end='\r')

            print('epoch {}: loss={:.5f}, acc={:5f}   '.format(epoch_id,
                np.mean(loss_cnt), np.mean(acc_cnt)))

            lr_scheduler.step() 

    def test(self, dataset, config):
        loader = DataLoader(dataset, **config.test_loader.to_dict())

        cnt = 0
        for batch_id, data in enumerate(prolog(loader)):
            data = to_device(data)
            logits = self.get_logits(data)
            cnt += evaluation.torch_acc(logits, data['label'])

        print('acc={}'.format(cnt / len(loader)))

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
