import os
import collections
import torch
import numpy as np
import json
from copy import deepcopy

from utils import tokens2str, str2data, is_test
from module.infer_graph import InferGraph

log_path = './log'
log_file = 'log.json'

class Logger():
    def __init__(self, path=log_path):
        if not os.path.exists(path):
            os.makedirs(path)

        self.log_path = os.path.join(log_path, log_file)

        if os.path.exists(self.log_path):
            self.logs = json.load(open(self.log_path, 'r'))
        else:
            self.logs = {}
            self.dump()

    def get_log(self, gp_name):
        if gp_name not in self.logs:
            self.logs[gp_name] = []
        return self.logs[gp_name]

    def save(self, gp_name, id, dialog, comments=''):
        self.get_log(gp_name).append({
            'id': id,
            'dialog': dialog,
            'comments': comments})

    @staticmethod
    def log_as_str(logs):
        spliter = '\n\n------------\n\n'
        spliter.join(['{}:\n({})\n{}'.format(
            log['id'], log['comments'], '\n'.join(log['dialog']))
            for log in logs])
        return spliter

    def dump(self):
        json.dump(self.logs, open(self.log_path, 'w'))

logger = Logger()

class Sess():
    def __init__(self, chatbot, id='sess', history=[]):
        self.chatbot = chatbot
        self.id = id
        self._history = deepcopy(history)

    @property
    def history(self):
        return self._history

    @property
    def raw_history(self):
        if len(self._history) == 0:
            return ['']
        return [uttr[uttr.find(':')+2:] for uttr in self._history]

    def infer(self, **kwargs):
        return self.chatbot.infer(self.raw_history, **kwargs)

    def step(self, string=None):
        if string is None:
            human_step()
        else:
            self.history.append(string)

    def infer_step(self, sample='greedy', **kwargs):
        self.step(self.chatbot.get(*self.infer(sample=sample, **kwargs)))

    def human_step(self):
        string = input('human: ')
        self.history.append('human: {}'.format(string))

    def copy(self, end_at, id_suf='copy'):
        return Sess(self.predictor, self.history[:end_at], '{}.{}'.format(
            self.id, id_suf))

    def save(self, gp_name=None, comments=''):
        dialog = deepcopy(self.history)
        logger.save(gp_name, self.id, deepcopy(self.history), comments)

    def send_target(self, target, *args, **kwargs):
        self.chatbot.send_target(target, *args, **kwargs)
        self.target = target

class Chatbot():
    def __init__(self, *agents, name='chatbot'):
        self.agents = list(agents)
        self.atoi = {agent.id:agents.index(agent) for agent in self.agents}
        self.name = name

    def infer(self, history, sample=None, **kwargs):
        results = {}
        for agent in self.agents:
            results[agent.id] = agent.infer(history, **kwargs)
        self._result_cache = results

        if sample is None:
            return results
        elif sample == 'random':
            v = list(zip(*list(results.items())))
            id = 0
            return v[0][id], np.random.choice(
                len(v[1][id]), p=list(zip(*v[1][id]))[1])
        elif sample == 'greedy':
            assert len(results) == 1
            t = list(zip(*results[self.agents[0].id]))
            return self.agents[0].id, np.argmax(t[1])
        else:
            raise NotImplementedError("sample={} is not supported.".format(
                sample))

    def get(self, id, num, result=None):
        if result is None:
            result = self._result_cache
        if not isinstance(id, int):
            id = self.atoi[id]
        return '{}: {}'.format(self.agents[id].id,
            self.agents[id].get(result[self.agents[id].id][num]))

    def sess(self, *args, **kwargs):
        return Sess(self, *args, **kwargs)

    def send_target(self, target, *args, **kwargs):
        for agent in self.agents:
            agent.send_target(target, *args, **kwargs)

class Agent():
    def __init__(self, predictor, id='agent'):
        self.id = id
        self.predictor = predictor
        self._corpus = [[], None]

    def corpus(self, dts, config):
        text_ids, code = self.predictor.build_corpus(dts, config)
        self._corpus[0].extend(text_ids)
        self._corpus[1] = torch.cat([self._corpus[1], code], dim=0) \
            if self._corpus[1] is not None else code
        return self

    def infer(self, history, **kwargs):
        data = str2data(history[-1], self.predictor.field)

        results = []
        for prob, id in zip(
            *self.predictor.retrieve(data, self._corpus, **kwargs)):
            results.append(
                (tokens2str(self._corpus[0][id], self.predictor.field), prob))
        return results

    @staticmethod
    def get(result):
        return result[0]

class InferGraphAgent():
    def __init__(self, predictor, id='agent'):
        self.id = id
        self.predictor = predictor
        self._corpus = [[], None]

    def corpus(self, dts, config):
        text_ids, code = self.predictor.build_corpus(dts, config)
        self._corpus[0].extend(text_ids)
        self._corpus[1] = torch.cat([self._corpus[1], code], dim=0) \
            if self._corpus[1] is not None else code
        return self

    def build(self, width=20):
        self.infer_graph = InferGraph(self.predictor, self._corpus, width)
        return self

    def send_target(self, target):
        self.infer_graph.build_for_target(target, max_turns=8)

    def infer(self, history, **kwargs):
        if len(history) > 1:
            history = history[-1:]
        data = str2data(history, self.predictor.field)
        probs, ids = [], []
        for prob, id in zip(
            *self.predictor.retrieve(data, self._corpus, **kwargs)):
            probs.append(prob * self.infer_graph.p[id])
            ids.append(id)

        results = [(tokens2str(self._corpus[0][id], self.predictor.field), prob)
                   for id, prob in zip(ids, probs)]
        return results

    @staticmethod
    def get(result):
        return result[0]

class MonteCarloInferGraphAgent(InferGraphAgent):
    def infer(self, history, **kwargs):
        raise NotImplementedError


