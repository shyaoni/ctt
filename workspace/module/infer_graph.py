import numpy as np
import torch
from scipy.sparse import csc_matrix

from module.keyword_extraction import KeywordExtractor

from utils import prolog, str2data, tokens2str, is_test, cacher

def ids2data(text_ids, field):
    return str2data([tokens2str(text_ids, field)], field)

class InferGraph():
    def __init__(self, predictor, corpus, width=20):
        field = predictor.field
        text_ids = corpus[0]

        self.predictor = predictor
        self.corpus = corpus
        self.field = self.predictor.field

        self.build_graph(width=width)
        self.build_kws_for_corpus()

    def build_graph(self, width):
        text_ids = self.corpus[0]
        corpus_size = len(self.corpus[0])

        cache = cacher('infergraph.build_graph', width)
        if cache.cached:
            self.transition_mat = cache.data
            return

        row, col, data = [], [], [] 
        for idx, text_ids in enumerate(
            prolog(self.corpus[0], name='infer graph: build graph')):
            probs, ids = self.predictor.retrieve(
                ids2data(text_ids, self.field), self.corpus, k=width) 
            probs = torch.nn.functional.softmax(probs, dim=0)

            for prob, idy in zip(probs.tolist(), ids.tolist()): 
                row.append(idx)
                col.append(idy)
                data.append(prob)
        
        self.transition_mat = cache.cache(
            csc_matrix((data, (row, col)), shape=(corpus_size, corpus_size)))

    def build_kws_for_corpus(self, *args):
        kwext = KeywordExtractor(self.predictor.field)

        cache = cacher('infergraph.build_kws_for_corpus', *args)
        if cache.cached:
            self.keywords = cache.data
            return

        keywords = []
        for idx, text_ids in enumerate(
            prolog(self.corpus[0], name='infer graph: build kws')):
            keywords.append(
                kwext.extract(tokens2str(text_ids, self.field))['keywords'])

        self.keywords = cache.cache(keywords)

    def build_for_target(self, target, max_turns=8):
        corpus_size = len(self.corpus[0])
        p = np.zeros(corpus_size)

        for i in range(len(self.corpus[0])):
            if target in self.keywords[i]:
                p[i] = 1

        mask = 1 - p
        p = [self.transition_mat.dot(p) * mask]

        for i in range(max_turns): 
            p.append(self.transition_mat.dot(
                self.transition_mat.multiply(
                    p[-1]).max(axis=-1).todense().getA().reshape(-1)) * mask)

        if is_test('build_for_target'):
            from IPython import embed
            embed()

        self.p = sum(p)

            
