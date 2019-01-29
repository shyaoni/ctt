import numpy as np
import collections
import itertools

import data.convai2
from utils import is_test, prolog, tokenize, preprocess, cacher, to_pack, \
    Dataset, Field
from module.keyword_extraction import KeywordExtractor

class NegativeSamplingDataset(Dataset):
    def __init__(self, examples, pairs, field):
        self.examples = examples
        self.pairs = pairs
        self.num_negs = None
        self.field = field

    def negs(self, num_negs):
        self.num_negs = num_negs
        return self
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, id):
        qs, r = self.pairs[id]
        cands = [r, ] + [np.random.randint(len(self.examples)) 
                for _ in range(self.num_negs)]

        return {
            'context': to_pack([self.examples[q]['uttr'].lst for q in qs]),
            'kwpos': [self.examples[q]['kwpos'] for q in qs],
            'keywords': [self.examples[q]['keywords'] for q in qs],
            'keywords_target': self.examples[r]['keywords'],
            'candidates': to_pack([self.examples[c]['uttr'].lst for c in cands]),
            'label': 0,
            'num_candidates': len(cands)
        }

class dts_ConvAI2(data.convai2.dts_ConvAI2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_vocab(self, vocab_size=10000):
        cache = cacher('dts_ConvAI2.get_vocab', vocab_size)
        if cache.cached:
            return cache.data

        counter = collections.Counter()
        dialogs = self.get_dialogs()

        for dialog in prolog(dialogs):
            for uttr in dialog:
                counter.update(tokenize(uttr))

        return cache.cache([token for token, times in sorted(
            list(counter.items()), key=lambda x: (-x[1], x[0]))][:vocab_size])

    def raw_utterance(self, train=True):
        cache = cacher('dts_ConvAI2.raw_utterance', train)
        if cache.cached:
            return cache.data

        corpus = self.get_data(train=train, cands=False)

        examples = []
        check_dict = {}
        for sess in prolog(corpus, name=' -extract utterances from dialog'):
            for i, uttr in enumerate(sess['dialog']):
                if uttr not in check_dict:
                    examples.append({'uttr': to_pack(uttr)})
                    check_dict[uttr] = len(examples) - 1

        return cache.cache((examples, corpus, check_dict))

    def raw_utterance_with_keyword(self, vocab, train=True):
        cache = cacher('dts_ConvAI2.raw_utterance_with_keyword', vocab, train)
        if cache.cached:
            return cache.data

        examples, corpus, check_dict = self.raw_utterance(train)
        field = Field(vocab)

        kwext = KeywordExtractor(field)
        for example in prolog(examples, name=' -extract keywords'):
            kws = kwext.extract(example['uttr'].lst)
            example['kwpos'] = kws['kwpos'] 
            example['keywords'] = kws['keywords'] 

        examples = preprocess(examples, field, log=' -process to_pack cls')
        return cache.cache((examples, field, corpus, check_dict))

    def utterance(self, vocab, train=True):
        examples, field = self.raw_utterance_with_keyword(vocab, train)[:2]
        return Dataset(examples, field)

    def utterance_with_cands_from_negative_sampling(self, 
                                                    vocab, 
                                                    train=True,
                                                    num_turns=1,
                                                    num_negs=20):
        examples, field, corpus, check_dict = \
            self.raw_utterance_with_keyword(vocab, train)

        pairs = []
        for sess in prolog(corpus, name='collect pairs'):
            for id, uttr in enumerate(sess['dialog']):
                if id % 2 == 1:
                    for xs in range(min(num_turns, id)): 
                        pairs.append((
                            [check_dict[sess['dialog'][id-x-1]] 
                                for x in range(xs+1)], check_dict[uttr]))

        return NegativeSamplingDataset(examples, pairs, field).negs(num_negs)

    def utterance_with_cands(self, vocab, train=True, num_turns=1):
        cache = cacher('dts_ConvAI2.utterance_with_cands', 
            vocab, train, num_turns)
        if cache.cached:
            return cache.data
        
        uttrs, field, corpus, check_dict = \
            self.raw_utterance_with_keyword(vocab, train)

        corpus = self.get_data(train=train, cands=True)

        examples = []
        for sess in prolog(corpus, name='mapping utterances...'):
            for id, candidates in enumerate(sess['candidates']): 
                if candidates is not None:
                    xs = min(num_turns, id+1)
                    examples.append({
                        'context': to_pack([
                            uttrs[check_dict[sess['dialog'][id-x]]]['uttr'].lst
                            for x in range(xs)]),
                        'kwpos': [
                            uttrs[check_dict[sess['dialog'][id-x]]]['kwpos']
                            for x in range(xs)], 
                        'keywords': list(itertools.chain([
                            uttrs[check_dict[sess['dialog'][id-x]]]['keywords']
                            for x in range(xs)])),
                        'keywords_target': 
                            uttrs[check_dict[candidates[-1]]]['keywords'],
                        'candidates': to_pack(preprocess(
                            to_pack(candidates[:-1]), field).lst + [
                                uttrs[check_dict[candidates[-1]]]['uttr'].lst]),
                        'label': len(candidates) - 1, 
                        'num_candidates': len(candidates)
                    })

        return cache.cache(Dataset(examples, field)) 

    def get_set(self, config, *args, **kwargs):
        if hasattr(config, 'kwargs'):
            kwargs.update(config.kwargs.to_dict())
        return getattr(self, config.type)(*args, **kwargs)
        

