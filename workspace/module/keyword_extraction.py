import nltk

import utils
import module.tokenizer as tokenizer
tokenizer.tokenize_pipeline()

from data.stopword import is_stopword

class KeywordExtractor():
    def __init__(self, field):
        self.ppln = [tokenizer.tokenize, tokenizer.pos_tag]
        self.field = field

    @staticmethod
    def is_keyword_tag(tag):
        return tag.startswith('VB') or tag.startswith('NN') or \
            tag.startswith('JJ')

    def extract(self, string):
        tokens = utils.tokenize(string, self.ppln)
        source = utils.tokenize(string)

        kwpos_alters = []
        for i, (word, tag) in enumerate(tokens):
            if source[i] in self.field.stoi and self.is_keyword_tag(tag):
                kwpos_alters.append(i)

        tokens = utils.tokenize(tokens, [
            tokenizer.truecasing, tokenizer.pos_tag, 
            tokenizer.to_basic_form, tokenizer.lower])

        kwpos, keywords = [], []
        for id in kwpos_alters:
            if not is_stopword(tokens[id]):
                kwpos.append(id)
                keywords.append(tokens[id])

        return {
            'kwpos': kwpos,
            'keywords': keywords
        }
