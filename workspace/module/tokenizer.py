import nltk
from nltk.stem import WordNetLemmatizer

import utils

_lemmatizer = WordNetLemmatizer()

def is_keyword_tag(tag):
    return tag.startswith('NN') or tag.startswith('VB') or \
        tag.startswith('JJ')

def tokenize(string):
    return nltk.word_tokenize(string)

def truecasing(tokens):
    ret = []
    is_start = True
    for word, tag in tokens:
        if word == 'i':
            ret.append('I')
        elif tag[0].isalpha():
            if is_start:
                ret.append(word[0].upper() + word[1:])
            else:
                ret.append(word)
            is_start = False
        else:
            if tag != ',':
                is_start = True
            ret.append(word)
    return ret

def pos_tag(tokens):
    return nltk.pos_tag(tokens)

def to_basic_form(tokens): 
    if not isinstance(tokens, tuple):
        return [to_basic_form(token) for token in tokens] 
    word, tag = tokens
    if tag.startswith('NN'):
        pos = 'n'
    elif tag.startswith('VB'):
        pos = 'v'
    elif tag.startswith('JJ'):
        pos = 'a'
    else:
        return word
    return _lemmatizer.lemmatize(word, pos)

def lower(tokens):
    if not isinstance(tokens, str):
        return [lower(token) for token in tokens]
    return tokens.lower()

############# initialization ###########

def tokenize_pipeline():
    utils.tokenize_pipeline([tokenize, lower])


