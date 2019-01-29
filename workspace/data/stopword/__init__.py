import os

stopwords_type = ['smart', 'nltk']

_stopwords_file = [os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 
        '{}.txt'.format(typ))
    for typ in stopwords_type]

_stopwords = {t: [x.strip() for x in open(f, 'r').readlines()] 
    for t, f in zip(stopwords_type, _stopwords_file)}

def is_stopword(a, id='smart'): 
    return a.startswith('\'') or a in _stopwords[id]
    



