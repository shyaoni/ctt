import os

_debug_flags = {}
_last_view = [None]

def is_test(flag=None):
    """use _DEBUG=flag1,flag2... to speed up some operations in debuging.
    """
    if not '_DEBUG' in os.environ:
        return False
    flags = os.environ['_DEBUG'].split(',')
    return '1' in flags or flag in flags

def debug_inflags(flag, value=None):
    _last_view[0] = flag
    if value is None:
        return _debug_flags.get(flag, False)
    _debug_flags[flag] = value

def current_flag():
    return _last_view[0]

def prolog(iterator, name=''):
    """an alternative of tqdm.tqdm.
    """
    len_str = str(len(iterator)) if hasattr(iterator, '__len__') else '?'
    for i, item in enumerate(iterator):
        print('{}: {}/{}'.format(name, i, len_str), end='\r')
        yield item 
    print('', end='\n')
    
