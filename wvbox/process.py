#!/usr/bin/env python
'''
process.py
author: Luke de Oliveira (lukedeo@ldo.io)
'''

import logging

LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log(msg):
    logger.info(LOGGER_PREFIX % msg)



try:
    log('Importing spaCy...')
    from spacy.en import English
    log('Initializing spaCy NLP engine...')
    _nlp = English()
    log('spaCy successfully loaded.')
    def _backend_tok(txt):
        '''
        Takes a text and returns a list of tokens
        '''
        return [tx.lower() for tx in (t.text for t in _nlp(u'' + txt.decode('ascii',errors='ignore'))) if tx != '\n']

except ImportError:
    log('spaCy not found.')
    log('falling back to regex based tokenization.')
    
    import re
    
    def _backend_tok(sent):
        return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

parse_tokens = _backend_tok




