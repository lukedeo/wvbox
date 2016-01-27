'''
wvbox.py -- a wrapper and handler for word vectors from the 
GloVe package released by the Stanford NLP Group. It is also a 
generic handler for convering text --> integers

author: Luke de Oliveira (lukedeo@ldo.io)
'''

from functools import wraps
import logging

import numpy as np

from .process import parse_tokens

LOGGER_PREFIX = ' %s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log(msg):
    logger.info(LOGGER_PREFIX % msg)

class WVBoxException(Exception):
    """
    Errors for WVBox
    """
    pass

class WVBox(object):
    '''

    A nice container for handing vocab issues and word vector lookup!

    >>> wv = WVBox('./glove.6B.300d.txt')
    >>> wv.build()
    >>> docs = ['The dog likes his owner', 'This is an interesting finding']
    >>> wv.get_indices(docs)
    [[1, 828, 1042, 57, 1379], [22, 9, 41, 917, 1482]]
    >>> for s in wv.get_words(wv.get_indices(docs)):
    ...     print ' '.join(s)
    the dog likes his owner
    this is an interesting finding
    '''
    def __init__(self, vector_file=None, verbose=True, tokenizer=parse_tokens):
        ''' 
        Ctor for word vector container class.
        
        Args:
        -----
            vector_file: a path to a file containing the word vectors 
                we are interested in. Each line should have a word followed
                by a space followed by the space-delimited elements of the
                calculated word vectors.

            tokenizer: a function that takes as input a string and returns
                a list of the tokens.
        '''
        
        super(WVBox, self).__init__()

        self._vector_file = vector_file
        self._verbose = verbose
        self._built = False

        self.W = None
        self.vocab = False

        self._w2i = {}
        self._i2w = {}


        self._tokenizer = tokenizer


    def builtmethod(f):
        def wrapper(*args, **kwargs):
            if not args[0]._built:
                raise WVBoxException(
                    'Method {} cannot be called without calling .build() first.'
                    .format(f.__name__)
                )
            return f(*args, **kwargs)
        return wrapper

    @property
    @builtmethod
    def wv_size(self):
        return self.W.shape[1]

    @property
    @builtmethod
    def vocab_size(self):
        return self.W.shape[0]


    def load_vectors(self, vector_file):
        ''' 
        Effectively resets the container to a new vector file
        
        Args:
        -----
            vector_file: path to the new vector file
        '''
        
        self._vector_file = vector_file
        self._built = False

        self.W = None
        self.vocab = False

        self._w2i = {}
        self._i2w = {}

        return self

    # def build(self, zero_token=False, normalize_variance=False, normalize_norm=False):
    def build(self, normalize_variance=False, normalize_norm=False):
        ''' 
        Builds the internal structure for the word vectors
        
        Args:
        -----
            normalize_variance: boolean, whether or not to normalize vectors
                to unit variance

            normalize_norm: boolean, whether or not to normalize vectors to
                unit norm
        
        Returns:
        --------
            self
        
        Raises:
        -------
            WVBoxException when things don't go according to plan
        '''
        
        if (self._vector_file is None):
            raise WVBoxException('Need to specify input vector and vocab files before building')

        with open(self._vector_file, 'r') as f:
            if self._verbose:
                log('Loading vectors from {}'.format(self._vector_file))
            vectors = {}
            words = []
            ctr = 0
            for line in f:
                if ctr % 10000 == 0:
                    log('Loading word {}'.format(ctr))
                line = line.decode('utf-8')
                vals = line.rstrip().split(' ')
                if vals[0] != u'<unk>':
                    words.append(vals[0])
                vectors[vals[0]] = [float(x) for x in vals[1:]]
                ctr += 1

        log('Building storage structures...')
        
        log('Mapping words to indices...')
        vocab_size = len(words)
        # if not zero_token:
        #     trf = lambda x : x
        # else:
        #     trf = lambda x : x + 1
        #     vocab_size += 1

        trf = lambda x : x + 1
        vocab_size += 1

        self._w2i = {unicode(w): trf(idx) for idx, w in enumerate(words)}
        self._w2i.update({'<unk>' : -1})

        # if zero_token:
        #     self._w2i.update({'<blank>' : 0})

        self._w2i.update({'<blank>' : 0})
        
        log('Mapping indices to words...')
        self._i2w = {trf(idx): unicode(w) for idx, w in enumerate(words + ['<unk>'])}
        self._i2w.update({-1 : '<unk>'})

        # if zero_token:
        #     self._i2w.update({0 : '<blank>'})
        self._i2w.update({0 : '<blank>'})


        vector_dim = len(vectors[self._i2w[1]])
        self.W = np.zeros((vocab_size + 1, vector_dim))
        ctr = 0

        vs, ix = [], []
        for word, v in vectors.iteritems():
            if ctr % 10000 == 0:
                log('Loading word vector {}'.format(ctr))
            if word == '<unk>':
                continue
            vs.append(v)
            ix.append(self._w2i[word])
            ctr += 1
        self.W[np.array(ix), :] = np.array(vs)
        try:
            self.W[-1, :] = vectors['<unk>']
        except KeyError:
            self.W[-1, :] = self.W[:-1, :].mean(axis=0)

        if normalize_variance:
            log('Normalizing vectors by variance...')
            # normalize each word vector to unit variance

            self.W[0, :] += 1
            W_norm = np.zeros(self.W.shape)
            d = (np.sum(self.W ** 2, 1) ** (0.5))
            W_norm = (self.W.T / d).T
            self.W = W_norm
            self.W[0, :] = 0
        if normalize_norm:
            log('Normalizing vectors by norm...')
            # normalize each word vector to unit variance
            # ptr = 0
            # if zero_token:
            #     ptr = 1
            ptr = 1

            self.W[ptr:] /= np.linalg.norm(self.W[ptr:], axis=1)[:, np.newaxis]
        else:
            log('No vector normalization performed...')
        self.vocab = words
        self._built = True
        return self

    @builtmethod
    def _get_w2i(self, w):
        try:
            return self._w2i[unicode(w)]
        except KeyError:
            return self.W.shape[0] - 1
    @builtmethod
    def _get_i2w(self, i):
        try:
            return self._i2w[i]
        except KeyError:
            return '<unk>'

    @builtmethod
    def get_indices(self, obj):
        ''' 
        Map strings and documents to integers
        
        Args:
        -----
            obj: a string or iterable of strings. Each string can either
                be a token or a document to be tokenized.
        
        Returns:
        --------
            An iterable of the indices of the given words in the WV 
            word bank
        
        Raises:
        -------
            WVBoxException if .build() hasn't been called
        '''
        
        if isinstance(obj, str) or isinstance(obj, unicode):
            tokenized = self._tokenizer(obj)
            if len(tokenized) == 1:
                return self._get_w2i(tokenized[0])
            return [self._get_w2i(o) for o in tokenized]
        elif hasattr(obj, '__iter__'):
            return [self.get_indices(o) for o in obj]

    @builtmethod
    def get_words(self, obj):
        ''' 
        Takes integers and maps them to their associated strings
        
        Args:
        -----
            obj: an int or an iterable of ints
        
        Returns:
        --------
            an iterable with the given words as unicode strings
        
        Raises:
        -------
            WVBoxException if .build() hasn't been called
        '''
        
        if isinstance(obj, int):
            return self._get_i2w(obj)
        elif hasattr(obj, '__iter__'):
            return [self.get_words(o) for o in obj]

    @builtmethod
    def __getitem__(self, key):
        if isinstance(key, unicode):
            tokenized = self._tokenizer(key)
            if len(tokenized) == 1:
                return self.W[self._get_w2i(key), :]
            return self.W[np.array([self._get_w2i(k) for k in key]), :]
        elif hasattr(key, '__iter__'):
            return np.array([self.__getitem__(k).tolist() for k in key])
        elif isinstance(key, str):
            raise WVBoxException('Keys must be unicode strings')




