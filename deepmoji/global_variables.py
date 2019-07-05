""" Global variables.
"""
import tempfile
from os.path import abspath, dirname

# The ordering of these special tokens matter
# blank tokens can be used for new purposes
# Tokenizer should be updated if special token prefix is changed
SPECIAL_PREFIX = 'CUSTOM_'
SPECIAL_TOKENS = ['CUSTOM_MASK',
                  'CUSTOM_UNKNOWN',
                  'CUSTOM_AT',
                  'CUSTOM_URL',
                  'CUSTOM_NUMBER',
                  'CUSTOM_BREAK']
SPECIAL_TOKENS.extend(['{}BLANK_{}'.format(SPECIAL_PREFIX, i) for i in range(6, 10)])

ROOT_PATH = dirname(dirname(abspath(__file__)))
VOCAB_PATH = '{}/models/vocabulary.json'.format(ROOT_PATH)
PRETRAINED_PATH = '{}/models/deepmoji_weights.hdf5'.format(ROOT_PATH)

NB_TOKENS = 50000
NB_EMOJI_CLASSES = 64