from __future__ import division, print_function
# coding=utf-8

from keras import backend as K
K.clear_session()

# import examples.example_helper
""" Module import helper.
Modifies PATH in order to allow us to import the deepmoji directory.
"""
import sys
import os
from os.path import abspath, dirname
sys.path.insert(0, dirname(dirname(abspath(__file__))))

import json
import csv
import numpy as np
from deepmoji.sentence_tokenizer import SentenceTokenizer
from deepmoji.model_def import deepmoji_emojis
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH

maxlen = 30
batch_size = 32
model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
model.summary()

with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

st = SentenceTokenizer(vocabulary, maxlen)

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

def model_predict(TEST_SENTENCES):
    print(TEST_SENTENCES)
    # print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
    tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
    print(tokenized)
    # print('Loading model from {}.'.format(PRETRAINED_PATH))
    prob = model.predict(tokenized)
    # prob = model.predict(TEST_SENTENCES)
    return prob

import pdb

def get_emoji(TEST_SENTENCES):
    # Find top emojis for each sentence. Emoji ids (0-63)
    # correspond to the mapping in emoji_overview.png
    # at the root of the DeepMoji repo.
    # print('Writing results to {}'.format(OUTPUT_PATH))
    scores = []
    t_score = []

    prob = model_predict(TEST_SENTENCES)

    print(type(prob),prob)

    # t = TEST_SENTENCES[0]
    # i = 0

    # for i, t in enumerate(TEST_SENTENCES):
    t_score.append(TEST_SENTENCES[0])
    # t_prob = prob[i]
    t_prob = prob[0]
    ind_top = top_elements(t_prob, 3)
    t_score.append(sum(t_prob[ind_top]))
    t_score.extend(ind_top)
    t_score.extend([t_prob[ind] for ind in ind_top])
    scores.append(t_score)
    print(t_score)
    return t_score

sentence = [['Hi nice to meet you'], ['something different sentence'],['for test']]

for i in range(3):
    get_emoji(sentence[i])