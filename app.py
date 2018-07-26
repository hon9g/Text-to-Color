from __future__ import division, print_function
# coding=utf-8

from keras import backend as K
# K.clear_session()

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

import glob
import re

# Keras
from keras import backend as K
# from keras.models import load_model
# from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

maxlen = 30
batch_size = 32

emo = ['😂', '😒', '😩', '😭', '😍',
       '😔', '👌', '😊', '❤', '😏',
       '😁', '🎶', '😳', '💯', '😴',
       '😌', '☺', '🙌', '💕', '😑',
       '😅', '🙏', '😕', '😘', '♥',
       '😐', '💁', '😞', '🙈', '😫',
       '✌', '😎', '😡', '👍', '😢',
       '😪', '😋', '😤', '✋', '😷',
       '👏', '👀', '🔫', '😣', '😈',
       '😓', '💔', '💓', '🎧', '🙊',
       '😉', '💀', '😖', '😄', '😜',
       '😠', '🙅', '💪', '👊', '💜',
       '💖', '💙', '😬', '✨']

emoToColor = ['191, 255, 0', '0, 96, 128', '0, 153, 204', '0, 0, 153', '255, 132, 102',
              '77, 255, 210', '255, 177, 0', '255, 132, 102', '230, 0, 0', '255, 147, 255',
              '255, 132, 102', '255, 51, 204', '191, 255, 0', '255, 255, 0', '147, 166, 89',
              '204, 51, 255', '255, 64, 0', '255, 177, 0', '255, 64, 0', '0, 96, 128',
              '191, 255, 0', '230, 0, 0', '0, 96, 128', '255, 64, 0', '230, 0, 0',
              '0, 96, 128', '204, 51, 255', '77, 255, 210', '255, 30, 98', '0, 153, 204',
              '255, 255, 0', '204, 51, 255', '99, 0, 77', '255, 177, 0', '0, 0, 153',
              '77, 255, 210', '255, 132, 102', '99, 0, 77', '255, 255, 0', '147, 166, 89',
              '255, 177, 0', '255, 30, 98', '147, 166, 89', '77, 255, 210', '204, 51, 255',
              '77, 255, 210', '0, 0, 153', '230, 0, 0', '255, 51, 204', '255, 30, 98',
              '255, 147, 255', '191, 255, 0', '0, 153, 204', '255, 132, 102', '255, 147, 255',
              '99, 0, 77', '255, 255, 0', '255, 255, 0', '255, 255, 0', '255, 64, 0',
              '255, 64, 0', '255, 64, 0', '191, 255, 0', '255, 64, 0'
              ]

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

# print('Loading model from {}.'.format(PRETRAINED_PATH))
model = deepmoji_emojis(maxlen, PRETRAINED_PATH)
model.summary()
model._make_predict_function()

with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)
st = SentenceTokenizer(vocabulary, maxlen)

def model_predict(TEST_SENTENCES):
    print(TEST_SENTENCES)

    # print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))
    tokenized, _, _ = st.tokenize_sentences(TEST_SENTENCES)
    print (tokenized)
    # pdb.set_trace()
    # prob = model.predict(tokenized)
    # pdb.set_trace()
    prob = model.predict_function([tokenized])[0]
    return prob

import pdb

def get_emoji(TEST_SENTENCES):
    # K.clear_session()
    # Find top emojis for each sentence. Emoji ids (0-63)
    # correspond to the mapping in emoji_overview.png
    # at the root of the DeepMoji repo.
    # print('Writing results to {}'.format(OUTPUT_PATH))
    scores = []
    t_score = []
    print (TEST_SENTENCES)
    prob = model_predict(TEST_SENTENCES)
    # K.clear_session()
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

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print ('called')
        # Get the input from post request
        temp = []
        temp.append(request.form['text'])
        length = len(temp[0].split())
        if length > 1:
            result = get_emoji(temp)
            # result = ['i am so hungry i could die', 0.32384027540683746, 29, 52, 2, 0.12622266, 0.11112564, 0.08649198]
            # print(result)
            # value = str(result[0])
            # confidency = str(result[1])
            norm = float(result[5]) + float(result[6]) + float(result[7])
            inx1 = int(result[2])
            inx2 = int(result[3])
            inx3 = int(result[4])
            value = emo[inx1]+emo[inx2]+emo[inx3]+'\n'+str(result[0])
            color1 = "rgba(" + emoToColor[inx1] + ', ' + str(float(result[5]) / norm) + ")"
            color2 = "rgba(" + emoToColor[inx2] + ', ' + str(float(result[6]) / norm) + "), rgba(" + emoToColor[inx2] + ",  0.0)"
            color3 = "rgba(" + emoToColor[inx3] + ', ' + str(float(result[7]) / norm) + "), rgba(" + emoToColor[inx3] + ",  0.0)"
            print (color1, color2, color3)
            return render_template('predict.html', result=value, color1=color1, color2=color2, color3=color3)
        else:
            return render_template('index.html')
    else:
        return render_template('index.html')


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(port=5002, debug=True)
    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()