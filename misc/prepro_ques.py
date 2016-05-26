"""
Preprocesses a question
Adapted from https://github.com/VT-vision-lab/VQA_LSTM_CNN/blob/master/prepro.py
"""
import copy
from random import shuffle, seed
import sys
import os.path
import argparse
import glob
import numpy as np
from scipy.misc import imread, imresize
import scipy.io
import pdb
import string
import h5py
from nltk.tokenize import word_tokenize
import json

import re


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i!='' and i!=' ' and i!='\n'];

def prepro_question(s, method='nltk'):
    if method == 'nltk':
        txt = word_tokenize(str(s).lower())
    else:
        txt = tokenize(s)
    return txt

def apply_vocab_question(tokens, wtoi):
    # apply the vocab on test.
    question = [w if wtoi.get(w,len(wtoi)+1) != (len(wtoi)+1) else 'UNK' for w in tokens]
    return question

def encode_question(ques, wtoi):
    max_length = 26

    label_arrays = np.zeros((max_length), dtype='uint32')
    label_length = min(max_length, len(ques)) # record the length of this sequence
    for k, w in enumerate(ques):
        if k < max_length :
            print(w)
            label_arrays[k] = wtoi[w]

    return label_arrays, label_length

def feat_ques(question):
    # tokenization and preprocessing training question
    tokens = prepro_question(question)

    # create the vocab for question
    # Load Vocabulary File
    with open('VQA_LSTM_CNN/data_prepro.json', 'r') as f:
        itow = json.load(f)['ix_to_word']
    wtoi = {w:i for i,w in itow.items()} # inverse table

    fin_ques = apply_vocab_question(tokens, wtoi)
    ques, ques_length = encode_question(fin_ques, wtoi)
    q = {}
    q['ques'] = ques.tolist()
    q['ques_length'] = ques_length
    with open('ques_feat.json','w') as q_file:
        json.dump(q,q_file)
    print ques.tolist()

    return ques.tolist(), ques_length

def main(params):
    question = params['question']
    ques, ques_length = feat_ques(question)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--question', dest='question', default='what is the man doing', help='question string')
    args = parser.parse_args()
    params = vars(args)
    main(params)
