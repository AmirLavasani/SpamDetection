#!/usr/bin/python
# -*- coding: utf-8 -*-
import codecs
from polyglot.mapping import Embedding
from random import shuffle
import numpy as np
from sklearn import svm
from tqdm import *
import click

def process_db(spam_path, not_spam_path):
    db_train = []
    db_target = []
    print 'Processing Spam Data\n'
    with codecs.open(spam_path, 'r', 'utf8') as spam:
        label = 0
        for row in tqdm(spam):
            row = row.strip('\n')
            split = row.split(' ')
            if len(split) == 1:
                if split[0] == '':
                    continue
            db_train.append(split)
            db_target.append(label)

    print 'Processing Not Spam Data\n'
    with codecs.open(not_spam_path, 'r', 'utf8') as spam:
        label = 1
        for row in tqdm(spam):
            row = row.strip('\n')
            split = row.split(' ')
            if len(split) == 1:
                if split[0] == '':
                    continue
            db_train.append(split)
            db_target.append(label)
    return db_train, db_target


def trunc_db(train, label, trunc_size=25):
    max_size = trunc_size
    train_trunc = []
    label_trunc = []
    for t,l in zip(train, label):
        if len(t) <= max_size:
            train_trunc.append(t)
            label_trunc.append(l)

    return train_trunc, label_trunc


def convert_labels_to_spars(label_seq):
    spars_label = []

    for l in label_seq:
        if l == '0':
            spars_label.append([1, 0])
        elif l == '1':
            spars_label.append([0, 1])
    if len(spars_label) == len(label_seq):
        return spars_label
    else:
        return None


def pad_train_data(train, pad_size=25):
    for t in train:
        while len(t) < pad_size:
            t.append(u'*')
    return train


def train_word_embeddings(train, polyglot_data):    
    embeddings = Embedding.load(polyglot_data)
    zpadd = [0] * 64
    train_embds = []
    for t in train:
        t_e = []
        for w in t:
            if w == u'*':
                t_e.append(zpadd)
            else:
                e = embeddings.get(w)
                if e is not None:
                    t_e.append(e)
                else:
                    t_e.append(zpadd)
        train_embds.append(t_e)
    return train_embds


def randomize_train(train, label):
    zipped = zip(train, label)
    shuffle(zipped)
    train, label = zip(*zipped)
    return list(train), list(label)


def prepare_data(spam_path, not_spam_path, polyglot_data):
    train, label = process_db(spam_path, not_spam_path)
    train, label = trunc_db(train, label)
    # no need for spars labels
    spars_label = label #convert_labels_to_spars(label)
    padded_train = pad_train_data(train)
    train_embds = train_word_embeddings(padded_train, polyglot_data)

    return randomize_train(train_embds, spars_label)


def split_data(data, label):
    test_num = 300
    train_input = data[:(-test_num)]
    train_output = label[:(-test_num)]  # everything till the last 50 numbers

    test_input = data[(-test_num):]
    test_output = label[(-test_num):]  # till 10,000

    return train_input, train_output, test_input, test_output


def train_svm(x, y):

    print 'Training linear SVM\n'
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(x, y)
    return clf


def accr_svm(model, x, y):
    print 'Calculating Accuracy\n'
    total_test = y.shape[0]
    corrects = 0.
    #print len(x), len(y)
    #print len(max(x, key=len))
    #print len(min(x, key=len))

    for t in zip(x, y):        
        l = model.predict([t[0]])        
        if l == t[1]:
            corrects += 1
    return corrects / total_test

@click.command()
@click.option('--spam', help='spam data file. one sentence per line', default='data/spam_norm.txt')
@click.option('--notspam', help='spam data file. one sentence per line', default='data/not_spam_norm.txt')
def main(spam, notspam):
    polyglot_data = "/home/amir/polyglot_data/embeddings2/fa/embeddings_pkl.tar.bz2"
    data, label = prepare_data(spam, notspam, polyglot_data)
    print len(data)
    train_input, train_output, test_input, test_output = split_data(data, label)
    train_input = np.array(train_input)
    train_input_reshaped = np.reshape(train_input, (train_input.shape[0], train_input.shape[1] * train_input.shape[2]))
    train_output = np.array(train_output)
    test_input = np.array(test_input)
    test_input_reshaped = np.reshape(test_input, (test_input.shape[0], test_input.shape[1] * test_input.shape[2]))
    test_output = np.array(test_output)

    model = train_svm(train_input_reshaped, train_output)
    print "SVM accuracy on random picked test set:\t{}".format(accr_svm(model, test_input_reshaped, test_output))

if __name__ == "__main__":
    main()