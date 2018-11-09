#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import codecs
import polyglot
from polyglot.mapping import Embedding
import tensorflow as tf
from random import shuffle
import click

def process_db(spam_path, not_spam_path):
    db_train = []
    db_target = []
    with codecs.open(spam_path, 'r', 'utf8') as spam:
        label = '0'
        for row in spam:
            row = row.strip('\n')
            split = row.split(' ')
            if len(split) == 1:
                if split[0] == '':
                    continue
            db_train.append(split)
            db_target.append(label)

    with codecs.open(not_spam_path, 'r', 'utf8') as spam:
        label = '1'
        for row in spam:
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


def train_word_embeddings(train):
    embeddings = Embedding.load("/home/amir/polyglot_data/embeddings2/fa/embeddings_pkl.tar.bz2")
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


def prepare_data(spam_path, not_spam_path):
    train, label = process_db(spam_path, not_spam_path)
    train, label = trunc_db(train, label)
    spars_label = convert_labels_to_spars(label)
    padded_train = pad_train_data(train)
    train_embds = train_word_embeddings(padded_train)

    return randomize_train(train_embds, spars_label)


def train(spam, not_spam):
    # CREATING GRAPH
    data, label = prepare_data(spam, not_spam)

    test_num = 300
    train_input = data[:(-test_num)]
    train_output = label[:(-test_num)]  # everything till the last 50 numbers

    test_input = data[(-test_num):]
    test_output = label[(-test_num):]  # till 10,000

    features_dim = 64  # word embeddings features
    time_step = 25
    num_classes = 2
    num_hidden = 128
    batch_size = 16
    no_of_batches = int(len(train_input) / batch_size)
    epoch = 10
    num_layers = 5

    data = tf.placeholder(tf.float32, [None, time_step, features_dim])
    target = tf.placeholder(tf.float32, [None, num_classes])

    # cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    # Stacking rnn cells
    # cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    # stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    # changed in tensorflow >= 1.2
    stack = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True)
                                         for _ in range(num_layers)], state_is_tuple=True)
    val, state = tf.nn.dynamic_rnn(stack, data, dtype=tf.float32)
    # transpose to change batch with seqence
    val = tf.transpose(val, [1, 0, 2])
    # only the last output is important
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))

    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    best_error = 1.
    print 'Total number of batches:\t{}'.format(no_of_batches)
    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(epoch):
            ptr = 0
            train_input, train_output = randomize_train(train_input, train_output)
            for j in range(no_of_batches):
                inp, out = train_input[ptr:ptr + batch_size], train_output[ptr:ptr + batch_size]
                ptr += batch_size
                sess.run(optimizer, {data: inp, target: out})
                if j % 16 == 0:
                    incorrect = sess.run(error, {data: inp, target: out})
                    print('Epoch {:2d} batch {} train error {:3.1f}%'.format(i + 1, j, 100 * incorrect))

            print "Epoch - ", str(i)
            if i % 1 == 0:
                incorrect = sess.run(error, {data: test_input, target: test_output})
                print('Epoch {:2d} valid error {:3.1f}%'.format(i + 1, 100 * incorrect))
                if incorrect < best_error:
                    best_error = incorrect
                    saver.save(sess, 'model/atf_model_best_error.ckpt')
    print best_error


def process_text(text, max_size=25):
    split = text.split(' ')
    if len(split) > max_size:
        print "TOO LONG TEXT"
        return None
    else:
        txt_padd = pad_train_data([split])
        txt_final = train_word_embeddings(txt_padd)
    return txt_final


def predict(model_path):
    # Building Graph
    features_dim = 64  # word embeddings features
    time_step = 25
    num_classes = 2
    num_hidden = 128
    batch_size = 16
    # no_of_batches = int(len(train_input) / batch_size)
    epoch = 10
    num_layers = 5

    data = tf.placeholder(tf.float32, [None, time_step, features_dim])
    target = tf.placeholder(tf.float32, [None, num_classes])

    # cell = tf.contrib.rnn.LSTMCell(num_hidden,state_is_tuple=True)

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    # Stacking rnn cells
    # cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
    # stack = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)

    # changed in tensorflow >= 1.2
    stack = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(num_units=num_hidden, state_is_tuple=True)
                                         for _ in range(num_layers)], state_is_tuple=True)
    val, state = tf.nn.dynamic_rnn(stack, data, dtype=tf.float32)
    # transpose to change batch with seqence
    val = tf.transpose(val, [1, 0, 2])
    # only the last output is important
    last = tf.gather(val, int(val.get_shape()[0]) - 1)

    weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
    bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

    prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
    cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))

    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    sess = tf.Session()

    saver.restore(sess, model_path)
    #	sess.run(init_op)

    #	meta_path = os.path.splitext(os.path.split(model_path)[1])[0]+'.meta'
    #	print meta_path
    #	saver = tf.train.import_meta_graph(os.path.splitext(os.path.split(model_path)[1])[0]+'.meta')
    #	saver.restore(sess, model_path)
    #	sess.run(tf.global_variables_initializer())

    while True:
        txt = raw_input("Please enter your text: ")
        processed_txt = process_text(txt)
        if processed_txt is not None:
            feed_dict = {data: processed_txt}
            pred = sess.run(tf.argmax(sess.run(prediction, feed_dict), 1))
            if pred == [0]:
                print 'SPAM'
            if pred == [1]:
                print 'NOT SPAM'

            print pred
@click.command()
@click.option('--spam', help='spam data file. one sentence per line', default='data/spam_norm.txt')
@click.option('--not-spam', help='spam data file. one sentence per line', default='data/not_spam_norm.txt')
@click.option('--train-flag', help='set it to True to train the model first.', default=False)
def main(spam, not_spam, train_flag):
    if train_flag:
        train(spam, not_spam)
    
    predict('model/atf_model_best_error.ckpt')

if __name__ == "__main__":
    main()