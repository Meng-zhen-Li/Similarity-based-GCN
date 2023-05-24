from __future__ import division, print_function
import scipy.sparse as sp
import numpy as np
import tensorflow.compat.v1 as tf

import time
import os

from utils import construct_feed_dict, sparse_to_tuple
from model import GCNModel
from optimizer import Optimizer

from sklearn.metrics import r2_score


flags = tf.app.flags
FLAGS = flags.FLAGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def r2(pair):
    label = pair[0]
    pred = pair[1]
    return r2_score(np.array(label.todense()), pred)

def train(adj, similarities, features):
    tf.reset_default_graph()

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'similarities': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    n = len(similarities)
    similarities = [x / np.max(x) for x in similarities]
    similarities = sp.vstack(similarities)
    similarities = sparse_to_tuple(similarities)


    model = GCNModel(placeholders, num_features, features_nonzero, n)
    with tf.name_scope('optimizer'):
        opt = Optimizer(model.reconstructions, tf.sparse_tensor_to_dense(placeholders['similarities'], validate_indices=False), n)

    # Initialize session
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    best_acc = -np.inf

    # Train model
    for epoch in range(FLAGS.epochs):
        t = time.time()
        feed_dict = construct_feed_dict(adj, similarities, features, placeholders)
        outs = sess.run([opt.costs, opt.opt_op, opt.r2], feed_dict=feed_dict)
        # save model
        if np.mean(outs[2]) >= best_acc:
            saver.save(sess, 'models/' + FLAGS.dataset)
            best_acc = np.mean(outs[2])

        # print loss and accuracy
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.5e}".format(np.mean(outs[0])), "r2=", "{:.5f}".format(np.mean(outs[2])), "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")

    # restore best model and reconstruct embeddings
    saver.restore(sess, 'models/' + FLAGS.dataset)
    emb = [None for i in range(len(similarities))]
    feed_dict = construct_feed_dict(adj, similarities, features, placeholders)
    emb = sess.run(model.embeddings, feed_dict=feed_dict)
    return np.split(emb, n)
