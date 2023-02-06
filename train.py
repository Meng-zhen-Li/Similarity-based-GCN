from __future__ import division, print_function
import scipy.sparse as sp
import numpy as np
import tensorflow.compat.v1 as tf

import time
import os

from utils import sparse_to_tuple, construct_feed_dict
from model import GCNModel
from optimizer import Optimizer

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""


flags = tf.app.flags
FLAGS = flags.FLAGS


def train(adj, similarity, features):

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'similarity': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    similarity = sparse_to_tuple(similarity + sp.eye(similarity.shape[0]))

    model = GCNModel(placeholders, num_features, features_nonzero)
    with tf.name_scope('optimizer'):
        opt = Optimizer(preds=model.reconstructions, labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['similarity'],validate_indices=False), [-1]))
    
    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj, similarity, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]

        print("Test set results:", "cost=", "{:.5f}".format(avg_cost), "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")
    
    emb = sess.run(model.embeddings, feed_dict=feed_dict)
    return emb
