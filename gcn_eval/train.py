import numpy as np
import scipy.sparse as sp
import tensorflow.compat.v1 as tf

import time

from evaluation import get_roc_score, evaluate, get_masked_auc
from gcn_eval.optimizer import OptimizerAE, OptimizerGCN
from gcn_eval.model import GAE, GCN
from gcn_eval.utils import construct_feed_dict, construct_feed_dict_ae, sparse_to_tuple

flags = tf.app.flags
FLAGS = flags.FLAGS


def train_gae(adj_norm, adj_train, features, val_edges, val_edges_false):
    """ 
        adj_norm: normalized similarity matrix
        adj_train: train graph
    """

    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    # Create model and optimizer
    model = GAE(placeholders, num_features, features_nonzero)
    pos_weight = float(
        adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) / adj_train.sum()
    norm = adj_train.shape[0] * adj_train.shape[0] / \
        float((adj_train.shape[0] * adj_train.shape[0] - adj_train.sum()) * 2)
    with tf.name_scope('optimizer'):
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(
                              placeholders['adj_orig'], validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict_ae(
            adj_norm, adj_label, features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                        feed_dict=feed_dict)

        # Compute average loss
        avg_cost = outs[1]
        avg_accuracy = outs[2]

        # Compute current roc and ap
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)
        roc_curr, ap_curr = get_roc_score(
            val_edges, val_edges_false, np.dot(emb, emb.T))

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
              "train_acc=", "{:.5f}".format(
                  avg_accuracy), "val_roc=", "{:.5f}".format(roc_curr),
              "val_ap=", "{:.5f}".format(ap_curr),
              "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")
    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.z_mean, feed_dict=feed_dict)
    return np.dot(emb, emb.T)


def train_gcn(adj_norm, y_train, features, train_mask, y_val, val_mask):
    """ 
        adj_norm: normalized similarity matrix
        y_train: train label
    """

    # Define placeholders
    placeholders = {
        'support': tf.sparse_placeholder(tf.float32),
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        # helper variable for sparse dropout
        'features_nonzero': tf.placeholder(tf.int32)
    }

    # Create model and optimizer
    model = GCN(placeholders, input_dim=features[2][1], logging=True)
    pos_weight = float(
        y_train.shape[0] * y_train.shape[1] - y_train.sum()) / y_train.sum()
    norm = y_train.shape[0] * y_train.shape[0] / \
        float((y_train.shape[0] * y_train.shape[0] - y_train.sum()) * 2)
    with tf.name_scope('optimizer'):
        opt = OptimizerGCN(layers=model.layers, outputs=model.outputs, placeholders=model.placeholders, pos_weight=pos_weight,
                           norm=norm)

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    costs_val = []

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(
            features, adj_norm, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([opt.opt_op, opt.loss, model.outputs],
                        feed_dict=feed_dict)
        auc, ap = get_masked_auc(outs[2], y_train, train_mask)

        # Validation
        cost_val, auc_val, ap_val = evaluate(
            features, adj_norm, y_val, val_mask, placeholders, sess, opt, model)
        costs_val.append(cost_val)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cost_val),
              "train_roc=", "{:.5f}".format(auc),
              "train_ap=", "{:.5f}".format(ap),
              "val_roc=", "{:.5f}".format(auc_val),
              "val_ap=", "{:.5f}".format(ap_val),
              "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")
    return placeholders, sess, opt, model
