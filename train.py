from __future__ import division, print_function
import scipy.sparse as sp
import numpy as np
import tensorflow.compat.v1 as tf

import time
import os

from utils import construct_feed_dict, preprocess_graph, sparse_to_tuple
from model import GCNModel
from optimizer import Optimizer

from sklearn.metrics import r2_score


flags = tf.app.flags
FLAGS = flags.FLAGS


def train(adj, similarities, features):
    tf.reset_default_graph()

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'similarity': tf.sparse_placeholder(tf.float32),
        'sim_idx': tf.placeholder_with_default(0, shape=()),
        'dropout': tf.placeholder_with_default(0., shape=()), 
        'gradient0': tf.placeholder(tf.float32), 
        'gradient1': tf.placeholder(tf.float32), 
        'gradient2': tf.placeholder(tf.float32)
    }

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    n = len(similarities)
    similarities_sparse = [x / np.max(x) for x in similarities]
    similarities = [sparse_to_tuple(x / np.max(x)) for x in similarities]


    model = GCNModel(placeholders, num_features, features_nonzero, n)
    with tf.name_scope('optimizer'):
        opt = Optimizer(model.reconstructions, tf.sparse_tensor_to_dense(placeholders['similarity'], validate_indices=False), placeholders)

    # Initialize session
    devices = tf.config.list_physical_devices('GPU')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.compat.v1.train.Saver()
    best_acc = 0

    # Train model(distributed)
    if FLAGS.distributed == 1:
        costs = [0 for i in range(len(similarities))]
        accs = [0 for i in range(len(similarities))]
        for epoch in range(FLAGS.epochs):
            t = time.time()
            for sim_idx in range(len(similarities)):
                # Construct feed dictionary
                feed_dict = construct_feed_dict(adj, similarities[sim_idx], features, placeholders, sim_idx=sim_idx)
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                # Run single weight update
                outs = sess.run([opt.cost, model.reconstructions, opt.grads_vars], feed_dict=feed_dict)
                acc = r2_score(np.array(similarities_sparse[sim_idx].todense()), outs[1])
                if sim_idx == 0:
                    grads = [x[0] for x in outs[2]]
                else:
                    grads = [grads[i] + outs[2][i][0] for i in range(3)]
                # Compute average loss
                costs[sim_idx] = outs[0]
                accs[sim_idx] = acc

            # update and apply gradients
            grads[:2] = [grads[i] / n for i in range(2)]
            for i in range(3):
                feed_dict.update({placeholders['gradient' + str(i)]: grads[i]})
            
            # save model
            if np.mean(accs) >= best_acc:
                saver.save(sess, 'models/' + FLAGS.dataset)
                best_acc = np.mean(accs)
            sess.run(opt.opt_op, feed_dict=feed_dict)

            # print loss and accuracy
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.5e}".format(np.mean(costs)), "acc=", "{:.5f}".format(np.mean(accs)), "time=", "{:.5f}".format(time.time() - t))

        print("Optimization Finished!")

        # restore best model and reconstruct embeddings
        saver.restore(sess, 'models/' + FLAGS.dataset)
        emb = [None for i in range(len(similarities))]
        for sim_idx in range(len(similarities)):
            feed_dict = construct_feed_dict(adj, similarities[sim_idx], features, placeholders, sim_idx=sim_idx)
            emb[sim_idx] = sess.run(model.embeddings, feed_dict=feed_dict)
    
    # Tringing without distributed
    else:
        for epoch in range(FLAGS.epochs):
            t = time.time()
            # Construct feed dictionary
            feed_dict = construct_feed_dict(adj, sparse_to_tuple(sp.vstack(similarities_sparse)), features, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Run single weight update
            outs = sess.run([opt.opt_op, opt.cost, model.reconstructions], feed_dict=feed_dict)

            # Compute average loss
            avg_cost = outs[1]
            # acc = r2_score(np.array(sp.vstack(similarities_sparse).todense()), outs[2])
            reconstructions = np.split(outs[2], n)
            acc = np.mean([r2_score(np.array(similarities_sparse[i].todense()), reconstructions[i]) for i in range(n)])
            
            # save model
            if acc >= best_acc:
                saver.save(sess, 'models/' + FLAGS.dataset)
                best_acc = acc

            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.5e}".format(avg_cost), "acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
        saver.restore(sess, 'models/' + FLAGS.dataset)
        emb = sess.run([model.embeddings, model.reconstructions], feed_dict=feed_dict)
    return emb
