from __future__ import division
from __future__ import print_function

from evaluation import *
from gcn_eval.train import *

from utils import preprocess_graph, preprocess_features
from similarity import similarity_matrix
from input_data import load_data
from train import train
from consensus import consensus

import scipy.sparse as sp
import numpy as np
import tensorflow.compat.v1 as tf

import os
from scipy.io import loadmat, savemat

tf.disable_eager_execution()

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 800, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('dataset', 'node2vec_PPI', 'Dataset string.')
flags.DEFINE_string('task', 'node_classification', 'Prediction task, link prediction or node classification.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('distributed', 1, 'Whether to use distributed training.')

dataset_str = FLAGS.dataset


if FLAGS.task == 'link_prediction':
    adj, features, adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = load_data(dataset_str)
else:
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset_str)
if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

features = preprocess_features(features)

# Link Prediction
if FLAGS.task == 'link_prediction':
    f = open("results_lp.txt","a")
    adj_norm = preprocess_graph(adj_train)
    similarities = similarity_matrix(adj_train)
    
    adj_pred = train_gae(preprocess_graph(adj_train), adj_train, features, val_edges, val_edges_false)
    roc_score, ap_score = get_roc_score(test_edges, test_edges_false, adj_pred)
    f.write("Using original adjacency matrix:\n")
    f.write('Test AUC score: ')
    f.write("{:.5f}".format(roc_score))
    f.write('\t')
    f.write('Test AUPR score: ')
    f.write("{:.5f}".format(ap_score))
    f.write('\n')

    adj_random = sp.random(adj.shape[0], adj.shape[0], density=adj.count_nonzero()/adj.shape[0]/adj.shape[0])
    adj_random = adj_random + adj_random.transpose()
    adj_random = adj_random / np.max(adj_random)
    adj_random = adj_random - sp.dia_matrix((adj_random.diagonal()[np.newaxis, :], [0]), shape=adj_random.shape)
    adj_random = preprocess_graph(adj_random)
    adj_pred = train_gae(adj_random, adj_train, features, val_edges, val_edges_false)
    roc_score, ap_score = get_roc_score(test_edges, test_edges_false, adj_pred)
    f.write('Random adjacency matrix:\n')
    f.write('Test AUC score: ')
    f.write("{:.5f}".format(roc_score))
    f.write('\t')
    f.write('Test AUPR score: ')
    f.write("{:.5f}".format(ap_score))
    f.write('\n')

    emb = train(adj_norm, similarities, features)
    emb = consensus(emb)
    # emb = consensus(np.split(emb, len(similarities)))
    adj_learned = sp.csr_matrix(np.dot(emb, emb.T))
    # thr = np.min(np.max(adj_learned, 1))
    thr = 0
    adj_learned.data[adj_learned.data < thr] = 0
    adj_learned.eliminate_zeros()
    adj_learned = preprocess_graph(adj_learned)
    adj_pred = train_gae(adj_learned, adj_train, features, val_edges, val_edges_false)
    roc_score, ap_score = get_roc_score(test_edges, test_edges_false, adj_pred)
    f.write('Learned adjacency matrix:\n')
    f.write('Test AUC score: ')
    f.write("{:.5f}".format(roc_score))
    f.write('\t')
    f.write('Test AUPR score: ')
    f.write("{:.5f}".format(ap_score))
    f.write('\n\n')


# Node Classification
if FLAGS.task == 'node_classification':
    f = open("results_nc.txt","a")
    adj_norm = preprocess_graph(adj)
    similarities = similarity_matrix(adj)
    
    placeholders, sess, opt, model = train_gcn(adj_norm, y_train, features, train_mask, y_val, val_mask)
    _, auc, ap = evaluate(features, preprocess_graph(adj), y_test, test_mask, placeholders, sess, opt, model)
    f.write('Using original adjacency matrix:\n')
    f.write("Test AUC score: ")
    f.write("{:.5f}".format(auc))
    f.write('\t')
    f.write('Test AUPR score: ')
    f.write("{:.5f}".format(ap))
    f.write('\n')

    adj_random = sp.random(adj.shape[0], adj.shape[0], density=adj.count_nonzero()/adj.shape[0]/adj.shape[0])
    adj_random = adj_random + adj_random.transpose()
    adj_random = adj_random / np.max(adj_random)
    adj_random = adj_random - sp.dia_matrix((adj_random.diagonal()[np.newaxis, :], [0]), shape=adj_random.shape)
    adj_random = preprocess_graph(adj_random)
    placeholders, sess, opt, model = train_gcn(adj_random, y_train, features, train_mask, y_val, val_mask)
    _, auc, ap = evaluate(features, adj_random, y_test, test_mask, placeholders, sess, opt, model)
    f.write('Random adjacency matrix:\n')
    f.write("Test AUC score: ")
    f.write("{:.5f}".format(auc))
    f.write('\t')
    f.write('Test AUPR score: ')
    f.write("{:.5f}".format(ap))
    f.write('\n')

    emb = train(adj_norm, similarities, features)
    emb = consensus(emb)
    adj_learned = sp.csr_matrix(np.dot(emb, emb.T))
    savemat('output.mat', {'adj':adj_learned})
    adj_learned = loadmat('output.mat')['adj']
    # thr = np.min(np.max(adj_learned, 1))
    thr = 0
    adj_learned.data[adj_learned.data < thr] = 0
    adj_learned.eliminate_zeros()
    adj_learned = preprocess_graph(adj_learned)
    placeholders, sess, opt, model = train_gcn(adj_learned, y_train, features, train_mask, y_val, val_mask)
    _, auc, ap = evaluate(features, adj_learned, y_test, test_mask, placeholders, sess, opt, model)
    f.write('Learned adjacency matrix:\n')
    f.write("Test AUC score: ")
    f.write("{:.5f}".format(auc))
    f.write('\t')
    f.write('Test AUPR score: ')
    f.write("{:.5f}".format(ap))
    f.write('\n\n')
