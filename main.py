from __future__ import division
from __future__ import print_function

from evaluation import *
from gcn_eval.train import *

from utils import preprocess_graph, preprocess_features
from similarity import similarity_matrix, normalize_rowsum
from input_data import load_data
from train import train
from consensus import consensus

import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow.compat.v1 as tf

import os
from scipy.io import loadmat, savemat

os.environ['CUDA_VISIBLE_DEVICES'] = ""
tf.disable_eager_execution()

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.02, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 
                   'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('dataset', 'STRING_PPI', 'Dataset string.')
flags.DEFINE_string('task', 'link_prediction', 'Prediction task, link prediction or node classification.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

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

    embeddings = []
    adj_learned = preprocess_graph(adj_train)
    adj_max = similarities[0]
    adj_min = similarities[0]
    f.write('Using similarity matrices:\n')
    for similarity in similarities:
        similarity = similarity - sp.dia_matrix((similarity.diagonal()[np.newaxis, :], [0]), shape=similarity.shape)
        similarity = similarity / np.max(similarity)
        similarity_norm = preprocess_graph(similarity)
        adj_pred = train_gae(similarity_norm, adj_train, features, val_edges, val_edges_false)
        roc_score, ap_score = get_roc_score(test_edges, test_edges_false, adj_pred)
        f.write('Test AUC score: ')
        f.write("{:.5f}".format(roc_score))
        f.write('\t')
        f.write('Test AUPR score: ')
        f.write("{:.5f}".format(ap_score))
        f.write('\n')
        adj_max = adj_max.maximum(similarity)
        adj_min = adj_min.minimum(similarity)
        emb = train(adj_learned, similarity, features)
        embeddings.append(emb)

    adj_random = sp.random(adj.shape[0], adj.shape[0], density=adj.count_nonzero()/adj.shape[0]/adj.shape[0])
    adj_random = adj_random / adj_random.max()
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
    
    adj_max = preprocess_graph(adj_max - sp.dia_matrix((adj_max.diagonal()[np.newaxis, :], [0]), shape=adj_max.shape))
    adj_pred = train_gae(adj_max, adj_train, features, val_edges, val_edges_false)
    roc_score, ap_score = get_roc_score(test_edges, test_edges_false, adj_pred)
    f.write('Maximum adjacency matrix:\n')
    f.write('Test AUC score: ')
    f.write("{:.5f}".format(roc_score))
    f.write('\t')
    f.write('Test AUPR score: ')
    f.write("{:.5f}".format(ap_score))
    f.write('\n')
    adj_min = preprocess_graph(adj_min - sp.dia_matrix((adj_min.diagonal()[np.newaxis, :], [0]), shape=adj_min.shape))
    adj_pred = train_gae(adj_min, adj_train, features, val_edges, val_edges_false)
    roc_score, ap_score = get_roc_score(test_edges, test_edges_false, adj_pred)
    f.write('Minimum adjacency matrix:\n')
    f.write('Test AUC score: ')
    f.write("{:.5f}".format(roc_score))
    f.write('\t')
    f.write('Test AUPR score: ')
    f.write("{:.5f}".format(ap_score))
    f.write('\n')

    emb = consensus(embeddings)
    adj_learned = sp.csr_matrix(np.dot(emb, emb.T))
    # adj_learned = sp.csr_matrix(cosine_similarity(emb))
    adj_learned = adj_learned - sp.dia_matrix((adj_learned.diagonal()[np.newaxis, :], [0]), shape=adj_learned.shape)
    thr = np.min(np.max(adj_learned, 1))
    adj_learned.data[adj_learned.data < thr] = 0
    adj_learned.eliminate_zeros()
    if thr < 0:
        adj_learned.data = adj_learned.data - thr
    adj_learned = adj_learned / np.max(adj_learned)
    # adj_learned = normalize_rowsum(adj_learned)
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
    placeholders, sess, opt, model = train_gcn(adj_norm, y_train, features, train_mask, y_val, val_mask)
    _, auc, ap = evaluate(features, preprocess_graph(adj), y_test, test_mask, placeholders, sess, opt, model)
    f.write('Using original adjacency matrix:\n')
    f.write("Test AUC score: ")
    f.write("{:.5f}".format(auc))
    f.write('\t')
    f.write('Test AUPR score: ')
    f.write("{:.5f}".format(ap))
    f.write('\n')

    f.write('Using similarity matrices:\n')
    similarities = similarity_matrix(adj)
    adj_max = similarities[0]
    adj_min = similarities[0]
    embeddings = []
    for similarity in similarities:
        similarity = similarity - sp.dia_matrix((similarity.diagonal()[np.newaxis, :], [0]), shape=similarity.shape)
        similarity = similarity / np.max(similarity)
        similarity.eliminate_zeros()
        emb = train(adj_norm, similarity, features)
        embeddings.append(emb)
        similarity_norm = preprocess_graph(similarity)
        placeholders, sess, opt, model = train_gcn(similarity_norm, y_train, features, train_mask, y_val, val_mask)
        _, auc, ap = evaluate(features, similarity_norm, y_test, test_mask, placeholders, sess, opt, model)
        f.write("Test AUC score: ")
        f.write("{:.5f}".format(auc))
        f.write('\t')
        f.write('Test AUPR score: ')
        f.write("{:.5f}".format(ap))
        f.write('\n')

        adj_max = adj_max.maximum(similarity)
        adj_min = adj_min.minimum(similarity)

    adj_random = sp.random(adj.shape[0], adj.shape[0], density=0.1)
    adj_random = adj_random / adj_random.max()
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

    adj_max = preprocess_graph(adj_max - sp.dia_matrix((adj_max.diagonal()[np.newaxis, :], [0]), shape=adj_max.shape))
    placeholders, sess, opt, model = train_gcn(adj_max, y_train, features, train_mask, y_val, val_mask)
    _, auc, ap = evaluate(features, adj_max, y_test, test_mask, placeholders, sess, opt, model)
    f.write('Maximum adjacency matrix:\n')
    f.write("Test AUC score: ")
    f.write("{:.5f}".format(auc))
    f.write('\t')
    f.write('Test AUPR score: ')
    f.write("{:.5f}".format(ap))
    f.write('\n')
    adj_min = preprocess_graph(adj_min - sp.dia_matrix((adj_min.diagonal()[np.newaxis, :], [0]), shape=adj_min.shape))
    placeholders, sess, opt, model = train_gcn(adj_min, y_train, features, train_mask, y_val, val_mask)
    _, auc, ap = evaluate(features, adj_max, y_test, test_mask, placeholders, sess, opt, model)
    f.write('Minimum adjacency matrix:\n')
    f.write("Test AUC score: ")
    f.write("{:.5f}".format(auc))
    f.write('\t')
    f.write('Test AUPR score: ')
    f.write("{:.5f}".format(ap))
    f.write('\n')

    emb = consensus(embeddings)
    # adj_learned = sp.csr_matrix(cosine_similarity(emb))
    adj_learned = sp.csr_matrix(np.dot(emb, emb.T))
    adj_learned = adj_learned - sp.dia_matrix((adj_learned.diagonal()[np.newaxis, :], [0]), shape=adj_learned.shape)
    # thr = np.min(np.max(adj_learned, 1))
    thr = np.min(np.sort(adj_learned.todense())[:,-100])
    adj_learned.data[adj_learned.data < thr] = 0
    adj_learned.eliminate_zeros()
    if thr < 0:
        adj_learned.data = adj_learned.data - thr
    adj_learned = adj_learned / np.max(adj_learned)
    adj_learned.data = np.random.rand(len(adj_learned.data))
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
