from __future__ import division
from __future__ import print_function

from bionev import LinkPrediction, NodeClassification

from utils import preprocess_graph, preprocess_features, add_noise
from similarity import similarity_matrix
from input_data import load_data
from train import train
from consensus import consensus

import scipy.sparse as sp
import numpy as np
import networkx as nx
import tensorflow.compat.v1 as tf

import os
from scipy.io import loadmat, savemat

tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('dataset', 'DrugBank_DDI', 'Dataset string.')
flags.DEFINE_float('noise_level', 0., 'Percentage of perturbed edges.')
flags.DEFINE_string('task', 'link_prediction', 'Prediction task, link prediction or node classification.')
flags.DEFINE_integer('sim_idx', 0, 'To use all similarities(0) or to use one of them(index of similarity).')

dataset_str = FLAGS.dataset


if FLAGS.task == 'link_prediction':
    adj, features, adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = load_data(dataset_str)
    adj_train = add_noise(adj_train, FLAGS.noise_level)
else:
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_list, labels = load_data(dataset_str)
    adj = add_noise(adj, FLAGS.noise_level)

features = preprocess_features(features)
    

# Link Prediction
if FLAGS.task == 'link_prediction':
    adj_norm = preprocess_graph(adj_train)
    similarities = similarity_matrix(adj_train)

    if FLAGS.sim_idx != 0:
        similarities = [similarities[FLAGS.sim_idx - 1]]

    emb = train(adj_norm, similarities, features)
    if len(emb) > 1:
        emb = consensus(emb)
    else:
        emb = emb[0]
    auc_roc, auc_pr, accuracy, f1 = LinkPrediction(emb, nx.from_numpy_matrix(adj.todense()), nx.from_numpy_matrix(adj_train.todense()), np.append(test_edges, val_edges, axis=0), None)



# Node Classification
if FLAGS.task == 'node_classification':
    adj_norm = preprocess_graph(adj)
    similarities = similarity_matrix(adj)

    if FLAGS.sim_idx != 0:
        similarities = [similarities[FLAGS.sim_idx - 1]]

    emb = train(adj_norm, similarities, features)
    emb = consensus(emb)
    accuracy, micro_f1, macro_f1 = NodeClassification(emb, node_list, labels, 0.2, 0)
