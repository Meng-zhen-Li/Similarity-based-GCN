import numpy as np
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.io import loadmat
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=bool)


def load_data(datastr):
    data = loadmat('data/' + datastr + '.mat')
    adj = data['adj']

    if 'labels' in data.keys():
        labels = data['labels'].todense()
        idx_train = data['train_idx']
        idx_test = data['test_idx']
        idx_val = data['val_idx']
        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]
    else:
        adj_train = data['adj_train']
        train_edges = data['train_edges']
        val_edges = data['val_edges']
        val_edges_false = data['val_edges_false']
        test_edges = data['test_edges']
        test_edges_false = data['test_edges_false']
    
    if 'features' in data.keys():
        features = data['features']
    else:
        features = sp.identity(adj.shape[0])

    if FLAGS.task == 'node_classification':
        return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    else:
        return adj, features, adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false