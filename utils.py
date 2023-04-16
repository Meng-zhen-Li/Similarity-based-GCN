import numpy as np
import scipy.sparse as sp

import tensorflow.compat.v1 as tf


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def construct_feed_dict(adj, similarity, features, placeholders, sim_idx=0):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['similarity']: similarity})
    feed_dict.update({placeholders['sim_idx']: sim_idx})
    return feed_dict


def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj, return_sparse=False):
    adj = sp.coo_matrix(adj)
    if np.sum(adj.diagonal()) == 0:
        adj_ = adj + sp.eye(adj.shape[0])
    else:
        adj_ = adj
    rowsum = np.array(adj_.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj_.dot(d_mat_inv_sqrt).transpose().dot(
        d_mat_inv_sqrt).tocoo()
    if return_sparse:
        return adj_normalized
    return sparse_to_tuple(adj_normalized)


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)
