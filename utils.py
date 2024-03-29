import numpy as np
import scipy.sparse as sp

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

flags = tf.app.flags
FLAGS = flags.FLAGS

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def construct_feed_dict(adj, similarities, features, placeholders):
    # construct feed dictionary
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj})
    feed_dict.update({placeholders['similarities']: similarities})
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    return feed_dict


def weight_variable_glorot(input_dim, output_dim, name=""):
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


def add_noise(adj, noise_level):
    if noise_level > 0:
        # add edges
        num_edges = sp.triu(adj).count_nonzero()
        num_perturb = np.int(num_edges * noise_level / 2)
        idx = sp.find(adj==0)
        idx = np.append([idx[0]], [idx[1]], axis=0)
        idx = np.transpose(idx[:, idx[0, :] < idx[1, :]])
        np.random.shuffle(idx)
        idx = np.transpose(idx[:num_perturb, :])
        noise_matrix = sp.csr_matrix((np.ones(num_perturb), (idx[0,:], idx[1,:])), shape=adj.shape)
        noise_matrix = noise_matrix + noise_matrix.transpose()
        noise_matrix.data = np.ones(len(noise_matrix.data))
        new_adj = adj + noise_matrix
        
        # remove edges
        idx = sp.find(sp.triu(adj))
        idx = np.transpose(np.append([idx[0]], [idx[1]], axis=0))
        values, counts = np.unique(idx, return_counts=True)
        to_remove = values[counts==1]
        idx = idx[~(np.isin(idx, to_remove)).any(axis=1), :]
        np.random.shuffle(idx)
        idx = np.transpose(idx[:num_perturb, :])
        noise_matrix = sp.csr_matrix((np.ones(num_perturb), (idx[0,:], idx[1,:])), shape=adj.shape)
        noise_matrix = noise_matrix + noise_matrix.transpose()
        new_adj = new_adj - noise_matrix
        new_adj.eliminate_zeros()
        return new_adj
    else:
        return adj