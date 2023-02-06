import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score
import tensorflow.compat.v1 as tf

from gcn_eval.utils import construct_feed_dict

def get_roc_score(edges_pos, edges_neg, adj_rec):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

def evaluate(features, support, labels, mask, placeholders, sess, opt, model):
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs = sess.run([opt.loss, model.outputs], feed_dict=feed_dict_val)
    auc, ap = get_masked_auc(outs[1], labels, mask)
    return outs[0], auc, ap

def get_masked_auc(preds, labels, mask):
    mask = tf.cast(mask, dtype=tf.float32)
    idx = tf.where(mask)
    idx = tf.Session().run(idx).reshape([-1])
    preds = preds[idx]
    labels = labels[idx]
    return np.average(roc_auc_score(labels, preds, average=None)), np.average(average_precision_score(labels, preds, average=None))