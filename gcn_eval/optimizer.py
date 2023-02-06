import tensorflow.compat.v1 as tf
from gcn_eval.utils import masked_softmax_cross_entropy

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32), tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

class OptimizerGCN(object):
    def __init__(self, layers, outputs, placeholders, pos_weight, norm):

        self.loss = 0
        self.accuracy = 0
        for var in layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += norm * masked_softmax_cross_entropy(outputs, placeholders['labels'], placeholders['labels_mask'], pos_weight)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

        self.opt_op = self.optimizer.minimize(self.loss)