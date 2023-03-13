import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Optimizer(object):
    def __init__(self, preds, labels):
        preds_sub = preds
        labels_sub = labels

        # self.cost = tf.nn.l2_loss(preds_sub - labels_sub)
        self.cost = tf.nn.softmax_cross_entropy_with_logits(labels=labels_sub, logits=preds_sub)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)