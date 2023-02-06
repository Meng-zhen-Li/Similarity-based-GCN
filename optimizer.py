import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Optimizer(object):
    def __init__(self, preds, labels):
        preds_sub = preds
        labels_sub = labels

        self.cost = tf.reduce_mean(tf.nn.l2_loss(preds_sub - labels_sub))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)