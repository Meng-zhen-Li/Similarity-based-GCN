import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

class Optimizer(object):
    def __init__(self, preds, labels, placeholders):
        self.preds_sub = preds
        self.labels_sub = labels

        self.cost = tf.losses.mean_squared_error(labels=self.labels_sub, predictions=self.preds_sub)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        if FLAGS.distributed == 0:
            self.opt_op = self.optimizer.minimize(self.cost)
        else:
            self.grads_vars = self.optimizer.compute_gradients(self.cost)
            self.vars = [x[1] for x in self.grads_vars]
            self.gradients = [placeholders['gradient' + str(i)] for i in range(3)]
            self.opt_op = self.optimizer.apply_gradients(list(zip(self.gradients, self.vars)))