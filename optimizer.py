import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

devices = tf.config.get_visible_devices('GPU')
if len(devices) == 0:
    devices = tf.config.get_visible_devices()
devices = [device.name.replace('physical_device:', '') for device in devices]

class Optimizer(object):
    def __init__(self, preds, labels, num_graphs):
        self.preds_sub = tf.split(preds, num_graphs)
        self.labels_sub = tf.split(labels, num_graphs)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.costs = []
        self.r2 = []
        self.grads_vars = None
        for sim_idx in range(num_graphs):
            with tf.device(devices[sim_idx % len(devices)]):
                cost = tf.losses.mean_squared_error(labels=self.labels_sub[sim_idx], predictions=self.preds_sub[sim_idx])
                grads_var = self.optimizer.compute_gradients(cost)
                if self.grads_vars is None:
                    self.grads_vars = grads_var
                else:
                    self.grads_vars = [(x[0] + y[0], x[1]) for x, y in zip(self.grads_vars, grads_var)]
                self.costs.append(cost)
                unexplained_error = tf.reduce_sum(tf.square(self.labels_sub[sim_idx] - self.preds_sub[sim_idx]))
                total_error = tf.reduce_sum(tf.square(self.labels_sub[sim_idx] - tf.reduce_mean(self.labels_sub[sim_idx], axis=0)))
                R2 = 1. - tf.div(unexplained_error, total_error)
                self.r2.append(R2)
        self.grads_vars[:2] = [(self.grads_vars[i][0] / num_graphs, self.grads_vars[i][1]) for i in range(2)]
        self.opt_op = self.optimizer.apply_gradients(self.grads_vars)