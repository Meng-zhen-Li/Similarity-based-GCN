import random
import tensorflow.compat.v1 as tf

from utils import weight_variable_glorot

devices = tf.config.get_visible_devices('GPU')
if len(devices) == 0:
    devices = tf.config.get_visible_devices()
devices = [device.name.replace('physical_device:', '') for device in devices]

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    # Properties
        name: String, defines the variable scope of the layer.
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., sparse_inputs=False, act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        if self.sparse_inputs:
            x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        else:
            x = tf.nn.dropout(x, 1-self.dropout)
            x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

class LinearLayer(Layer):
    def __init__(self, input_dim, output_dim, num_graphs=1, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim * num_graphs, name="weights")
        self.num_graphs = num_graphs
        self.dropout = dropout
        self.act = act
        self.input_dim = input_dim

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        outputs = [None for i in range(self.num_graphs)]
        for sim_idx in range(self.num_graphs):
            with tf.device(devices[sim_idx % len(devices)]):
                outputs[sim_idx] = tf.matmul(inputs, tf.gather(self.vars['weights'], tf.range(self.input_dim) + sim_idx * self.input_dim, axis=1))
        outputs = self.act(tf.concat(outputs, 0))
        return outputs

class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, sim_idx=0, num_graphs=1, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.num_graphs = num_graphs
        self.dropout = dropout
        self.act = act
        self.input_dim = input_dim
        self.sim_idx = sim_idx

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        inputs = tf.split(inputs, self.num_graphs)
        outputs = [None for i in range(self.num_graphs)]
        for sim_idx in range(self.num_graphs):
            with tf.device(devices[sim_idx % len(devices)]):
                outputs[sim_idx] = tf.matmul(inputs[sim_idx], tf.transpose(inputs[sim_idx]))
        outputs = self.act(tf.concat(outputs, 0))
        return outputs