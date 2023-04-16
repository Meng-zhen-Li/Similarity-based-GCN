from layers import GraphConvolution, InnerProductDecoder, LinearLayer
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModel(Model):
    def __init__(self, placeholders, num_features, features_nonzero, num_graphs, **kwargs):
        super(GCNModel, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.adj = placeholders['adj']
        self.sim_idx = placeholders['sim_idx']
        self.dropout = placeholders['dropout']
        self.features_nonzero = features_nonzero
        self.num_graphs = num_graphs
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolution(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              sparse_inputs=True,
                                              dropout=self.dropout)(self.inputs)

        self.hidden2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           features_nonzero=self.features_nonzero,
                                           act=tf.nn.relu,
                                           dropout=self.dropout)(self.hidden1)

        self.embeddings = LinearLayer(input_dim=FLAGS.hidden2,
                                           output_dim=FLAGS.hidden2,
                                           num_graphs = self.num_graphs,
                                           act=lambda x: x,
                                           sim_idx=self.sim_idx,
                                           dropout=self.dropout)(self.hidden2)

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                           num_graphs = self.num_graphs,
                                           act=lambda x: x)(self.embeddings)