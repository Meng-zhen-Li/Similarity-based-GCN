import numpy as np
import scipy.sparse as sp
from scipy.linalg import eig
import logging
import os
import tensorflow.compat.v1 as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

class GCCA:

    def __init__(self, n_components=2, reg_param=0.1):

        # log setting
        program = os.path.basename(__name__)
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s')

        # GCCA params
        self.n_components = n_components
        self.reg_param = reg_param

        # result of fitting
        self.data_num = 0
        self.cov_mat = [[]]
        self.h_list = []
        self.eigvals = np.array([])

        # result of transformation
        self.z_list = []

    def eigvec_normalization(self, eig_vecs, x_var):
        self.logger.info("normalization")
        z_var = np.dot(eig_vecs.T, np.dot(x_var, eig_vecs))
        invvar = np.diag(np.reciprocal(np.sqrt(np.diag(z_var))))
        eig_vecs = np.dot(eig_vecs, invvar)
        # print np.dot(eig_vecs.T, np.dot(x_var, eig_vecs)).round().astype(int)
        return eig_vecs


    def solve_eigprob(self, left, right):

        self.logger.info("calculating eigen dimension")
        eig_dim = min([np.linalg.matrix_rank(left), np.linalg.matrix_rank(right)])

        self.logger.info("calculating eigenvalues & eigenvector")
        eig_vals, eig_vecs = eig(left, right)

        self.logger.info("sorting eigenvalues & eigenvector")
        sort_indices = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sort_indices][:eig_dim].real
        eig_vecs = eig_vecs[:,sort_indices][:,:eig_dim].real

        return eig_vals, eig_vecs

    def calc_cov_mat(self, x_list):

        data_num = len(x_list)

        self.logger.info("calc variance & covariance matrix")
        z = np.vstack([x.T for x in x_list])
        cov = np.cov(z)
        d_list = [0] + [sum([len(x.T) for x in x_list][:i + 1]) for i in range(data_num)]
        cov_mat = [[np.array([]) for col in range(data_num)] for row in range(data_num)]
        for i in range(data_num):
            for j in range(data_num):
                i_start, i_end = d_list[i], d_list[i + 1]
                j_start, j_end = d_list[j], d_list[j + 1]
                cov_mat[i][j] = cov[i_start:i_end, j_start:j_end]

        return cov_mat

    def add_regularization_term(self, cov_mat):

        data_num = len(cov_mat)

        # regularization
        self.logger.info("adding regularization term")
        for i in range(data_num):
            cov_mat[i][i] += self.reg_param * np.average(np.diag(cov_mat[i][i])) * np.eye(cov_mat[i][i].shape[0])

        return cov_mat

    def fit(self, *x_list):

        # data size check
        data_num = len(x_list)
        self.logger.info("data num is %d", data_num)
        for i, x in enumerate(x_list):
            self.logger.info("data shape x_%d: %s", i, x.shape)

        self.logger.info("normalizing")
        x_norm_list = [ self.normalize(x) for x in x_list]

        d_list = [0] + [sum([len(x.T) for x in x_list][:i + 1]) for i in range(data_num)]
        cov_mat = self.calc_cov_mat(x_norm_list)
        cov_mat = self.add_regularization_term(cov_mat)

        self.logger.info("calculating generalized eigenvalue problem ( A*u = (lambda)*B*u )")
        # left = A, right = B
        left = 0.5 * np.vstack(
            [
                np.hstack([np.zeros_like(cov_mat[i][j]) if i == j else cov_mat[i][j] for j in range(data_num)])
                for i in range(data_num)
            ]
        )
        right = np.vstack(
            [
                np.hstack([np.zeros_like(cov_mat[i][j]) if i != j else cov_mat[i][j] for j in range(data_num)])
                for i in range(data_num)
            ]
        )

        # calc GEV
        self.logger.info("solving")
        eigvals, eigvecs = self.solve_eigprob(left, right)

        h_list = [eigvecs[start:end] for start, end in zip(d_list[0:-1], d_list[1:])]
        h_list_norm = [self.eigvec_normalization(h, cov_mat[i][i]) for i, h in enumerate(h_list)]

        # substitute local variables for member variables
        self.data_num = data_num
        self.cov_mat = cov_mat
        self.h_list = h_list_norm
        self.eigvals = eigvals

    def transform(self, *x_list):

        # data size check
        data_num = len(x_list)
        self.logger.info("data num is %d", data_num)
        for i, x in enumerate(x_list):
            self.logger.info("data shape x_%d: %s", i, x.shape)

        if self.data_num != data_num:
            raise Exception('data num when fitting is different from data num to be transformed')

        self.logger.info("normalizing")
        x_norm_list = [ self.normalize(x) for x in x_list]

        self.logger.info("transform matrices by GCCA")
        z_list = [np.dot(x, h_vec)[:, :self.n_components] for x, h_vec in zip(x_norm_list, self.h_list)]

        self.z_list = z_list

        return z_list

    def fit_transform(self, *x_list):
        self.fit(x_list)
        self.transform(x_list)

    @staticmethod
    def normalize(mat):
        m = np.mean(mat, axis=0)
        mat = mat - m
        return mat

def consensus(embeddings):
    gcca = GCCA(reg_param=0.1, n_components=FLAGS.hidden2)
    gcca.fit(*embeddings)
    embeddings = gcca.transform(*embeddings)
    return np.mean(embeddings, axis=0)