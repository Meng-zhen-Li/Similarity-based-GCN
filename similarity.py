import numpy as np
import scipy.sparse as sp
import networkx as nx
from itertools import combinations

def similarity_matrix(adj):
    n = adj.shape[0]
    # G = nx.from_scipy_sparse_array(adj)

    """ simrank = nx.simrank_similarity(G)
    simrank = sp.csr_matrix([[simrank[u][v] for v in G] for u in G]) """

    degrees = adj.sum(axis=0)
    degrees[degrees == 1] = 0
    weights = sp.csr_matrix(1 / np.log10(degrees))
    A = adj.multiply(weights) * adj.T
    A.data[np.isnan(A.data)] = 0
    A.data[np.isinf(A.data)] = 0
    adamic_adar = A

    D = sp.csr_matrix(np.sum(adj, axis=0))
    C = adj.dot(adj.transpose())
    C_logical = C
    C_logical.data[np.nonzero(C.data)] = 1
    U = D.multiply(C_logical) + D.transpose().multiply(C_logical) - C
    U.data = 1 / U.data
    A = C.multiply(U)
    A.data[np.isnan(A.data)] = 0
    A.data[np.isinf(A.data)] = 0
    jaccard_index = A

    alpha = 0.5
    S = sp.csr_matrix(np.sqrt(1 / np.sum(adj, axis=0))).multiply(adj)
    S = S.multiply(np.sqrt(1 / np.sum(adj, axis=1)))
    S.data[np.isnan(S.data)] = 0
    S.data[np.isinf(S.data)] = 0
    S2 = S
    A = alpha * S
    alpha2 = alpha
    for i in range(1, 3):
        S2 = S2 * S
        alpha2 = alpha2 * alpha
        A = A + alpha2 * S2
    A.data[np.isnan(A.data)] = 0
    A.data[np.isinf(A.data)] = 0
    von_neumann = A

    D = sp.csr_matrix(np.sum(adj, axis=0))
    D[D == 0] = 1
    D.data = 1 / D.data
    S = D.multiply(adj)
    S.data[np.isnan(S.data)] = 0
    S.data[np.isinf(S.data)] = 0
    S2 = S
    A = alpha * S
    alpha2 = alpha
    for i in range(3):
        S2 = S2 * S
        alpha2 = alpha2 * alpha
        A = A + alpha2 * S2
    A.data[np.isnan(A.data)] = 0
    A.data[np.isinf(A.data)] = 0
    rwr = A

    return [adamic_adar, jaccard_index, von_neumann, rwr, adj]