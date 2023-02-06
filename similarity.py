import numpy as np
import scipy.sparse as sp
import networkx as nx
from itertools import combinations
from sklearn.preprocessing import normalize
import time

def normalize_rowsum(data):
    data = normalize(data, norm='l1')
    normalized = (data + data.transpose()) / 2
    return normalized

def similarity_matrix(adj):
    n = adj.shape[0]
    G = nx.from_scipy_sparse_array(adj)

    """ simrank = nx.simrank_similarity(G)
    simrank = sp.csr_matrix([[simrank[u][v] for v in G] for u in G])
    simrank = simrank - sp.dia_matrix((simrank.diagonal()[np.newaxis, :], [0]), shape=simrank.shape)
    simrank.eliminate_zeros()
    simrank = normalize_rowsum(simrank)

    adamic_adar = list(nx.adamic_adar_index(G, combinations(range(n), 2)))
    adamic_adar = nx.Graph((x, y, {'weight': v}) for x, y, v in adamic_adar)
    adamic_adar = nx.adjacency_matrix(adamic_adar)
    adamic_adar.eliminate_zeros()
    adamic_adar = normalize_rowsum(adamic_adar)

    jaccard = list(nx.jaccard_coefficient(G, combinations(range(n), 2)))
    jaccard = nx.Graph((x, y, {'weight': v}) for x, y, v in jaccard)
    jaccard = nx.adjacency_matrix(jaccard)
    jaccard.eliminate_zeros()
    jaccard = normalize_rowsum(jaccard) """

    D = sp.csr_matrix(np.sum(adj, axis=0))
    C = adj.dot(adj.transpose())
    C_logical = C
    C_logical.data[np.nonzero(C.data)] = 1
    U = D.multiply(C_logical) + D.transpose().multiply(C_logical) - C
    U.data = 1 / U.data
    A = C.multiply(U)
    A.data[np.isnan(A.data)] = 0
    A.data[np.isinf(A.data)] = 0
    A.data = 1 + A.data
    jaccard_index = A
    jaccard_index = jaccard_index - sp.dia_matrix((jaccard_index.diagonal()[np.newaxis, :], [0]), shape=jaccard_index.shape)
    jaccard_index.eliminate_zeros()
    jaccard_index = normalize_rowsum(jaccard_index)
    # jaccard_index.data[np.nonzero(adj.data)] = 0

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
    A.data = A.data + 1
    von_neumann = A
    # von_neumann.data[np.nonzero(adj.data)] = 0
    von_neumann = von_neumann - sp.dia_matrix((von_neumann.diagonal()[np.newaxis, :], [0]), shape=von_neumann.shape)
    von_neumann.eliminate_zeros()
    von_neumann = normalize_rowsum(von_neumann)

    An = adj.multiply(np.sqrt(1 / np.sum(adj, axis=0)))
    A = (An * An) * adj
    A.data[np.isnan(A.data)] = 0
    A.data[np.isinf(A.data)] = 0
    A.data = A.data + 1
    L3 = A
    # L3.data[np.nonzero(adj.data)] = 0
    L3 = L3 - sp.dia_matrix((L3.diagonal()[np.newaxis, :], [0]), shape=L3.shape)
    L3.eliminate_zeros()
    L3 = normalize_rowsum(L3)

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
    A.data = A.data + 1
    rwr = A
    rwr = rwr - sp.dia_matrix((rwr.diagonal()[np.newaxis, :], [0]), shape=rwr.shape)
    rwr.eliminate_zeros()
    # rwr.data[np.nonzero(adj.data)] = 0
    rwr = normalize_rowsum(rwr)

    return [jaccard_index, von_neumann, L3, rwr]