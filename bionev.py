import numpy as np
import networkx as nx
import random
import copy
import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer


def precision_at_k_score(y_true, y_pred_proba, k=1000, pos_label=1):
    topk = [y_true_ == pos_label for y_true_, y_pred_proba_ in sorted(zip(y_true, y_pred_proba), key=lambda y: y[1], reverse=True)[:k]]
    return sum(topk) / len(topk)

def LinkPrediction(embedding_look_up, original_graph, train_graph, test_pos_edges, seed):
    random.seed(seed)

    train_neg_edges = generate_neg_edges(original_graph, len(train_graph.edges()), seed)

    # create a auxiliary graph to ensure that testing negative edges will not used in training
    G_aux = copy.deepcopy(original_graph)
    G_aux.add_edges_from(train_neg_edges)
    test_neg_edges = generate_neg_edges(G_aux, len(test_pos_edges), seed)

    # construct X_train, y_train, X_test, y_test
    X_train = []
    y_train = []
    for edge in train_graph.edges():
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_train.append(feature_vector)
        y_train.append(1)
    for edge in train_neg_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_train.append(feature_vector)
        y_train.append(0)

    X_test = []
    y_test = []
    for edge in test_pos_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_test.append(feature_vector)
        y_test.append(1)
    for edge in test_neg_edges:
        node_u_emb = embedding_look_up[edge[0]]
        node_v_emb = embedding_look_up[edge[1]]
        feature_vector = np.append(node_u_emb, node_v_emb)
        X_test.append(feature_vector)
        y_test.append(0)

    # shuffle for training and testing
    c = list(zip(X_train, y_train))
    random.shuffle(c)
    X_train, y_train = zip(*c)

    c = list(zip(X_test, y_test))
    random.shuffle(c)
    X_test, y_test = zip(*c)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    clf1 = LogisticRegression(random_state=seed, solver='lbfgs', max_iter=1000)
    clf1.fit(X_train, y_train)
    y_pred_proba = clf1.predict_proba(X_test)[:, 1]
    y_pred = clf1.predict(X_test)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f = open('results.txt', 'a')

    """ precision_at_k = [precision_at_k_score(y_test, y_pred_proba, x) for x in [1, 10, 100, 1000, 10000]]
    for score in precision_at_k:
        f.write(f'{score:.3f},')
    f.write('\n') """

    f.write('#' * 9 + ' Link Prediction Performance ' + '#' * 9 + '\n')
    f.write(f'AUC-ROC: {auc_roc:.3f}, AUC-PR: {auc_pr:.3f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}' + '\n')
    f.write('#' * 50 + '\n' + '\n')
    f.close()
    return auc_roc, auc_pr, accuracy, f1


def NodeClassification(embedding_look_up, node_list, labels, testing_ratio, seed):

    X_train, y_train, X_test, y_test = split_train_test_classify(embedding_look_up, node_list, labels,
                                                                 testing_ratio=testing_ratio,seed=seed)
    binarizer = MultiLabelBinarizer(sparse_output=True)
    y_all = np.append(y_train, y_test)
    binarizer.fit(y_all)
    y_train = np.array(binarizer.transform(y_train).todense())
    y_test = np.array(binarizer.transform(y_test).todense())
    model = OneVsRestClassifier(LogisticRegression(random_state=seed, solver='lbfgs'))
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)

    ## small trick : we assume that we know how many label to predict
    y_pred = get_y_pred(y_test, y_pred_prob)

    accuracy = accuracy_score(y_test, y_pred)
    micro_f1 = f1_score(y_test, y_pred, average="micro")
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    f = open('results.txt', 'a')
    f.write('#' * 9 + ' Node Classification Performance ' + '#' * 9 + '\n')
    f.write(f'Accuracy: {accuracy:.3f}, Micro-F1: {micro_f1:.3f}, Macro-F1: {macro_f1:.3f}' + '\n')
    f.write('#' * 50 + '\n' + '\n')
    f.close()
    return accuracy, micro_f1, macro_f1


def generate_neg_edges(original_graph, testing_edges_num, seed):
    L = list(original_graph.nodes())

    # create a complete graph
    G = nx.Graph()
    G.add_nodes_from(L)
    G.add_edges_from(itertools.combinations(L, 2))
    # remove original edges
    G.remove_edges_from(original_graph.edges())
    random.seed(seed)
    neg_edges = random.sample(G.edges, testing_edges_num)
    return neg_edges


def split_train_test_classify(embedding_look_up, X, Y, seed, testing_ratio=0.2):
    state = np.random.get_state()
    training_ratio = 1 - testing_ratio
    training_size = int(training_ratio * len(X))
    shuffle_indices = np.random.permutation(np.arange(len(X)))
    X_train = [embedding_look_up[X[shuffle_indices[i]]] for i in range(training_size)]
    Y_train = [Y[shuffle_indices[i]] for i in range(training_size)]
    X_test = [embedding_look_up[X[shuffle_indices[i]]] for i in range(training_size, len(X))]
    Y_test = [Y[shuffle_indices[i]] for i in range(training_size, len(X))]

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    np.random.set_state(state)
    return X_train, Y_train, X_test, Y_test


def get_y_pred(y_test, y_pred_prob):
    y_pred = np.zeros(y_pred_prob.shape)
    sort_index = np.flip(np.argsort(y_pred_prob, axis=1), 1)
    for i in range(y_test.shape[0]):
        num = np.sum(y_test[i])
        for j in range(num):
            y_pred[i][sort_index[i][j]] = 1
    return y_pred