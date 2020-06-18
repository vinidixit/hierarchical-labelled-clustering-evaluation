import networkx as nx
import itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import functools
import time

def is_valid_category(category):
    return category in ['CCAT', 'ECAT'] or ((category.startswith('I') or \
                                             category.startswith('C') or category.startswith('E')) \
                                            and category[1].isnumeric())

def get_label_doc_graph(label_G, doc_df, label_name):
    G = nx.Graph(label_G.edges())

    if label_name in ['labels', 'selected_labels']:
        G.remove_node(('root',))

    for doc in doc_df.index:
        labels = doc_df.loc[doc][label_name]

        for label in labels:
            G.add_edge(label, doc)

    return G


def get_disconnected_docs(G, doc_df, remove_nodes=['CCAT','ECAT']):
    disconnected_docs = []
    disconnected_labels = []

    for node in remove_nodes:
        if node in G:
            G.remove_node(node)

    for comp in nx.connected_components(G):
        docs = []
        labels = []

        for node in comp:
            if node in doc_df.index:
                docs.append(node)
            else:
                labels.append(node)

        if len(docs) > 0:
            disconnected_docs.append(docs)
            disconnected_labels.append(labels)

    return disconnected_docs, disconnected_labels


def get_common_labels(doc1, doc2, doc_df, label_name, remove_labels=[]):
    doc1_labels = doc_df.loc[doc1][label_name]
    doc2_labels = doc_df.loc[doc2][label_name]

    if not remove_labels:
        return set(doc1_labels).isdisjoint(doc2_labels)

    if not doc1_labels or not doc2_labels:
          return None

    intersection = set(doc1_labels).intersection(doc2_labels)

    for label in remove_labels:
        if label in intersection:
            intersection.remove(label)

    return intersection

# Test docs from same cluster
def test_same_cluster(arg_input):
    docs, pred_label_name, pred_doc_df, remove_labels = arg_input

    match = 0
    mismatch = 0
    total = 0
    t0 = time.time()

    for i, d1 in enumerate(docs):
        for j in range(i + 1, len(docs)):
            d2 = docs[j]

            common_classes = get_common_labels(d1, d2, pred_doc_df, pred_label_name, remove_labels)

            if common_classes is None:
                continue

            total += 1

            if common_classes:
                match += 1
            else:
                mismatch += 1

    true_positive = match
    false_negative = mismatch

    return total, true_positive, false_negative

def test_intercluster_docs(arg_input):
    docs1, docs2, pred_label_name, pred_doc_df, remove_labels = arg_input
    # Test docs from different cluster

    match = 0
    mismatch = 0
    total = 0

    for d1 in docs1:

        for d2 in docs2:

            common_classes = get_common_labels(d1, d2, pred_doc_df, pred_label_name, remove_labels)

            if common_classes is None:
                continue

            total += 1

            if common_classes:
                match += 1
            else:
                mismatch += 1

    true_negative = mismatch
    false_positive = match

    return total, false_positive, true_negative


def parallelize(docs, pred_label_name, pred_doc_df, remove_labels=['CCAT', 'ECAT']):
    if pred_label_name in ['labels', 'selected_labels']:
        remove_labels = []

    docs = list(set(docs).intersection(pred_doc_df.index))
    print('\nTotal documents: ', len(docs))

    num_partitions = 30

    splits = list(zip(np.array_split(docs, num_partitions), itertools.repeat(pred_label_name), \
                                                        itertools.repeat(pred_doc_df), itertools.repeat(remove_labels)))


    with ProcessPoolExecutor() as executor:
        mapped_res_gen = executor.map(test_same_cluster, splits)

        result = functools.reduce(
            lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]),
            mapped_res_gen)

    total, match, mismatch = result

    print('Total pairs: ', total, ' Match count: ', match, 'mismatch count :', mismatch)

    if total > 0:
        print('true_positive %.3f %%, false_negative %.3f %%' % (match * 100 / total, mismatch * 100 / total))


    return result

def parallelize_intercluster(docs1, docs2, pred_label_name, pred_doc_df, remove_labels=['CCAT', 'ECAT']):
    num_partitions = 20

    docs1 = list(set(docs1).intersection(pred_doc_df.index))
    docs2 = list(set(docs2).intersection(pred_doc_df.index))
    print('\nTotal documents: ', (len(docs1), len(docs2)))


    splits = list(zip(np.array_split(docs1, num_partitions),np.array_split(docs2, num_partitions), \
                                                        itertools.repeat(pred_label_name), \
                                                        itertools.repeat(pred_doc_df), itertools.repeat(remove_labels)))




    with ProcessPoolExecutor() as executor:
        mapped_res_gen = executor.map(test_intercluster_docs, splits)

        result = functools.reduce(
            lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]),
            mapped_res_gen)

    total, match, mismatch = result

    print('Total pairs: ', total, ' Match count: ', match, 'mismatch count :', mismatch)

    if total > 0:
        print('true_positive %.3f %%, false_negative %.3f %%' % (match * 100 / total, mismatch * 100 / total))

    return result
