import pandas as pd
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import functools
import itertools


class TicketsEvaluator:

    def __init__(self):
        data_dir = '/Users/vdixit/Work/Data/processed_data/'
        data_file = data_dir + 'problem_tickets_consolidated.csv'

        self.problem_tickets_df = pd.read_csv(data_file)
        self.unique_problems = list(self.problem_tickets_df.subject_pr.unique())

        self.problem_ticket_distr = self.get_disconnected_docs()
        self.all_pos_pairs = get_pos_pairs(list(self.problem_ticket_distr.values()))
        self.all_neg_pairs = get_neg_pairs(list(self.problem_ticket_distr.values()))

    def get_disconnected_docs(self):
        disconnected_docs = {}

        for problem in self.unique_problems:
            docids = list(self.problem_tickets_df[self.problem_tickets_df.subject_pr == problem].index)
            disconnected_docs[problem] = list(docids)

        print('disconnected_docs: ', len(disconnected_docs))
        return disconnected_docs

    def prepare_data(self):
        self.problem_tickets_df.subject_pr = self.problem_tickets_df.subject_pr.apply(
            lambda x: x.strip() if type(x) == str else '')

        self.problem_tickets_df.subject_ticket = self.problem_tickets_df.subject_ticket.apply(
            lambda x: x.strip() if type(x) == str else '')

        self.problem_tickets_df.description = self.problem_tickets_df.description.apply(
            lambda x: x.strip() if type(x) == str else '')


    def get_pos_neg_samples(self, sample_size, seed=42):
        random.seed(seed)
        sample_pos_pairs = random.sample(self.all_pos_pairs, sample_size)

        random.seed(seed)
        sample_neg_pairs = random.sample(self.all_neg_pairs, sample_size)
        return sample_pos_pairs, sample_neg_pairs


    def evaluate(self, pos_pairs, neg_pairs, cl_df, label_name='labels'):

        param_names = ['total_pairs', 'pos_pairs', 'neg_pairs']
        metric_names = ['disconnected_clusters', 'precison', 'recall', 'tnr', 'fscore', 'accuracy']

        param_values, metric_values = _get_evaluation_scores(pos_pairs, neg_pairs, cl_df, label_name)

        eval_suffix = "@"+  label_name

        params_map = {}
        for name, value in zip(param_names, param_values):
            params_map[name+eval_suffix] = value

        metrics_map = {}
        for name, value in zip(metric_names, metric_values):
            metrics_map[name + eval_suffix] = value

        return params_map, metrics_map

def _get_evaluation_scores(pos_pairs, neg_pairs, cl_df, label_name='labels'):

    total_assigned_pos, true_positives, false_negatives = parallelize(pos_pairs, cl_df, label_name)

    total_assigned_neg, false_positives, true_negatives = parallelize(neg_pairs, cl_df, label_name)

    tnr = round(true_negatives*100/(true_negatives+false_positives), 4)

    accuracy = round((true_positives + true_negatives) * 100 / (total_assigned_pos + total_assigned_neg), 4)

    precision = round(true_positives * 100 / (true_positives + false_positives), 4)
    recall = round(true_positives * 100 / (true_positives + false_negatives), 4)
    fscore = round(2 * precision * recall / (precision + recall), 4)

    total_pairs = total_assigned_pos + total_assigned_neg

    return (total_pairs, total_assigned_pos, total_assigned_neg), (38, precision, recall, tnr, fscore, accuracy)


def get_pos_pairs(disconnected_docs):
    pos_doc_pairs = []

    for docs in disconnected_docs:
        if len(docs) < 2:
            continue

        for i, d1 in enumerate(docs):
            for j in range(i + 1, len(docs)):
                d2 = docs[j]
                pos_doc_pairs.append(((d1, d2)))

    return pos_doc_pairs


def get_common_labels(doc1, doc2, doc_df, label_name):
    doc1_labels = doc_df.loc[doc1][label_name]
    doc2_labels = doc_df.loc[doc2][label_name]

    if not doc1_labels or not doc2_labels:
        return None

    doc1_labels = np.array(doc1_labels)[:, 0]
    doc2_labels = np.array(doc2_labels)[:, 0]

    intersection = set(doc1_labels).intersection(doc2_labels)
    return intersection


def test_doc_pairs(arg_input):
    doc_pairs, pred_doc_df, pred_label_name = arg_input
    match = 0
    mismatch = 0
    total = 0

    for d1, d2 in doc_pairs:

        if d1 not in pred_doc_df.index or d2 not in pred_doc_df.index:
            continue

        common_classes = get_common_labels(d1, d2, pred_doc_df, pred_label_name)

        if common_classes is None:
            continue

        total += 1

        if common_classes:
            match += 1
        else:
            # print(d1, d2)
            mismatch += 1

    return total, match, mismatch


def get_neg_pairs(disconnected_docs):
    diff_doc_pairs = []

    for i, docs1 in enumerate(disconnected_docs):
        for j in range(i+1, len(disconnected_docs)):
            docs2 = disconnected_docs[j]
            for d1 in docs1:
                for d2 in docs2:
                    diff_doc_pairs.append((d1, d2))


    return diff_doc_pairs


def parallelize(eval_pairs, pred_doc_df, pred_label_name):

    num_partitions = max(len(eval_pairs)//30, 1)

    splits = list(zip(np.array_split(eval_pairs, num_partitions), itertools.repeat(pred_doc_df), \
                                                         itertools.repeat(pred_label_name)))

    with ProcessPoolExecutor() as executor:
        mapped_res_gen = executor.map(test_doc_pairs, splits)

        result = functools.reduce(
            lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]),
            mapped_res_gen)

    return result

def get_unassigned_scores(df, te_obj, label_name, outlier_thresh):
    total_docs = len(df)

    unassigned_df = df[df[label_name].map(len)==0]
    outlier_predicted = len(unassigned_df)

    absent_features = 0
    true_outlier_docs = 0
    false_positives = 0

    for doc_id, row in unassigned_df.iterrows():

        if len(row[label_name]) == 0:
            absent_features += 1

        problem = te_obj.problem_tickets_df.loc[doc_id].subject_pr
        docs_count = len(te_obj.problem_ticket_distr[problem])

        if docs_count <= outlier_thresh:
            true_outlier_docs += 1
        else:
            false_positives += 1

    absent_feature_score = round(absent_features * 100 / total_docs, 4)
    outlier_predicted_score = round(outlier_predicted * 100 / total_docs, 4)
    tpr = round(true_outlier_docs * 100 / outlier_predicted, 4)
    fpr = round(false_positives * 100 / outlier_predicted, 4)

    metrics_map = {'absent_features':absent_feature_score, 'outlier_predicted': outlier_predicted_score, \
                   'outlier_precision': tpr, 'outlier_fpr': fpr}

    return metrics_map

def evaluate_unassigned_docs(cluster_obj, te_obj):
    clean_docs = cluster_obj.cluster_processed_df[~cluster_obj.cluster_processed_df.isNoisy]

    total_docs = len(clean_docs)

    ### TMP assignment ***
    outlier_thresh = cluster_obj.outlier_thresh
    print('outlier_threshold :', outlier_thresh)

    labels_metrics_map = get_unassigned_scores(clean_docs, te_obj, 'labels', outlier_thresh)

    sel_labels_metrics_map = get_unassigned_scores(clean_docs, te_obj, 'selected_labels', outlier_thresh)

    metrics_map = {}

    for metric_map, suffix in zip(list([labels_metrics_map, sel_labels_metrics_map]), list(['@labels', '@selected_labels'])):
        for name, value in metric_map.items():
            metrics_map[name + suffix] = value

    return metrics_map


def ticket_eval_mlflowrun(cfi_obj, sample_size=10000):
    cluster_df = cfi_obj.cluster_processed_df

    te = TicketsEvaluator()
    pos_pairs, neg_pairs = te.get_pos_neg_samples(sample_size)

    l_params_map, l_metrics_map = te.evaluate(pos_pairs, neg_pairs, cluster_df, label_name='labels')

    sel_params_map, sel_metrics_map = te.evaluate(pos_pairs, neg_pairs, cluster_df, label_name='selected_labels')

    outlier_metrics_map = evaluate_unassigned_docs(cfi_obj, te)

    params_map = {**l_params_map, **sel_params_map}
    metrics_map = {**l_metrics_map, **sel_metrics_map, **outlier_metrics_map}


    return params_map, metrics_map