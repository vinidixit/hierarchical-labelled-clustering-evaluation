import networkx as nx
from reuters_utils import *
import pickle
import random
import clustering.hierarchical_emb_clustering
#from clustering.hierarchical_emb_clustering import CFIEClustering

class ReutersEvaluator:


    def __init__(self, reuters_df, topics_G, industry_cat_G):

        self.reuters_df = reuters_df
        self.reuter_topic_G = nx.compose(topics_G, industry_cat_G)
        self.reuters_df.categories = self.reuters_df.categories.apply(lambda x: [c for c in x if is_valid_category(c)])

        reuters_categories = set(itertools.chain(*self.reuters_df.categories))
        self.category_doc_map = self.__get_category_doc_distribution(reuters_categories)

    def __get_category_doc_distribution(self, reuters_categories):
        # existing category distribution
        categ_doc_map = {}
        for c in reuters_categories:
            #print(c)
            #print(self.reuters_df[self.reuters_df.categories.apply(lambda x: c in x)].index.values)
            categ_doc_map[c] = set(self.reuters_df[self.reuters_df.categories.apply(lambda x: c in x)].index.values)

        return categ_doc_map


    def evaluation_setup(self, cluster_obj, sample_size, label_name):
        cluster_df = cluster_obj.cluster_processed_df

        label_doc_G = get_label_doc_graph(cluster_obj._closed_fi_graph, cluster_df, label_name)
        disconnected_docs_by_G = get_disconnected_docs(label_doc_G, cluster_df)[0]

        print(label_name, ': disconnected_docs_by_G: ', len(disconnected_docs_by_G))

        all_pos_pairs = get_pos_pairs(disconnected_docs_by_G)
        all_neg_pairs = get_neg_pairs(disconnected_docs_by_G)

        pos_pairs, neg_pairs = get_pos_neg_samples(sample_size, all_pos_pairs, all_neg_pairs)

        return len(disconnected_docs_by_G), pos_pairs, neg_pairs

    def evaluate(self, cluster_obj, sample_size = 100, label_name='labels'):

        disconnected_cluster_count, pos_pairs, neg_pairs =  self.evaluation_setup(cluster_obj, sample_size, label_name)

        param_names = ['total_pairs', 'pos_pairs', 'neg_pairs']
        metric_names = ['disconnected_clusters', 'precison', 'recall', 'tnr', 'fscore', 'accuracy']

        param_values, metric_values = _get_evaluation_scores(pos_pairs, neg_pairs, self.reuters_df)
        metric_values = [disconnected_cluster_count] + list(metric_values)

        eval_suffix = "@" + label_name

        params_map = {}
        for name, value in zip(param_names, param_values):
            params_map[name + eval_suffix] = value

        metrics_map = {}
        for name, value in zip(metric_names, metric_values):
            metrics_map[name + eval_suffix] = value

        return params_map, metrics_map

# def get_b_cubed_scores(re_obj, cluster_obj, label_name)
#     cluster_df = cluster_obj.cluster_processed_df
#
#     label_doc_G = get_label_doc_graph(cluster_obj._closed_fi_graph, cluster_df, label_name)
#     disconnected_docs_by_G, disconnected_labels_by_G = get_disconnected_docs(label_doc_G, cluster_df)
#


def _get_evaluation_scores(pos_pairs, neg_pairs, cl_df, label_name='categories'):

    total_assigned_pos, true_positives, false_negatives = parallelize(pos_pairs, cl_df, label_name,
                                                                      remove_labels=[])  # 'CCAT','ECAT'

    total_assigned_neg, false_positives, true_negatives = parallelize(neg_pairs, cl_df, label_name,
                                                                      remove_labels=['CCAT', 'ECAT'])  # 'CCAT','ECAT'

    tnr = round(true_negatives * 100 / (true_negatives + false_positives), 4)

    accuracy = round((true_positives + true_negatives) * 100 / (total_assigned_pos + total_assigned_neg), 4)

    print('total_assigned_pos, true_positives, false_negatives :', total_assigned_pos, true_positives,
          false_negatives)
    print('total_assigned_neg, false_positives, true_negatives :', total_assigned_neg, false_positives,
          true_negatives)

    precision = round(true_positives * 100 / (true_positives + false_positives), 4)
    recall = round(true_positives * 100 / (true_positives + false_negatives), 4)
    fscore = round(2 * precision * recall / (precision + recall), 4)

    total_pairs = total_assigned_pos + total_assigned_neg

    print('classwise accuracy :', 100 * true_positives / total_assigned_pos,
          100 * true_negatives / total_assigned_neg)
    return (total_pairs, total_assigned_pos, total_assigned_neg), (precision, recall, tnr, fscore, accuracy)

def get_unassigned_scores(df, re_obj, label_name, outlier_thresh):
    outlier_thresh = max(outlier_thresh, 10)

    total_docs = len(df)

    unassigned_df = df[df[label_name].map(len)==0]
    outlier_predicted = len(unassigned_df)

    absent_features = 0
    true_outlier_docs = 0
    false_positives = 0

    for doc_id, row in unassigned_df.iterrows():

        if len(row[label_name]) == 0:
            absent_features += 1

        categs = re_obj.reuters_df.loc[doc_id].categories
        docs_count = [len(re_obj.category_doc_map[c]) for c in categs]

        outlier_cands = [(cat, count) for cat, count in zip(categs, docs_count) if
                         count < outlier_thresh]

        #for cat in ['CCAT','ECAT']:
        #    if cat in outlier_cands:


        if outlier_cands:
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


def evaluate_unassigned_docs(cluster_obj, re_obj):
    clean_docs = cluster_obj.cluster_processed_df[~cluster_obj.cluster_processed_df.isNoisy]

    ### TMP assignment ***
    outlier_thresh = cluster_obj.outlier_thresh
    labels_metrics_map = get_unassigned_scores(clean_docs, re_obj, 'labels', outlier_thresh)

    sel_labels_metrics_map = get_unassigned_scores(clean_docs, re_obj, 'selected_labels', outlier_thresh)

    metrics_map = {}

    for metric_map, suffix in zip(list([labels_metrics_map, sel_labels_metrics_map]), list(['@labels', '@selected_labels'])):
        for name, value in metric_map.items():
            metrics_map[name + suffix] = value

    return metrics_map


def prepare_reuters_obj(reuters_dir):
    df_file = reuters_dir + '19961119_selected_df.pkl'
    topics_file = reuters_dir + 'reuters_topics_G.pkl'
    industry_file = reuters_dir + 'reuters_industry_cat_G.pkl'

    reuters_df = pickle.load(open(df_file, 'rb'))
    topics_G = pickle.load(open(topics_file, 'rb'))
    industry_cat_G = pickle.load(open(industry_file, 'rb'))

    re_obj = ReutersEvaluator(reuters_df, topics_G, industry_cat_G)
    return re_obj


def re_eval_mlflowrun(reuters_dir, cluster_obj, sample_size=10000):

    re_obj = prepare_reuters_obj(reuters_dir)
    cluster_df = cluster_obj.cluster_processed_df

    l_params_map, l_metrics_map = re_obj.evaluate(cluster_obj, sample_size, label_name='labels')

    sel_params_map, sel_metrics_map = re_obj.evaluate(cluster_obj, sample_size, label_name='selected_labels')


    #eval_params_map, cluster_metrics_map = evaluate_assigned_clusters(cluster_obj, re_obj)
    outlier_metrics_map = evaluate_unassigned_docs(cluster_obj, re_obj)

    params_map = {**l_params_map, **sel_params_map}
    metrics_map = {**l_metrics_map, **sel_metrics_map, **outlier_metrics_map}


    return params_map, metrics_map

if __name__=='__main__':
    data_dir = '../../sample_data/'
    reuters_dir = data_dir + 'reuters_selected/'

    categ_file = reuters_dir + 'selected_categs.pkl'
    df_file = reuters_dir + '19961119_selected_df.pkl'
    topics_file = reuters_dir + 'reuters_topics_G.pkl'
    industry_file = reuters_dir + 'reuters_industry_cat_G.pkl'

    reuters_df = pickle.load(open(df_file, 'rb'))
    topics_G = pickle.load(open(topics_file, 'rb'))
    industry_cat_G = pickle.load(open(industry_file, 'rb'))
    reuters_categories = pickle.load(open(categ_file, 'rb'))

    re_obj = ReutersEvaluator(reuters_df, topics_G, industry_cat_G)

    cfi_file = '../mlruns/0/2871e7e72da4446fb4de935b655d1bb9/artifacts/cfi_obj_dir/cfi_obj.pkl'

    mlrun = '82f0518419cc4a67ba9ebd6846d68c8c'

    cfi_file = '../mlruns/0/' + mlrun + '/artifacts/cfi_obj_dir/cfi_obj.pkl'

    cfi_file = '../test/cfi_obj.pkl'
    cfi_obj = pickle.load(open(cfi_file, 'rb'))

    #re_obj = prepare_reuters_obj(reuters_dir)
    #print('GCAT' in re_obj.category_doc_map.keys())


    p, m = re_eval_mlflowrun(reuters_dir, cfi_obj) #re_obj.evaluate(cfi_obj)

    for k, v in p.items():
        print(k, v)

    print('\n')

    for k, v in m.items():
        print(k, v)
