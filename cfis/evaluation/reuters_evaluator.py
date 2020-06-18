from hierarchical_emb_clustering import CFIEClustering
import pickle
import numpy as np
import networkx as nx
import pandas as pd
from reuters_utils import is_valid_category, get_label_doc_graph, get_disconnected_docs, parallelize, parallelize_intercluster
import itertools
import time

class ReutersEvaluator:

    def __init__(self, reuters_df, topics_G, industry_cat_G, reuters_categories):
        self.reuters_df = reuters_df
        self.reuter_topic_G = nx.compose(topics_G, industry_cat_G)
        self.reuters_df.categories = self.reuters_df.categories.apply(lambda x: [c for c in x if is_valid_category(c)])

        self.category_doc_map = self.__get_category_doc_distribution(reuters_categories)

    def __get_category_doc_distribution(self, reuters_categories):
        # existing category distribution
        categ_doc_map = {}
        for c in reuters_categories:
            #print(c)
            #print(self.reuters_df[self.reuters_df.categories.apply(lambda x: c in x)].index.values)
            categ_doc_map[c] = set(self.reuters_df[self.reuters_df.categories.apply(lambda x: c in x)].index.values)

        return categ_doc_map

    def evaluate(self, source_label_G, source_doc_df, source_label_name, pred_df, pred_label_name):

        param_names = ['total_pairs', 'pos_pairs', 'neg_pairs']
        metric_names = ['disconnected_clusters', 'precison', 'recall', 'tnr', 'fscore', 'accuracy']


        param_values, metric_values = self.__get_evaluation_scores(source_label_G, source_doc_df, source_label_name, \
                                                                                        pred_df, pred_label_name)

        eval_suffix = "@"+ source_label_name + '#' + pred_label_name

        params_map = {}
        for name, value in zip(param_names, param_values):
            params_map[name+eval_suffix] = value

        metrics_map = {}
        for name, value in zip(metric_names, metric_values):
            metrics_map[name + eval_suffix] = value

        return params_map, metrics_map


    def __get_evaluation_scores(self, cluster_G, doc_df, cluster_label_name, pred_df, pred_label_name):

        print('\n\nGetting clusters by : ', cluster_label_name)
        label_doc_G = get_label_doc_graph(cluster_G, doc_df, cluster_label_name)
        labels_disconnected_docs, _ = get_disconnected_docs(label_doc_G, doc_df)
        print('Disconnected clusters counts: ', len(labels_disconnected_docs),
                                                                        [len(c) for c in labels_disconnected_docs])

        print('\n\nEvaluating same clusters by matching: ', pred_label_name)
        total_same_pairs, true_positives, false_negatives = 0, 0, 0

        for connected_docs in labels_disconnected_docs:
            if len(connected_docs) < 2:
                continue

            t0 = time.time()
            total, tp, fn = parallelize(connected_docs, pred_label_name, pred_df, remove_labels=[])
            print('Evaluator same cluster: {:.3f} sec'.format(time.time()-t0))

            total_same_pairs += total
            true_positives += tp
            false_negatives += fn

        print('\n\nEvaluating Different clusters by matching: ', pred_label_name)
        docs1, docs2 = labels_disconnected_docs[0], list(itertools.chain(*labels_disconnected_docs[1:]))

        t0 = time.time()
        total_diff_pairs, false_positives, true_negatives = parallelize_intercluster(docs1, docs2, pred_label_name, pred_df)
        print('Evaluator diff cluster: {:.3f} sec'.format(time.time() - t0))

        total_pairs = total_same_pairs+total_diff_pairs

        ### calculate prediction scores

        precision = get_score(true_positives, false_positives)
        recall = get_score(true_positives, false_negatives)
        tnr = get_score(true_negatives, false_positives)
        fscore = get_fscore(precision, recall)
        accuracy = get_accuracy(true_positives, true_negatives, total_pairs)

        return (total_pairs, total_same_pairs, total_diff_pairs), \
               (len(labels_disconnected_docs), precision, recall, tnr, fscore, accuracy)


def get_score(true_count, false_count):
    if true_count==0 and false_count==0:
        return None

    return round(true_count*100/(true_count+false_count), 4)

def get_fscore(precision, recall):
    if precision is None or recall is None or precision==0 or recall==0:
        return None

    return round(2*precision*recall/(precision+recall), 4)

def get_accuracy(true_positives, true_negative, total_pairs):
    if total_pairs == 0:
        return None

    return round((true_positives+true_negative)*100/total_pairs, 4)


def evaluate_assigned_clusters(cluster_obj, re_obj):
    t0 = time.time()

    label_G = cluster_obj._closed_fi_graph
    cluster_doc_df = cluster_obj.cluster_processed_df

    cluster_doc_df.selected_labels = cluster_doc_df.selected_labels.apply(lambda x: list(np.array(x)[:,0]) if x else [])
    cluster_doc_df.labels = cluster_doc_df.labels.apply(lambda x: list(np.array(x)[:,0]) if x else [])

    # Cluster evaluation
    print('\n', '='*10, 'Evaluating predicted label clusters..', '='*10)
    l_params_map, l_metrics_map = re_obj.evaluate(label_G, cluster_doc_df, 'labels', re_obj.reuters_df, 'categories')
    sel_params_map, sel_metrics_map = re_obj.evaluate(label_G, cluster_doc_df, 'selected_labels', re_obj.reuters_df, \
                                                                                                    'categories')
    
    ## Evaluate Reuter clusters by their Categories
    print('\n\n\n','='*10, 'Evaluating Reuters clusters by categories..', '='*10)
    #assigned_clusters_df = cluster_doc_df[cluster_doc_df.labels.map(len)>0]
    cat_l_params_map, cat_l_metrics_map = re_obj.evaluate(re_obj.reuter_topic_G, re_obj.reuters_df, 'categories', \
                                                          cluster_doc_df, 'labels')

    #assigned_clusters_df = cluster_doc_df[cluster_doc_df.selected_labels.map(len) > 0]
    cat_sel_params_map, cat_sel_metrics_map = re_obj.evaluate(re_obj.reuter_topic_G, re_obj.reuters_df, 'categories', \
                                                              cluster_doc_df, 'selected_labels')


    params_map = {**l_params_map, **sel_params_map, **cat_l_params_map, **cat_sel_params_map}
    metrics_map = {**l_metrics_map, **sel_metrics_map, **cat_l_metrics_map, **cat_sel_metrics_map}

    print('{:.3f} mins.'.format((time.time()-t0)/60))
    return params_map, metrics_map


def get_unassigned_scores(df, re_obj, label_name, outlier_thresh):
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

    total_docs = len(clean_docs)

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
    categ_file = reuters_dir + 'selected_categs.pkl'

    reuters_df = pickle.load(open(df_file, 'rb'))
    topics_G = pickle.load(open(topics_file, 'rb'))
    industry_cat_G = pickle.load(open(industry_file, 'rb'))
    reuters_categories = pickle.load(open(categ_file, 'rb'))

    re_obj = ReutersEvaluator(reuters_df, topics_G, industry_cat_G, reuters_categories)
    return re_obj


def re_eval_mlflowrun(reuters_dir, cluster_obj):

    re_obj = prepare_reuters_obj(reuters_dir)
    eval_params_map, cluster_metrics_map = evaluate_assigned_clusters(cluster_obj, re_obj)
    outlier_metrics_map = evaluate_unassigned_docs(cluster_obj, re_obj)

    return eval_params_map, {**cluster_metrics_map, **outlier_metrics_map}


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

    re_obj = ReutersEvaluator(reuters_df, topics_G, industry_cat_G, reuters_categories)

    """
    corpus_name = 'Reuters-text'

    cluster_dir = data_dir + 'clustering_results/' + corpus_name + '/'
    cluster_obj_file = cluster_dir + 'cfi_emb.pkl'

    #eval_dir = data_dir + 'evaluation_results/re_text/dim0.1/'

    cluster_properties = {'Dimension':0.1, 'FeaturesSize':195, 'Embedding':'Fasttext', 'NegativeLabels':False}
    """

    cfi_sb_emb_obj = pickle.load(open('../../sample_data/clustering_results/Reuters-headline/cluster_sb_emb_obj.pkl', 'rb'))
    #evaluate_unassigned_docs(cfi_sb_emb_obj, re_obj)

    #print(cfi_sb_emb_obj.cluster_processed_df.head(1)[['labels', 'selected_labels']])

    cfi_file = '../mlruns/0/a33df08dd917443097dea4f5d8365755/artifacts/cfi_obj_dir/cfi_obj.pkl'
    cfi_obj = pickle.load(open(cfi_file, 'rb'))
    doc_df = cfi_obj.cluster_processed_df
    label_doc_G = get_label_doc_graph(cfi_obj._closed_fi_graph, doc_df, 'selected_labels')
    labels_disconnected_docs, _ = get_disconnected_docs(label_doc_G, doc_df)

    print(len(labels_disconnected_docs))

    evaluate_assigned_clusters(cfi_obj, re_obj)
    """
    max_dup = 2
    max_labels_per_doc = math.ceil(max_dup * len(cfi_sb_emb_obj._clusters) / 100) if max_dup else None
    print(max_dup, max_labels_per_doc)

    illegal_docs_assigns = 0
    for doc_id, row in cfi_sb_emb_obj.cluster_processed_df.iterrows():
        if len(row.labels)> max_labels_per_doc:
            illegal_docs_assigns += 1
            print(row, '\n')

    print('illegal_docs_assigns: ', illegal_docs_assigns)
    """

    """
    print(reuters_df.index)
    print(cfi_sb_emb_obj.cluster_processed_df.index)

    params_map, metrics_map = re_eval_mlflowrun(reuters_dir, cfi_sb_emb_obj) #evaluate_unassigned_docs(cfi_sb_emb_obj, re_obj)
    print(params_map)
    print(metrics_map)
    """

    """
    cluster_eval_df, category_eval_df = process_evaluation(cfi_sb_emb_obj, re_obj)

    print("\ncluster scores..")
    eval_df = get_eval_scores(cluster_eval_df)
    for k, v in eval_df.iteritems():
        print(k, ':', v.values)

    print("\n\ncategory scores..")
    eval_df = get_eval_scores(category_eval_df)
    for k, v in eval_df.iteritems():
        print(k, ':', v.values)

    """
    """
    category_eval_df = pickle.load(open(eval_dir + 'category_eval_scores.pkl', 'rb'))
    cluster_eval_df = pickle.load(open(eval_dir + 'cluster_eval_scores_df.pkl', 'rb'))

    cluster_obj = pickle.load(open(cluster_obj_file, 'rb'))
    run_mlflow('Reuters-text', cluster_obj, cluster_properties, cluster_eval_df)
    run_mlflow('Reuters-text', cluster_obj, cluster_properties, category_eval_df)
    """

