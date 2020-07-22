import os
from feature_extraction import fe_mlflowrun
from feature_selection import fs_mlflowrun
from hierarchical_emb_clustering import cl_mlflowrun
from reuters_evaluator import re_eval_mlflowrun

import mlflow

def workflow(corpus_name, corpus_type, max_dup, embedding, num_embedding_passes):
    
    _already_ran = mlflow.active_run() is not None

    if not _already_ran:
        mlflow.start_run()

    fe_params_map, fe_metrics_map, fe_artifacts_map, feature_extracted_df = fe_mlflowrun(text_file)

    fs_params_map, fs_metrics_map, fs_artifacts_map, feature_selected_df = fs_mlflowrun(feature_extracted_df, \
                                                                                        max_features_frac, embedding, \
                                                                                        num_embedding_passes)

    cl_params_map, cl_metrics_map, cl_artifacts_map, cfi_obj = cl_mlflowrun(feature_selected_df, corpus_type, max_dup,
                                                                                            without_negative_label)

    eval_params_map, eval_metrics_map = re_eval_mlflowrun(reuters_dir, cfi_obj)

    params_maps = [{'corpus_name':corpus_name}, fe_params_map, fs_params_map, cl_params_map, eval_params_map]
    metrics_maps = [fs_metrics_map, fe_metrics_map, cl_metrics_map, eval_metrics_map]
    artifacts_maps = [fs_artifacts_map, fe_artifacts_map, cl_artifacts_map]

    for param_map in params_maps:
        for k, v in param_map.items():
            k = k.replace('@', '/')
            k = k.replace('#', '-')
            mlflow.log_param(k, v)

    for metric_map in metrics_maps:
        for k, v in metric_map.items():
            k = k.replace('@', '/')
            k = k.replace('#', '-')
            if v:
                mlflow.log_metric(k, v)
            else:
                mlflow.log_metric(k, -1)

    for artifact_map in artifacts_maps:
        for k, v in artifact_map.items():
            mlflow.log_artifact(v, k)


    if not _already_ran:
        mlflow.end_run()



if __name__ == '__main__':
    
    # Avaliable hyperparamters for each step
    corpus_type_ops = ['short', 'long']
    lemmatization_ops = ['default', 'ticket']
    label_ops = [False, True]
    embedding_ops = [None, 'fasttext', 'sentbert']

    max_dup_ops = [None, 1, 2, 3]
    max_features_frac_ops = [None, 0.1, 0.2]
    
    ## data parameter setup for Reuters/Tickets datasets
    data_dir = '../sample_data/'
    tickets_dir = data_dir + 'tickets_selected/'
    reuters_dir = data_dir + 'reuters_selected/'
    text_file = reuters_dir + '19961119_selected_df.pkl' #tickets_dir + 'subjects_df.pkl'
    
    ## current parameters chosen for an experiment
    corpus_name = 'Reuters-text'
    corpus_type = 'long'
    max_dup = 2
    without_negative_label = True
    max_features_frac = .20
    embedding = 'fasttext'
    num_embedding_passes = None
    
    workflow(corpus_name, corpus_type, max_dup, embedding, num_embedding_passes)
