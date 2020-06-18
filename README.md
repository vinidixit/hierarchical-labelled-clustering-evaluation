# hierarchical-labelled-clustering-evaluation

Evaluation codes and result to test labelling and clustering scores obtained from **Embedding based closed frequent itemset hierarchical clustering** technique.

What to expect -
1. hierarchical clusters for documents with labels

2. suggested **outlier** documents (as per current pass) and accommodate them to consider for main clusters in next iteration of new documents.

3. Evaluation setup for Reuters and ticket documents and logged using mlflow. Scoring metrics used to choose best parameters are -
    i cluster assignment score
    ii label assignment score 
    ii precision @cluster
    iii recall @cluster
    iv fscore @cluster
    v accuracy @cluster
    vi predicted outlier %
    vii precision @outlier
    viii FPR @outlier
    
