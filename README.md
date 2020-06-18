# hierarchical-labelled-clustering-evaluation

Evaluation codes and result to test labelling and clustering scores obtained from **Embedding based closed frequent itemset hierarchical clustering** technique.

What to expect -
1. hierarchical clusters for documents with labels

2. suggested **outlier** documents (as per current pass) and accommodate them to consider for main clusters in next iteration of new documents.

3. Evaluation setup for Reuters and ticket documents and logged using mlflow. Scoring metrics used to choose best parameters are -
    1. cluster assignment score
    2. label assignment score 
    3. precision @cluster
    4. recall @cluster
    5. fscore @cluster
    6. accuracy @cluster
    7. predicted outlier %
    8. precision @outlier
    9. FPR @outlier
    10. Number of disconnected clusters
    
