# Hierarchical Labelled Clustering Evaluation

Evaluation codes and results to test labelling and clustering scores obtained from [**Embedding based Generalized Closed Frequent Itemset Hierarchical Clustering**](https://github.com/vinidixit/hierarchical-labelled-clustering) technique.
Research paper link https://dl.acm.org/doi/10.1145/3493700.3493727

### Data for evaluation
Evaluations are done on a publically available Reuters [dataset](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm) and a private tickets dataset.

### Flow chart of the working system
This repository works at 3rd component level (highlighted in yellow)for evaluating our GCFI++ model.

![FLow chart](images/flow_chart.png)
 
 
**What to expect -**
1. Hierarchical clusters for documents with overlapping labels. Technique is most suited for short text documents (sentences/paragraphs) with overlapping concepts. 

2. Its **iterative** setup to extract pair of clusters and outliers for *current pass* of documents. Clusters contain documents, those have strong associativity with respective cluster topics, whereas outliers showed weak features and associativity with mainstream clusters.

3. Suggesting **outlier** documents (as per current pass) and accommodating them to consider for main clusters in the next iteration with new documents.

4. Parameters used in the various steps of clustering pipeline for building experiments are -
    1. **Feature Processing**
        1. phrase extraction pass count
        2. Maximum features fraction
        3. Embedding type  (None / [Fasttext](https://fasttext.cc/) / [Bert](https://github.com/UKPLab/sentence-transformers))
        4. Number of passes to form embedding clusters
        
    2. **Clustering**
        1. number of passes and support threshold fractions for each pass
        2. corpus type  ("short"/"long")
        3. maximum overlapping labels (fraction)
        4. keep negative label scores (boolean)
    
    3. **Evaluation**
        1. True label dataset
        2. outlier threshold
        3. Number of positive pairs
        4. Number of negative pairs
        
5. Evaluation setup for [Reuters](http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/lyrl2004_rcv1v2_README.htm) and ticket (private) documents and logged using [MLFlow](https://mlflow.org/). Scoring metrics used to choose best combination of parameters are -
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
 

