import numpy as np
import pandas as pd
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import random
import sys

"""Returns CV score, given a classifer and set of labeled data points"""

def cross_val_score_solution(clf, gene_exp, tissue_type, cv=5):
    
    # randomize input arrays
    c = list(zip(gene_exp, tissue_type))
    random.shuffle(c)

    gene_exp_s = [i[0] for i in c] 
    tissue_type_s = [i[1] for i in c]

    # split to arrays according to cv
    gene_exp = np.asarray(np.array_split(gene_exp_s, cv, axis=0))
    tissue_type = np.asarray(np.array_split(tissue_type_s, cv, axis=0)) 
     
    scores = []
    count = 0
    # fit training sections 
    for i in range(len(gene_exp)):
        gene_train = gene_exp[np.arange(len(gene_exp))!=count]
        gene_train = np.concatenate(gene_train)
        
        tissue_train = tissue_type[np.arange(len(tissue_type))!=count]
        tissue_train = np.concatenate(tissue_train).ravel()
        
    
        clf.fit(gene_train, tissue_train)
        scores.append(clf.score(gene_exp[count], tissue_type[count]))
        count += 1
        
    scores = np.asarray(scores)
    
    return scores.mean()

n_neighbors = 15
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

gene_exp = pd.read_csv("gene_expr.csv", header=None).values
tissue_type = pd.read_csv("gene_labels.csv", header=None).values
gene_exp = np.transpose(gene_exp)
tissue_type = np.ravel(tissue_type)


mean_accuracy = cross_val_score_solution(clf, gene_exp, tissue_type, cv = int(sys.argv[1]))
print("CV accuracy: %0.2f" % (mean_accuracy))