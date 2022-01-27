from sklearn.metrics.pairwise import euclidean_distances

import pandas as pd
import numpy as np

def neighborhood_preservation_trustworthiness(k, inpt, som):

    # number of neighbours, k, should be < data size // 2 
    n = inpt.shape[0]
    assert k < (n / 2), 'k must be < N/2.'
    
    #d = euclidean_distances(inpt, som) # calculating euclidean distances between input and som
    #projections = som[np.argmin(d, axis=1)] # vector with smallest distance for each observation
    
    d_data = euclidean_distances(inpt) + np.diag(np.inf * np.ones(n))        # calculating distances on the input
    d_projections = euclidean_distances(som) + np.diag(np.inf * np.ones(n))  # calculating distances on the som
    
    original_ranks = pd.DataFrame(d_data).rank(method='min', axis=1)         # computing ranks from lowest in group to max
    projected_ranks = pd.DataFrame(d_projections).rank(method='min', axis=1) # computing ranks from lowest in group to max
    
    weights = (projected_ranks <= k).sum(axis=1) / (original_ranks <= k).sum(axis=1)  # weight k-NN ties
    
    # comparing both neighbourhoods
    nps = np.zeros(n)
    trs = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if (i != j) and (original_ranks.iloc[i, j] <= k) and (projected_ranks.iloc[i, j] > k):
                nps[i] += (projected_ranks.iloc[i, j] - k) * weights[i]
            elif (i != j) and (original_ranks.iloc[i, j] > k) and (projected_ranks.iloc[i, j] <= k):
                trs[i] += (original_ranks.iloc[i, j] - k) / weights[i]
    npr = 1.0 - 2.0 / (n * k * (2*n - 3*k - 1)) * np.sum(nps)
    tr = 1.0 - 2.0 / (n * k * (2*n - 3*k - 1)) * np.sum(trs)
    
    return npr, tr


def