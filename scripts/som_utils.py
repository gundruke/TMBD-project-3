from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import patches as patches
import matplotlib.lines as mlines

import scipy.integrate


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



def predict(df, som):
    bmu, bmu_idx = som.find_bmu(df.values)
    df['bmu'] = bmu
    df['bmu_idx'] = bmu_idx
    return df



def som_train_predict(som, trunc_data, agri_data, num_epochs=200, init_learning_rate=0.01):
    som.train(trunc_data.values,num_epochs,init_learning_rate)
    clustered_df = trunc_data.apply(predict, args=(som, ), axis=1)
    joined_df = agri_data.join(clustered_df, rsuffix="_norm")
    
    return joined_df, clustered_df



def visualize_som(som, df):
    fig = plt.figure()
    # setup axes
    ax = fig.add_subplot(111)
    scale = 50
    ax.set_xlim((0, som.net.shape[0]*scale))
    ax.set_ylim((0, som.net.shape[1]*scale))
    ax.set_title("Cash Crops Clustering by using SOM")
    
    for x in range(0, som.net.shape[0]):
        for y in range(0, som.net.shape[1]):
            ax.add_patch(patches.Rectangle((x*scale, y*scale), scale, scale,
                                           facecolor='white',
                                           edgecolor='grey'))
            
    legend_map = {}
    
    for index, row in df.iterrows():
        x_cor = row['bmu_idx'][0] * scale
        y_cor = row['bmu_idx'][1] * scale
        x_cor = np.random.randint(x_cor, x_cor + scale)
        y_cor = np.random.randint(y_cor, y_cor + scale)
        color = row['bmu'][0]
        marker = "$\\ " + row['Crop'][0]+"$"
        marker = marker.lower()
        ax.plot(x_cor, y_cor, color=color, marker=marker, markersize=10)
        label = row['Crop']
        if not label in legend_map:
            legend_map[label] =  mlines.Line2D([], [], color='black', 
                                               marker=marker, linestyle='None',
                                               markersize=10, label=label)
    plt.legend(handles=list(legend_map.values()), bbox_to_anchor=(1, 1))
    plt.show()
    
    
def quantization_error_test(orig, som):
    return (np.sum(np.abs(orig.values - som.values)))/orig.shape[0]


def som_abs_error(som, trunc_data, num_epochs=200, init_learning_rate=0.01):

    prim_h, weight = som.train(trunc_data.values,num_epochs,init_learning_rate)
    return max(abs(prim_h - weight)[0])
