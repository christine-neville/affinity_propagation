"""Runs affinity propagation using multiple preference parameters. Outputs csv file 
and png plot of H(S) and H(K) values for different preference parameters.

Syntax: python affinity_param_testing.py <score matrix file in csv format> <interval value for preference parameter>"""

import sys
import numpy as np
from sklearn.cluster import AffinityPropagation
from scipy.stats import entropy
import matplotlib.pyplot as plt

score_file = sys.argv[1]
pref_iter = float(sys.argv[2])
score_matrix1 = np.genfromtxt(score_file,delimiter=",")
score_matrix = score_matrix1 + score_matrix1.T - np.diag(np.diagonal(score_matrix1))
info_out = score_file[:-17]+"_prefvals.csv"
plot_name = score_file[:-17]+"_prefvals.png"

stats = []
for p in np.arange(np.amin(score_matrix),np.amax(score_matrix),pref_iter):
    clustering = AffinityPropagation(damping = 0.5, preference=p, random_state=2, affinity="precomputed").fit(score_matrix)
    num_clusters = len(clustering.cluster_centers_indices_)
    cluster_ids = range(num_clusters)
    labels = list(clustering.labels_)
    counts = []
    for cid in cluster_ids:
        count = labels.count(cid)
        counts.append(count)
    freqs1 = [c/num_clusters for c in counts]
    spc = []
    clust_clust = []
    checked = []
    for c in counts:
        if c in checked:
            pass
        else:
            clust_clust.append(c)
            spc.append(counts.count(c))
            checked.append(c)
    freqs2 = [s/len(clust_clust) for s in spc]
    HS = entropy(freqs1, base=2)
    HK = entropy(freqs2, base=2)
    info = [p, HS, HK, num_clusters]
    stats.append(info)

#Save image of HS vs HK
HSs = [i[1] for i in stats]
HKs = [i[2] for i in stats]
plt.scatter(HSs, HKs)
plt.xlabel("HS")
plt.ylabel("HK")
plt.savefig(plot_name)

#Save HS, HK, and number-of-clusters data for each preference value
stats = np.array(stats)
np.savetxt(info_out,stats,delimiter=",")
