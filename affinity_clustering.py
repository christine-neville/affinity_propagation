"""Runs affinity propagation clustering on an input score matrix. Fasta file with matching name must be in same directory.
Outputs a png demonstrating cluster sizes and a text file of clusters.
Syntax: python affinity_clustering.py <score_matrix_file> <preference value>"""

from sklearn.cluster import AffinityPropagation
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

score_matrix = np.genfromtxt(sys.argv[1],delimiter=",")
p = float(sys.argv[2])
fname = sys.argv[1][:-17]

#Build clusterer
clustering = AffinityPropagation(damping = 0.5, random_state=2, preference=p, affinity="precomputed", max_iter=1000, convergence_iter=20).fit(score_matrix)

#Make list of cluster id's and corresponding list of counts in each cluster
num_clusters = len(clustering.cluster_centers_indices_)
cluster_ids = range(num_clusters)
labels = list(clustering.labels_)
counts = []
for cid in cluster_ids:
    count = labels.count(cid)
    counts.append(count)

#Plot cluster sizes
plt.bar(cluster_ids,counts)
plt.savefig(fname+"_cluster-sizes.png")

##Organize clustering results
labels = clustering.labels_
ffile = open(fname+".fa").read()
entries = ffile.split(">")[1:]
accs = []
# for e in entries:
#     e = e.split("|")
#     accs.append(e[2].split()[0])
for e in entries:
    e = e.split("\n")
    accs.append(e[0])

##Make dictionary with seqid's as keys and cluster labels as values
#clusterdict = {}
#for x,y in zip(accs,labels):
#    clusterdict[x] = y

##Make list of accession numbers for cluster centers
centers = []
for j in clustering.cluster_centers_indices_:
    centers.append(accs[j])

##Make dictionary with cluster labels as keys and lists of seqid's as values
dict2 = {}
for x,y in zip(accs,labels):
    if y not in dict2:
        dict2[y] = [x]
    else:
        dict2[y].append(x)

outfile = open(fname+"_affprop_results.txt",'w')
for x,y in zip(range(len(dict2)),dict2.keys()):
    outfile.write(">Cluster %d:\n"%x)
    for name in dict2[y]:
        outfile.write(name+"\n")
    outfile.write("\n")
outfile.close()
