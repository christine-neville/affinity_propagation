"""Script to generate a numpy-format score matrix from an MSA. Be sure to update path to blosum_matrix.npy file.
Syntax: python score_matrix_parallel.py <msafile> <# of processors>

referenced this code for parallelization: 
https://stackoverflow.com/questions/25888255/how-to-use-python-multiprocessing-pool-map-to-fill-numpy-array-in-a-for-loop"""

import numpy as np
import sys
import time
from multiprocessing import Pool

t1 = time.time()
infile = sys.argv[1]
prc = int(sys.argv[2]) #number of processes

def scorer(seq1,seq2):
    counter = 0
    for x,y in zip(seq1,seq2):
        counter += blosum_matrix[x,y]
    return counter

def fill_matrix(i):
    row = [scorer(seqs[i],seqs[j]) for j in range(nseqs)]
    return(row)

##Load in sequences and blosum matrix
blosum_matrix = np.load("/home/chris/Documents/TRP-channels_Machine-Learning/clustering/blosum_matrix.npy")
ffile = open(infile).read()
entries = ffile.split(">")[1:] 
seqs = []
for e in entries:
    e = e.split("\n")
    seq = e[1]
    seqo = [ord(s) for s in seq]
    seqs.append(seqo)
nseqs = len(seqs)

##Build score matrix in parallel
if __name__ == "__main__":
    score_matrix = np.zeros((nseqs, nseqs))
    pool = Pool(prc)
    ids = range(nseqs)
    pool_result = pool.imap(fill_matrix, ids, chunksize=100)
    pool.close()
    pool.join()
    for line,result in enumerate(pool_result):
        score_matrix[line,:] = result
    #Save score matrix to file
    sm_name = infile[:-3]+"_score_matrix.csv"
    np.savetxt(sm_name,score_matrix,delimiter=",")

t2 = time.time()
tt = t2-t1
print("Total time: %d" %tt)
print(score_matrix.shape)
