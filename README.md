# affinity_propagation
Set of python scripts to cluster protein sequences using affinity propagation.

Workflow:
* Use score_matrix_parallel.py to generate a matrix containing the each pairwise blosum similarity score for all sequences in the input MSA.
* Use affinity_param_testing.py to determine the value of the affinity propagation preference parameter that optimizes H(S) and H(K) for the dataset. (Look at .png and .csv file outputs; choose last preference value where H(K) and H(S) can reasonably be said to be increasing linearly with each other.)
* Use affinity_clustering.py to cluster the sequences using sklearn's implementation of affinity propagation.
