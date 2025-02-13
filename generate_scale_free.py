import matplotlib.pyplot as plt
import numpy as np
import os
import numpyro
from qml_benchmarks.data.ising import IsingSpins, energy
from joblib import Parallel, delayed
import yaml
import networkx as nx
import pennylane as qml
np.random.seed(666)

# Generates a dataset with correlations based on the Barabasi-Albert scale free graph construction

# Numpyro usually treats the CPU as 1 device
# You can set this variable to split the CPU for parallel processing
numpyro.set_host_device_count(8)

###### SETTINGS ########
N = 1000  # number of nodes
m = 2 #connectivity
T = 1 #Temperature
bias_weight = -.01
burn_in = 100
num_samples_per_chain = 100
num_chains = 8
num_samples_train = 20
num_samples_test = 20
###### END OF SETTINGS ########

if not(os.path.exists('./data/scale_free_train.csv') and os.path.exists('./data/scale_free_test.csv')):

    G = nx.barabasi_albert_graph(N, m)
    J = nx.adjacency_matrix(G).toarray()

    J = J*np.random.rand(N, N)
    for i in range(J.shape[0]):
        for j in range(i):
            J[i,j] = J[j,i]

    degrees = np.array([deg for node, deg in G.degree()])
    b = degrees*bias_weight

    total_samples = num_samples_per_chain*num_chains
    #thin the samples as much as possible
    thinning = total_samples//(num_samples_train+num_samples_test)
    print('chain thinning: '+str(thinning))

    all_samples = []

    model = IsingSpins(N=N, J=J, b=b, T=T, sparse=True)
    all_samples = model.sample(num_samples_per_chain, num_chains=num_chains, thinning=thinning, num_warmup=burn_in)
    all_samples = all_samples*2-1 #convert to pm1

    #shuffle the data to remove correlations (takes a while)
    idxs = np.random.permutation(all_samples.shape[0])

    samples_train = all_samples[idxs[:num_samples_train]]
    samples_test = all_samples[idxs[num_samples_train:num_samples_train+num_samples_test]]

    #save as binary data
    np.savetxt('./data/scale_free_train.csv', samples_train, delimiter=",", fmt='%d')
    np.savetxt('./data/scale_free_test.csv', samples_test, delimiter=",", fmt='%d')

X_train = np.loadtxt('./data/scale_free_train.csv', delimiter=",")
X_test = np.loadtxt('./data/scale_free_test.csv', delimiter=",")

dataset = qml.data.Dataset(data_name = "scale_free",
                           train = X_train,
                           test = X_test)

dataset.write("./datasets/scale_free.h5")