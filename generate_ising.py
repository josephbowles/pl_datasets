from qml_benchmarks.data.ising import generate_ising, IsingSpins
import networkx as nx
import numpy as np
import numpyro
import pennylane as qml

np.random.seed(42)
numpyro.set_host_device_count(8)

T = 3  # Temperature
burn_in = 50000
num_samples_per_chain = 1000000
num_chains = 8
total_samples = num_samples_per_chain * num_chains

data = {}
data['train'] = {}
data['test'] = {}

for height in [2, 3, 4, 5, 6]:
    for width in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        if width * height <= 25 and width * height >= 8 and width >= height:
            if width * height < 16:
                n_train = int(2 ** (width * height) * 0.1)  # 10% of all bit strings
                n_test = int(2 ** (width * height) * 0.3)
                thinning = total_samples // (n_train + n_test) // 2
                print('chain thinning: ' + str(thinning))
            else:
                n_train = 5000
                n_test = 50000
                thinning = total_samples // (n_train + n_test)
                print('chain thinning: ' + str(thinning))

            if width == 4 and height == 4:
                # to reproduce data from paper
                np.random.seed(666)

            N = width * height

            # create a random 2D lattice lattice graph
            G = nx.grid_2d_graph(height, width, periodic=True)
            J = nx.adjacency_matrix(G).toarray()
            J = J * np.random.rand(*J.shape) * 2  # random positive weights
            for i in range(J.shape[0]):
                for j in range(i):
                    av = (J[i, j] + J[j, i]) / 2
                    J[i, j] = av
                    J[j, i] = av
            print(J)
            b = np.zeros(N)

            model = IsingSpins(N=N, J=J, b=b, T=T)
            all_samples = model.sample(num_samples_per_chain, num_chains=num_chains, thinning=thinning,
                                       num_warmup=burn_in)

            if all_samples.shape[0] < n_test + n_train:
                print('not enough points')
                break

            # randomise order
            idxs = np.random.permutation(all_samples.shape[0])
            samples_train = all_samples[idxs[:n_train]]
            samples_test = all_samples[idxs[n_train:n_train + n_test]]

            samples_train = np.array(samples_train, dtype='int32')
            samples_test = np.array(samples_test, dtype='int32')

            np.savetxt(f'./data/ising_{height}_{width}_train.csv', samples_train, fmt='%d')
            np.savetxt(f'./data/ising_{height}_{width}_test.csv', samples_test, fmt='%d')

            data['train'][f'({height},{width})'] = samples_train
            data['test'][f'({height},{width})'] = samples_test

dataset = qml.data.Dataset(data_name = "ising", train=data['train'], test=data['test'])
dataset.write("./datasets/ising.h5")
