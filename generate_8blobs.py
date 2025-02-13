import matplotlib.pyplot as plt
from qml_benchmarks.data.spin_blobs import generate_8blobs
import numpy as np
import os
import yaml
import pennylane as qml

np.random.seed(66)

###### SETTINGS #######
num_samples_train = 5000
num_samples_test = 10000
noise_prob = 0.05
### END OF SETTINGS ###

n_spins = 16
n_blobs = 8

X, y = generate_8blobs(num_samples_train+num_samples_test, noise_prob)

fig, axes = plt.subplots(ncols=20, figsize = (10,5))
for i, config in enumerate(X[:20]):
    axes[i].imshow(np.reshape(config, (4,4)))
    axes[i].set_xticklabels([])
    axes[i].set_yticklabels([])
plt.show()

np.savetxt('./data/8blobs_x_train.csv', X[:num_samples_train], delimiter=',', fmt='%d')
np.savetxt('./data/8blobs_y_train.csv', y[:num_samples_train], delimiter=',', fmt='%d')
np.savetxt('./data/8blobs_x_test.csv', X[-num_samples_test:], delimiter=',', fmt='%d')
np.savetxt('./data/8blobs_y_test.csv', y[-num_samples_test:], delimiter=',', fmt='%d')

dataset = qml.data.Dataset(data_name = "binary_blobs",
                           train={'inputs': X[:num_samples_train], 'labels': y[:num_samples_train]},
                           test={'inputs': X[:num_samples_test], 'labels': y[:num_samples_test]})

dataset.write("./datasets/binary_blobs.h5")

