import torch
import torchvision
import numpy as np
import os
import pennylane as qml

if not os.path.exists('./datasets'):
    os.makedirs('./datasets')
if not os.path.exists('./data'):
    os.makedirs('./data')

# Download MNIST dataset
dataset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True)

X_train = dataset.data[:50000]
X_test = dataset.data[-10000:]
y_train = np.array(dataset.targets[:50000])
y_test = np.array(dataset.targets[-10000:])

X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))

#binarize data
X_train = np.array(X_train)/256
X_test = np.array(X_test)/256
X_train = np.array(X_train > 0.5, dtype=int)
X_test = np.array(X_test > 0.5, dtype=int)

np.savetxt('./data/mnist_x_train.csv', X_train, delimiter=",", fmt='%d')
np.savetxt('./data/mnist_x_test.csv', X_test, delimiter=",", fmt='%d')
np.savetxt('./data/mnist_y_train.csv', y_train, delimiter=",", fmt='%d')
np.savetxt('./data/mnist_y_test.csv', y_test, delimiter=",", fmt='%d')

dataset = qml.data.Dataset(data_name = "binarized_mnist",
                           train={'inputs': X_train, 'labels': y_train},
                           test={'inputs': X_test, 'labels': y_test})

dataset.write("./datasets/binarized_mnist.h5")




