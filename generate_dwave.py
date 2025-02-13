import requests
import tarfile
import os
import numpy as np
import pennylane as qml

# Define the URL and file names
url = "https://zenodo.org/records/7250436/files/datasets.tar.gz?download=1"
file_name = "./data/dwave_data.tar.gz"
extract_path = "./data/dwave/"

# Download the file
print("Downloading dwave data..")
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(file_name, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print(f"Downloaded {file_name} successfully.")
else:
    print(f"Failed to download file. Status code: {response.status_code}")
    exit(1)

# Extract the tar.gz file
print("Extracting the file...")
if tarfile.is_tarfile(file_name):
    with tarfile.open(file_name, "r:gz") as tar:
        tar.extractall(path=extract_path)
    print(f"Extracted files to {os.path.abspath(extract_path)}")
else:
    print(f"{file_name} is not a valid tar.gz file.")

X_train = np.load('./data/dwave/datasets/484-z8-100mus/train-484spins-3nn-uniform-100mus.npy')
X_test = np.load('./data/dwave/datasets/484-z8-100mus/test-484spins-3nn-uniform-100mus.npy')

X_train = np.reshape(X_train,(X_train.shape[0],-1))
X_train = X_train[:10000]
X_train = (1+X_train)/2

X_test = np.reshape(X_test,(X_test.shape[0],-1))
X_test = (1+X_test)/2


np.savetxt('./data/dwave_X_train.csv', X_train, delimiter=',',  fmt='%d')
np.savetxt('./data/dwave_X_test.csv', X_test, delimiter=',', fmt='%d')

dataset = qml.data.Dataset(data_name = "d-wave",
                           train = X_train,
                           test = X_test)

dataset.write("./datasets/d_wave.h5")

