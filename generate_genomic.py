import os
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pennylane as qml

np.random.seed(666)

if not os.path.exists('./datasets'):
    os.makedirs('./datasets')
if not os.path.exists('./data'):
    os.makedirs('./data')

def load_data_skip_columns(filename):
    # Load the data file, skipping the first two columns
    data = pd.read_csv(filename, delim_whitespace=True, header=None)

    # Drop the first two columns (index 0 and 1)
    data = data.drop(columns=[0, 1])

    # Convert the DataFrame to a NumPy array
    data_array = data.values

    return data_array

### 805

url = 'https://gitlab.inria.fr/ml_genetics/public/artificial_genomes/-/raw/29c1ef7cf242e842df4360abae2eebeec995f40e/1000G_real_genomes/805_SNP_1000G_real.hapt'
filename ='./data/805_SNP_1000G_real.hapt'
status = status = os.system(f'wget -O {filename} {url}')
if status != 0:
    print('Error downloading 805_SNP_1000G_real.hapt. Check if you have the latest version of wget installed in your PC.')
data = load_data_skip_columns(filename)
X_train, X_test = train_test_split(data, test_size=1/3)
np.savetxt('./data/805_SNP_1000G_real_train.csv', X_train, fmt='%d', delimiter=',')
np.savetxt('./data/805_SNP_1000G_real_test.csv', X_test, fmt='%d', delimiter=',')

dataset = qml.data.Dataset(data_name = "genomic",
                           train = X_train,
                           test = X_test)

dataset.write("./datasets/genomic.h5")







