#!/usr/bin/python3
"""
This script provides useful common functions
"""
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from sklearn.datasets import fetch_rcv1

def LoadOpenMLData(dataset_id=150, test_size=0.33, random_state=42):
    """ Load data from openml.org
    @dataset_id: id of dataset on www.openml.org (covertype: 150)
    @test_size: the size of the test set, range: (0, 1)
    @random_state: how the data is split
    return [X_train, X_test, y_train, y_test]
    """
    # get dataset
    datasets = openml.datasets.list_datasets()
    dataset = openml.datasets.get_dataset(dataset_id)
        # @NOTE target makes get_data() return X and y seeperate,
        #       return_categorical_indicator makes get_data() return a boolean array
        #       which indicate which attributes are categorical (and should be one hot encoded if necessary)
    X, y, categorical = dataset.get_data(
        target=dataset.default_target_attribute,
        return_categorical_indicator=True)
    print("shape of data points:", X.shape, "shape of targets:", y.shape)
    # one-hot code
    #print(categorical)
    #enc = preprocessing.OneHotEncoder(categorical_features=categorical)  # one-hot code
    #print(enc)
    #X = enc.fit_transform(X).todense()
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def LoadYearPredictionData(path='../data/YearPredictionMSD.txt', test_size=0.33, random_state=42, scale=False, sep=','):
    df = pd.read_csv(path, sep=sep, header=None)
    dataset = df.values
    if scale:
        X = preprocessing.scale(dataset[:, 1:])
        y = preprocessing.scale(dataset[:, 0])
    else:
        X = dataset[:, 1:]
        y = dataset[:, 0]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def LoadRCV1Data(test_size=0.33, random_state=42):
    rcv1 = fetch_rcv1()
    X = rcv1.data # @NOTE the feature matrix is a scipy csr matrix
    y = rcv1.target
    labels = rcv1.target_names[:].tolist()
    ccat = labels.index('CCAT')
    y = y.toarray()
    y_bin = np.ones(y.shape[0])
    for i in range(y.shape[0]):
        if y[i][ccat] != 1:
            y_bin[i] = -1
    # return X_train, X_test, y_train, y_test, where X_{train, test} is of csr matrix type
    return train_test_split(X, y_bin, test_size=test_size, random_state=random_state)

def LoadSidoData(dataPath='../data/sido/sido2_train.data', targetPath='../data/sido/sido2_train.targets', test_size=0.33, random_state=42, scale=False, sep=' '):
    df = pd.read_csv(dataPath, sep=sep, header=None)
    dataset = df.values
    tf = pd.read_csv(targetPath, sep=sep, header=None)
    targets = tf.values
    if scale:
        X = preprocessing.scale(dataset)
        y = preprocessing.scale(targets[:, 0])
    else:
        X = dataset
        y = targets[:, 0]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
