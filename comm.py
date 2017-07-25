#!/usr/bin/python3
"""
This script provides useful common functions
"""
import openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing

def LoadOpenMLData(dataset_id=150, test_size=0.33, random_state=42):
    """ Load data from openml.org
    @dataset_id: id of covertype dataset on www.openml.org
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

def LoadTxtData(path, test_size=0.33, random_state=42, scale=False):
    df = pd.read_csv(path, sep=',', header=None)
    dataset = df.values
    y = dataset[:, 0]
    if scale:
        X = preprocessing.scale(dataset[:, 1:])
    else:
        X = dataset[:, 1:]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
