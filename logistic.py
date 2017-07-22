#!/usr/bin/python3
"""
This script import datasets from openml and build a logistic regression with different solvers
"""
import openml
from sklearn import preprocessing, linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# get dataset
datasets = openml.datasets.list_datasets()
dataset_id = 150  # 150 is the id of covertype dataset on www.openml.org
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# model 1

print("\nVanilla Logistic regression")
clf = linear_model.LogisticRegression(C=1.e4 / X.shape[0], penalty='l2', solver='sag')
clf.fit(X_train, y_train)

# evaluate model
y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)
test_score = accuracy_score(y_test, y_test_pred)
train_score = accuracy_score(y_train, y_train_pred)
print("Training accuracy: %f" % train_score)
print("Test accuracy: %f" % test_score)

# model 2

clf = SGDClassifier()
clf.fit(X_train, y_train)

# evaluate model
y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)
test_score = accuracy_score(y_test, y_test_pred)
train_score = accuracy_score(y_train, y_train_pred)
print("Training accuracy: %f" % train_score)
print("Test accuracy: %f" % test_score)
