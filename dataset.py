import openml
from sklearn import preprocessing, ensemble
from sklearn.metrics import accuracy_score

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
print(X.shape, y.shape)
# one-hot code
print(categorical)
enc = preprocessing.OneHotEncoder(categorical_features=categorical)  # one-hot code
print(enc)
X = enc.fit_transform(X).todense()

# model
clf = ensemble.RandomForestClassifier()
clf.fit(X, y)

# evaluate model
y_pred = clf.predict(X)
score = accuracy_score(y, y_pred)
print("Training accuracy: %f" % score)
