import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_tree():
    """
    Trains and saves a decision tree using knn_data.pickle and knn_labels.pickle.
    """
    pickle_in = open("./data/knn_data.pickle", "rb")
    dataset = pickle.load(pickle_in)
    dataset = dataset.reshape((3300, 50176))

    pickle_in = open("./data/knn_labels.pickle", "rb")
    labels = pickle.load(pickle_in)

    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    forest = RandomForestClassifier()
    search = RandomizedSearchCV(estimator=forest, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
    search.fit(dataset, labels.ravel())

    print(search.best_params_)
    best_model = search.best_estimator_

    file = open("./data/tree.pickle", "wb")
    pickle.dump(best_model, file)

def get_label(input):
    """
    Loads the saved decision tree and returns the resulting classification of input.
    """
    pickle_in = open("./data/tree.pickle", "rb")
    tree = pickle.load(pickle_in)
    img = input.reshape(1, 50176)
    return tree.predict(img).tolist()[0]