import collections
import numpy as np
import pickle

def euclidian_distance(point1, point2):
    """
    Find the Euclidian distance between data points.
    """
    return np.linalg.norm(point1 - point2)

def get_nearest_neighbors(input, k):
    """
    Given a set of datapoints and their corresponding labels (saved in knn_data.pickle and
    knn_labels.pickle), take an input and find the k nearest neighbors, using the majority
    vote to classify the unlabeled input.
    """
    pickle_in = open("./data/knn_data.pickle", "rb")
    dataset = pickle.load(pickle_in)

    pickle_in = open("./data/knn_labels.pickle", "rb")
    labels = pickle.load(pickle_in)

    distances = []
    for data in dataset:
        distances.append(euclidian_distance(data, input))

    top_k = np.argsort(np.array(distances))[:k]
    votes = [labels[i][0] for i in top_k]
    try:
        vote_result = collections.Counter(votes).most_common(2)
        return [vote_result[0][0], vote_result[1][0]]

    except: # If there is only one classification in all k neighbors
        vote_result = collections.Counter(votes).most_common(2)
        return [vote_result[0][0], vote_result[0][0]]

def search_k(dataset, min_k, max_k):
    """
    Given a dataset with validation data, test the k-nearest neighbors algorithm with odd k
    values ranging from min_k to max_k and display their accuracies.
    """
    accuracies = {}
    num_samples = len(dataset)

    from tqdm import tqdm
    for k in tqdm(range(min_k, max_k + 1, 2)):
        num_accurate = 0

        for data, label in dataset:
            if get_nearest_neighbors(data, k)[0] == label:
                num_accurate += 1

        accuracies[k] = num_accurate / num_samples

    print(accuracies)