import pickle


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


def preprocess(X, y):
    # Normalize the data
    X = X / 255.0
    # Reshape the data so that it can be fed into the model
    for idx in range(len(X)):
        X[idx] = X[idx].reshape(3, 32, 32).transpose(1, 2, 0)
    return X, y


def load_data(path_list):
    data_dict = {}
    for path in path_list:
        data_dict = data_dict | unpickle(path)
    X = data_dict[b"data"]
    y = data_dict[b"labels"]
    X, y = preprocess(X, y)
    return X, y
