import numpy as np


def split_data(scaled_data, prediction_days=14):
    """
    Splits the dataset into three sets: one for training, validation, and testing each.

    Parameters:
        - scaled_data (DataFrame): The input data with the 'Close' column being normalized.
    """
    X_train = []
    t_train = []

    for x in range(prediction_days, len(scaled_data)):
        X_train.append(scaled_data[x - prediction_days:x, 0])
        t_train.append(scaled_data[x, 0])

    X_train, t_train = np.array(X_train), np.array(t_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, t_train
