import torch
import math


def data_splitter(X: torch.Tensor, train_test_split, it, max_iter):
    train_perc, valid_perc = train_test_split[0], train_test_split[1] + train_test_split[2]
    n_windows = max_iter - it
    window_size = math.ceil(X.shape[0] / n_windows)
    X_ = X.unfold(0, n_windows, window_size)
    X_train = X_[:, :int(X_.shape[1] * train_perc)]
    X_valid = X_[:, int(X_.shape[1] * train_perc):]

    return X_train, X_valid
