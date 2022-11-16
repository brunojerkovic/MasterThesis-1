import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader


def format_edges(config, edges):
    # train_size, valid_size, test_size = args.train_size, args.valid_size, args.test_size
    train_size = config.trainset_size
    valid_size = int(config.trainset_size * config.tvt_split)
    test_size = int(config.trainset_size * config.tvt_split)

    np.fill_diagonal(edges, 0)

    edges = np.expand_dims(edges, axis=-1)

    edges_train = np.tile(edges, (train_size)).transpose(2,1,0)
    edges_valid = np.tile(edges, (valid_size)).transpose(2,1,0)
    edges_test = np.tile(edges, (test_size)).transpose(2,1,0)

    return edges_train, edges_valid, edges_test

def format_features(config, features):
    train_sims = config.trainset_size
    valid_sims = int(config.trainset_size * config.tvt_split)
    test_sims = int(config.trainset_size * config.tvt_split)
    time = train_sims + valid_sims + test_sims + 2*config.timesteps

    # Format the data in a nice way
    set_ = []
    timesteps = config.timesteps
    for s in range(0, time - config.timesteps * 2):
        if s == train_sims + valid_sims:
            timesteps *= 2
        X_ = np.array(features[s:s + timesteps, :].T)
        set_.append(X_)

    # Create splits of data
    train_set = set_[:train_sims]
    train_set = np.expand_dims(np.array(train_set), -1).transpose(0,2,3,1)

    valid_set = set_[train_sims:(train_sims + valid_sims)]
    valid_set = np.expand_dims(np.array(valid_set), -1).transpose(0,2,3,1)

    test_set = set_[(train_sims + valid_sims):]
    test_set = np.expand_dims(np.array(test_set), -1).transpose(0,2,3,1)

    return train_set, valid_set, test_set


def parse_data(config, features, edges):
    feat_train, feat_valid, feat_test = format_features(config, features)
    edges_train, edges_valid, edges_test = format_edges(config, edges)

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = feat_train.shape[3]

    feat_max = feat_train.max()
    feat_min = feat_train.min()

    # Normalize to [-1, 1]
    feat_train = (feat_train - feat_min) * 2 / (feat_max - feat_min) - 1
    feat_valid = (feat_valid - feat_min) * 2 / (feat_max - feat_min) - 1
    feat_test = (feat_test - feat_min) * 2 / (feat_max - feat_min) - 1

    # reshape to [num_sims, num_atoms, num_timesteps, num_dimensions]
    feat_train = np.transpose(feat_train, [0, 3, 1, 2])
    feat_valid = np.transpose(feat_valid, [0, 3, 1, 2])
    feat_test = np.transpose(feat_test, [0, 3, 1, 2])

    edges_train = np.reshape(edges_train, [-1, num_atoms ** 2])
    edges_valid = np.reshape(edges_valid, [-1, num_atoms ** 2])
    edges_test = np.reshape(edges_test, [-1, num_atoms ** 2])
    edges_train = np.array((edges_train + 1) / 2, dtype=np.int64)
    edges_valid = np.array((edges_valid + 1) / 2, dtype=np.int64)
    edges_test = np.array((edges_test + 1) / 2, dtype=np.int64)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    # Exclude self edges
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges_train = edges_train[:, off_diag_idx]
    edges_valid = edges_valid[:, off_diag_idx]
    edges_test = edges_test[:, off_diag_idx]

    # Convert to dataset
    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    # Convert to DataLoaders
    train_data_loader = DataLoader(train_data, batch_size=config.batch_size)
    valid_data_loader = DataLoader(valid_data, batch_size=config.batch_size)
    test_data_loader = DataLoader(test_data, batch_size=config.batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, feat_max, feat_min

def clean_data(config, series, coef_mat, edges):
    coef_mat = coef_mat.T
    edges = edges.T

    train_loader, valid_loader, test_loader, _, _ = parse_data(config, series, edges)
    return train_loader, valid_loader, test_loader
