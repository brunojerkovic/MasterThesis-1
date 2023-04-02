import math
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def clean_data(config, series, coef_mat, edges):
    if config.loader != 'stocks':
        series = series[:(-config.timesteps*2), :]
    edges = edges.astype('int')

    return series, config.beta_value, edges, coef_mat


def clean_data2(data, device, x_dim=7, train_perc=0.8, batch_size=1):
    # Take metadata
    N, p = data.shape

    # Generate dataset
    idx_mids = [math.floor(x_dim / 2) + p_ for p_ in range(1)]
    inputs_np = [data[i:(i + x_dim * 1)] for i in range(0, N * 1, 1)]  # TODO: 3 puta si 'p' promijenija u '1'
    inputs_np = [[x for (i, x) in enumerate(input) if i not in idx_mids] for input in inputs_np]
    inputs_np = np.array([d for d in inputs_np if len(d) == len(inputs_np[0])])
    outputs_np = np.array([data[i:(i + 1), :] for i in range(idx_mids[0], len(data) - idx_mids[0])])

    inputs = torch.tensor(inputs_np, dtype=torch.float32).to(device)  # [:-1]
    outputs = torch.tensor(outputs_np, dtype=torch.float32).to(device)

    # Split dataset (and store into data loaders)
    N_train = int(inputs.shape[0] * train_perc)
    train_set = TensorDataset(inputs[:N_train], outputs[:N_train])
    valid_set = TensorDataset(inputs[N_train:], outputs[N_train:])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader