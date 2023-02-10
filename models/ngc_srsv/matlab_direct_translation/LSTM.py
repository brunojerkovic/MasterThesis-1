import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, num_paths, num_obs, theta):
        super(LSTM, self).__init__()
        self.h = torch.zeros(num_paths, num_obs)
        self.eta = torch.zeros(num_paths, num_obs)
        self.eta[:, 1] = theta.beta0 + torch.normal(0, torch.sqrt(theta.sigma2).item(), num_paths)
        self.memory_cell = torch.zeros(num_paths, num_obs)
        self.dim_state = num_paths

    def forward(self, theta, idx):
        a_d = theta.v_d * self.eta[:, idx] + theta.w_d * self.h[:, idx] + theta.b_d
        a_i = theta.v_i * self.eta[:, idx] + theta.w_i * self.h[:, idx] + theta.b_i
        a_o = theta.v_o * self.eta[:, idx] + theta.w_o * self.h[:, idx] + theta.b_o
        a_f = theta.v_f * self.eta[:, idx] + theta.w_f * self.h[:, idx] + theta.b_f

        z_d = utils.activation(a_d, 'Tanh')
        g_i = utils.activation(a_i, 'Sigmoid')
        g_o = utils.activation(a_o, 'Sigmoid')
        g_f = utils.activation(a_f, 'Sigmoid')

        self.memory_cell[:, idx+1] = g_f @ self.memory_cell[:, idx] + g_i @ z_d
        self.h[:, idx+1] = g_o @ tanh(self.memory_cell[:, idx+1])
        self.eta[:, idx+1] = theta.beta_0 + theta.beta_1 @ self.h[:, idx+1] + torch.normal(0, torch.sqrt(theta.sigma2).item(), self.dim_state)

    def resampling(self, idx, resample_idx):
        self.eta[:, idx] = self.eta[resample_idx, idx]
        self.h[:, idx] = self.h[resample_idx, idx]
        self.memory_cell[:, idx] = self.memory_cell[resample_idx, idx]