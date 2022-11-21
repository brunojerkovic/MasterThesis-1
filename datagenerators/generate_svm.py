import numpy as np
import random
import os
from datagenerators.generator import DataGenerator


class SVMGenerator(DataGenerator):
    def __init__(self, config):
        super(SVMGenerator, self).__init__(config)
        self.c11, self.c12, self.c21, self.c22 = config.c11, config.c12, config.c21, config.c22
        self.sigma_eta_diag, self.sigma_eta_off_diag = config.sigma_eta_diag, config.sigma_eta_off_diag
        self.sigma_eps_diag, self.sigma_eps_off_diag = config.sigma_eps_diag, config.sigma_eps_off_diag
        self.mu = np.array([config.mu_value] * self.n_data)

    def _generate_series(self) -> tuple:
        # Generate matrices for coefficients and noise
        coef_mat = np.array([
            [self.c11, self.c12],
            [self.c21, self.c22]
        ])
        coef_mat = self._make_coef_stationary(coef_mat)
        sigma_eta = np.array([
            [self.sigma_eta_diag, self.sigma_eta_off_diag],
            [self.sigma_eta_off_diag, self.sigma_eta_diag]
        ])
        sigma_eps = np.array([
            [self.sigma_eps_diag, self.sigma_eps_off_diag],
            [self.sigma_eps_off_diag, self.sigma_eps_diag]
        ])

        # Generate noises
        noise_eta = np.random.multivariate_normal(mean=[0] * self.n_data, cov=sigma_eta,
                                                  size=(self.time + self.burn_in,))
        noise_eps = np.random.multivariate_normal(mean=[0] * self.n_data, cov=sigma_eps,
                                                  size=(self.time + self.burn_in,))

        # Generate datagenerators
        y = np.empty((self.time + self.burn_in, self.n_data))
        h = np.empty((self.time + self.burn_in, self.n_data))
        for i in range(self.lag):
            h[i] = noise_eta[i]
        for t in range(self.lag, self.time + self.burn_in):
            h[t] = self.mu @ (h[t - 1] - self.mu) + noise_eta[t]
            omega = np.eye(self.n_data) * np.exp(h[t] / 2)
            y[t] = omega @ noise_eps[t]

        y = y[self.burn_in:]

        return y, coef_mat
