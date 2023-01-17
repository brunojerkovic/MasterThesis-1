import numpy as np
from datagenerators.generator import DataGenerator


class VARGenerator(DataGenerator):
    def __init__(self, config):
        super(VARGenerator, self).__init__(config)
        self.c11, self.c12, self.c21, self.c22 = config.c11, config.c12, config.c21, config.c22
        self.sigma_eta_diag, self.sigma_eta_off_diag = config.sigma_eta_diag, config.sigma_eta_off_diag
        self.mu = np.array([config.mu_value] * self.n_data)

        self.noise = None

    def _generate_series(self) -> tuple:
        # Generate matrices for coefficients and noise
        coef_mat = np.array([
            [self.c11, self.c12],
            [self.c21, self.c22]
        ])
        coef_mat = self._make_coef_stationary(coef_mat)
        noise_var = np.array([
            [self.sigma_eta_diag, self.sigma_eta_off_diag],
            [self.sigma_eta_off_diag, self.sigma_eta_diag]
        ])

        # Generate initial values
        X = np.empty((self.time + self.burn_in, self.n_data))
        for i in range(self.lag):
            X[i, :] = np.random.multivariate_normal(mean=[0] * self.n_data, cov=noise_var, size=(1,))

        # Generate datagenerators
        self.noise = np.random.multivariate_normal(mean=[0] * self.n_data, cov=noise_var, size=(self.time + self.burn_in,))
        for t in range(self.lag, self.time + self.burn_in):
            X[t, :] = self.mu + (coef_mat @ X[t - 1, :] - self.mu) + self.noise[t, :]
        X = X[self.burn_in:]
        self.noise = self.noise[self.burn_in:]

        return X, coef_mat

    def get_noise(self) -> np.ndarray:
        '''
        Copy noise vector and return it
        :return: A numpy array of noise
        '''
        return np.copy(self.noise)
