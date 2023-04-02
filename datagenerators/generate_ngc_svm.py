import numpy as np
from datagenerators.generator import DataGenerator


class NGCSVMGenerator(DataGenerator):
    def __init__(self, config):
        super(NGCSVMGenerator, self).__init__(config)
        self.sparsity = config.sparsity
        self.beta_value = config.beta_value
        self.sigma_eta_diag = config.sigma_eta_diag
        self.sigma_eps_diag = config.sigma_eps_diag
        self.std1 = np.sqrt(self.sigma_eta_diag)
        self.std2 = np.sqrt(self.sigma_eps_diag)

    def _generate_series(self) -> tuple:
        # Set up coefficients and Granger causality ground truth.
        GC = np.eye(self.n_data, dtype=int)
        coef_mat = np.eye(self.n_data) * self.beta_value

        num_nonzero = int(self.n_data * self.sparsity) - 1
        for i in range(self.n_data):
            choice = np.random.choice(self.n_data - 1, size=num_nonzero, replace=False)
            choice[choice >= i] += 1
            coef_mat[i, choice] = self.beta_value
            GC[i, choice] = 1

        coef_mat = np.hstack([coef_mat for _ in range(self.lag)])
        coef_mat = self.__create_stat_coef_mat(coef_mat)

        # Generate datagenerators.
        noise_eta = np.random.normal(scale=self.std1, size=(self.n_data, self.time + self.burn_in))
        noise_eps = np.random.normal(scale=self.std2, size=(self.n_data, self.time + self.burn_in))
        X = np.zeros((self.n_data, self.time + self.burn_in))
        H = np.zeros((self.n_data, self.time + self.burn_in))
        X[:, :self.lag] = noise_eps[:, :self.lag]
        H[:, :self.lag] = noise_eta[:, :self.lag]
        for t in range(self.lag, self.time + self.burn_in):
            H[:, t] = np.dot(coef_mat, H[:, (t - self.lag):t].flatten(order='F')) + noise_eta[:, t]
            omega = np.eye(self.n_data) * np.exp(H[:, t] / 2)
            X[:, t] = omega @ noise_eps[:, t]

        X = X.T[self.burn_in:]
        return X, coef_mat

    def __create_stat_coef_mat(self, coef_mat):
        '''Rescale coefficients of VAR model to make stable.'''
        p = coef_mat.shape[0]
        lag = coef_mat.shape[1] // p
        bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
        beta_tilde = np.vstack((coef_mat, bottom))
        eigvals = np.linalg.eigvals(beta_tilde)
        max_eig = max(np.abs(eigvals))
        nonstationary = max_eig > self.stationarity_radius
        if nonstationary:
            return self.__create_stat_coef_mat(0.95 * coef_mat)
        else:
            return coef_mat
