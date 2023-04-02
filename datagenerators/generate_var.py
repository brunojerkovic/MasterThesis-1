import numpy as np
from scipy.stats import multivariate_t as t
from datagenerators.generator import DataGenerator

#written by Enzo Michelangeli, style changes by josef-pktd
# Student's T random variable
def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution
    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = np.ones(n)
    else:
        x = np.random.chisquare(df, n) / df
    z = np.random.multivariate_normal(np.zeros(d), S, (n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal



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
        if self.sigma_eta_diag == self.sigma_eta_off_diag:
            self.sigma_eta_off_diag = {
                0.5: 0.4,
                0.1: 0.9,
                0.05: 0.04,
                0.01: 0.005
            }[self.sigma_eta_diag]
            self.config.sigma_eta_off_diag = self.sigma_eta_off_diag
        noise_var = np.array([
            [self.sigma_eta_diag, self.sigma_eta_off_diag],
            [self.sigma_eta_off_diag, self.sigma_eta_diag]
        ])

        # Generate initial values
        X = np.empty((self.time + self.burn_in, self.n_data))
        #self.noise = multivariate_t_rvs([0]*self.n_data, S=noise_var, df=2, n=self.time+self.burn_in)
        self.noise = np.random.multivariate_normal(mean=[0]*self.n_data, cov=noise_var, size=(self.time+self.burn_in,))
        X[:self.lag, :] = self.noise[:self.lag, :]
        #for i in range(self.lag):
        #    X[i, :] = noise[i, :]

        # Generate datagenerators
        #self.noise = np.random.multivariate_normal(mean=[0] * self.n_data, cov=noise_var, size=(self.time + self.burn_in,))
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
