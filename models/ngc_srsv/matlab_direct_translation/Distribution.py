import numpy as np
from scipy import stats, special


class Distribution:
    """
    A superclass for probability distribution.
    """

    def __init__(self, name='normal', parameters=None, dimension=None, transform_type='linear',
                 inv_transform_for_random_walk=None, log_jacobian_for_random_walk=None):
        self.name = name
        self.dimension = np.array([1, 1] if dimension is None else dimension)
        self.parameters = np.array([0, 0.1] if parameters is None else parameters)

        # TODO: dodaj ove vrijednosti
        self.inv_transform_for_random_walk = inv_transform_for_random_walk
        self.log_jacobian_for_random_walk = log_jacobian_for_random_walk
        self.transform_type = transform_type

    def log_pdf(self, theta):
        a, b = self.parameters
        if self.name == 'normal':
            return np.log(stats.norm.pdf(theta, a, np.sqrt(b)))
        elif self.name == 'IG':
            return a*np.log(b) - special.gammaln(a) - (a+1)*np.log(theta) - b/theta
        elif self.name == 'beta':
            return np.log(stats.beta.pdf(theta, a, b))
        elif self.name == 'gamma':
            return np.log(stats.gamma.pdf(theta, a, b))
        elif self.name == 'cauchy':
            return np.log(b / (np.pi * (np.pow(b, 2) + np.pow(theta-1, 2))))
        elif self.name == 'uniform':
            return np.log(stats.uniform.pdf(theta, a, b))
        else:
            raise ValueError("Attribute 'name' is not set correctly for Distribution.log_pdf")

    # TODO: provjeri je li ovo sa dimenzijom dobro
    def random_number_generator(self, dim=1):
        a, b = self.parameters

        if self.name == 'normal':
            return np.random.normal(a, np.sqrt(b), dim)  # b is variance, sqrt(b) is std
        elif self.name == 'IG':
            return 1 / np.random.gamma(a, 1/b, dim)
        elif self.name == 'beta':
            return np.random.beta(a, b, dim)
        elif self.name == 'gamma':
            return np.random.gamma(a, b, dim)
        elif self.name == 'cauchy':
            return a + b * np.tan(np.pi * (np.random.uniform() - 0.5))
        elif self.name == 'uniform':
            return np.random.uniform(a, b, dim)
        else:
            raise ValueError("Attribute 'name' is not set correctly for Distribution.random_number_generator")

    def transform_for_random_walk(self, value):
        if self.transform_type == 'linear':
            return value
        elif self.transform_type == 'log':
            return np.log(value)
        elif self.transform_type == 'logit':
            return np.log(value / (1-value))

    def inv_transform_after_random_walk(self, value):
        if self.transform_type == 'linear':
            return value
        elif self.transform_type == 'log':
            return np.exp(value)
        elif self.transform_type == 'logit':
            return utils.sigmoid(value)

    def log_jacobian_random(self, value):
        if self.transform_type == 'linear':
            return 0
        elif self.transform_type == 'log':
            return np.log(value)
        elif self.transform_type == 'logit':
            return np.log(value) + np.log(1-value)

    def set_transform_for_random_walk(self):
        transforms = {
            'normal': 'linear',
            'IG': 'log',
            'beta': 'logit',
            'gamma': 'log',
            'cauchy': 'log',
            'uniform': 'linear'
        }
        return transforms[self.name]