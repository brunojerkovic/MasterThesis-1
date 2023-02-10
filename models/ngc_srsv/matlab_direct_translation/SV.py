import numpy as np

from models.ngc_srsv.matlab_direct_translation.Parameter import Parameter
from models.ngc_srsv.matlab_direct_translation.Distribution import Distribution


class SV:
    def __init__(self, model_name='SV', distr_name='gaussian'):
        self.model_name = model_name
        self.distr_name = distr_name

        self.mu = Parameter(name='mu')
        # TODO: i think sigma2
        self.sigma2 = Parameter(name='sigma2',
                                prior=Distribution(name='IG', parameters=[2.5, 0.25]))
        self.phi = Parameter(name='phi',
                             prior=Distribution(name='beta', parameters=[20,1.5]),
                             prior_transform=lambda x: (x+1)/2,
                             prior_inv_transform=lambda x: 2*x-1,
                             jacobian_offset=lambda: np.log(0.5))
        #self.measurement_fcn = '@SVmeasurementFnc' # TODO: fix this
        #self.state_transition_fcn = '@SVstateTransitionFnc' # TODO: fix this

        self.params_SV = [
            self.mu,
            self.sigma2,
            self.phi
        ]
        self.num_params = len(self.params_SV)

    def state_initialize(self, num_particle):
        """
        Initialize state
        """
        mu, sigma2, phi = np.empty(num_particle), np.empty(num_particle), np.empty(num_particle)
        for i in range(num_particle):
            mu[i] = self.mu.prior.random_number_generator()
            sigma2[i] = self.sigma2.prior.random_number_generator()
            phi[i] = self.phi.prior.random_number_generator()
        state_init = mu + (sigma2 / (1-phi**2))**0.5 * np.random.normal(loc=0, scale=1, size=(num_particle,))
        #state_init = self.mu.prior + (
        #        (self.sigma2 / (1 - self.param_phi ** 2)) ** 0.5 *
        #        np.random.randn(1, num_particle)
        #)
        return state_init

    #def print_msg(self, iteration):
    #    """
    #    Customize sampling message during the sampling phase.
    #    """
    #    print(f"Iteration: {iteration} | mu: {self.mu} | phi: {self.phi} | sigma2: {self.sigma2}")

    def SVmeasurementFnc(self, current_state, current_obs):
        logY_givenZ = -0.5 * np.log(2 * np.pi) - 0.5 * current_state - 0.5 * (current_obs ** 2) * np.exp(
            -current_state)
        return logY_givenZ

    def SVstateTransitionFnc(self, theta, old_state, random_number):
        new_state = theta['mu'] + theta['phi'] * (old_state - theta['mu']) + np.sqrt(theta['sigma2']) * random_number
        return new_state
