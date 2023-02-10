import numpy as np
from models.ngc_srsv.matlab_direct_translation.SV import SV
from models.ngc_srsv.matlab_direct_translation.Parameter import Parameter
from models.ngc_srsv.matlab_direct_translation.Distribution import Distribution
import models.ngc_srsv.matlab_direct_translation.utils as utils
from typing import List


class LSTMSV(SV):
    def __init__(self, model_name='LSTM-SV', distr_name='gaussian', *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.model_name = model_name
        self.distr_name = distr_name

        self.beta0 = Parameter(name='beta0', prior=Distribution(name='normal', parameters=[0, 0.01]))
        self.beta1 = Parameter(name='beta1', prior=Distribution(name='IG', parameters=[2.5, 0.25]))
        self.phi = Parameter(name='phi', prior=Distribution(name='beta',parameters=[20, 1.5]),
                             prior_transform=lambda x: (x+1)/2, prior_inv_transform=lambda x: 2*x-1,
                             jacobian_offset=lambda x: np.log(0.5))
        self.sigma2 = Parameter(name='sigma2', prior=Distribution(name='IG', parameters=[2.5, 0.25]))

        self.param_vd = Parameter(name='v_d')
        self.param_wd = Parameter(name='w_d')
        self.param_bd = Parameter(name='b_d')
        self.param_vi = Parameter(name='v_i')
        self.param_wi = Parameter(name='w_i')
        self.param_bi = Parameter(name='b_i')
        self.param_vo = Parameter(name='v_o')
        self.param_wo = Parameter(name='w_o')
        self.param_bo = Parameter(name='b_o')
        self.params_vf = Parameter(name='v_f')
        self.params_wf = Parameter(name='w_f')
        self.params_bf = Parameter(name='b_f')

        self.params: List[Parameter] = [
            self.beta0,
            self.beta1,
            self.phi,
            self.sigma2,
            self.param_vd,
            self.param_wd,
            self.param_bd,
            self.param_vi,
            self.param_wi,
            self.param_bi,
            self.param_vo,
            self.param_wo,
            self.param_bo,
            self.params_vf,
            self.params_wf,
            self.params_bf,
        ]
        self.num_params = len(self.params)
        self.name_params = [p.name for p in self.params]

    def lstmSVstateTransitionFnc(self, previous_state):
        pass
        #eta[:, t] = self.beta0 + self.beta1 * np.random.normal(0, np.sqrt(self.sigma2), N)
        #new_state[:, t] = eta[:, t] + model['phi'] * previous_state
        #return new_state

    def lstmSVmeasurementFnc(model, pre_state):
        pass

    def simulate(self, num_obs, *args, **kwargs):
        """
        Simulate from an LSTM-SV model
        """
        V, Y, theta = lstmSVsimulate(self, num_obs, *args, **kwargs)
        return V, Y, theta

    def initialize(self, N):
        """
        Initialize a parameter arrays from priors
        """
        theta = np.empty((len(self.params), N))

        for j in range(N):
            for i, param in enumerate(self.params):
                theta[i, j] = param.prior.random_number_generator()

        return theta

    def forecast_out(self, *args, **kwargs):
        """
        Make forecase with an LSTM model.
        """
        forecast_out = lstmSVforecast(self, *args, **kwargs)
        return forecast_out

    def print_msg(self, theta, i):
        """
        Print function.
        """
        print(f"Iteration: {i} | "
              f"beta0: {theta.beta0} |"
              f"beta1: {theta.phi} |"
              f"sigma2: {theta.sigma2}")

    def lstmSVcorrSMC(self, theta, N, y, u_pro, u_res):
        T = len(y)

        z = np.zeros((N, T))
        h = np.zeros((N, T))
        eta = np.zeros((N, T))
        memory_cell = np.zeros((N, T))
        weights = np.zeros((T, N))
        indx = np.zeros((N, T), dtype=np.int32)  # Store N resampling index in each of T time step

        # Sample particles at time t = 1
        t = 0
        eta[:, t] = theta.beta0 + np.sqrt(theta.sigma2) * u_pro[:, t].T
        z[:, t] = eta[:, t] + theta.phi * np.log(np.var(y))

        # Calculate weights for particles at time = 1
        logw = -0.5 * np.log(2 * np.pi) - 0.5 * z[:, t] - 0.5 * y[t] ** 2 * np.exp(-z[:, t])

        # Numerical stability
        weights[t, :] = np.exp(logw - np.max(logw))

        # Estimate marginal likelihood
        sir_llh = np.log(np.mean(weights[t, :])) + np.max(logw)

        # Normalize weigths
        weights[t, :] = weights[t, :] / np.sum(weights[t, :])

        for t in range(1, T):
            # Resampling
            indx[:, t] = utils.rs_multinomial_sort(z[:N, t - 1], weights[t - 1, :], u_res[:, t].T)
            z[:, t-1] = z[indx[:, t], t - 1]
            eta[:, t-1] = eta[indx[:, t], t - 1]
            h[:, t-1] = h[indx[:, t], t - 1]
            memory_cell[:, t-1] = memory_cell[indx[:, t], t - 1]

            # Generate particles at time t>=2.
            z_d = utils.activation(theta.v_d * eta[:, t-1] + theta.w_d * h[:, t-1] + theta.b_d,
                                   'tanh')  # Data input
            g_i = utils.activation(theta.v_i * eta[:, t-1] + theta.w_i * h[:, t-1] + theta.b_i,
                                   'sigmoid')  # Input gate
            g_o = utils.activation(theta.v_o * eta[:, t-1] + theta.w_o * h[:, t-1] + theta.b_o,
                                   'sigmoid')  # Output gate
            g_f = utils.activation(theta.v_f * eta[:, t-1] + theta.w_f * h[:, t-1] + theta.b_f,
                                   'sigmoid')  # Forget gate
            memory_cell[:, t] = g_i * z_d + g_f * memory_cell[:, t-1]  # Update recurrent cell
            h[:, t] = g_o * np.tanh(memory_cell[:, t])
            eta[:, t] = theta.beta0 + theta.beta1 * h[:, t] + np.sqrt(theta.sigma2) * u_pro[:, t].T
            z[:, t] = eta[:, t] + theta.phi * z[:, t-1]

            # Calculate weights for particles at time t>=2.
            logw = -0.5 * np.log(2 * np.pi) - 0.5 * z[:, t] - 0.5 * y[t] ** 2 * np.exp(-z[:, t])

            # Numerical stabability
            weights[t, :] = np.exp(logw - np.max(logw))

            # Estimate marginal likelihood
            sir_llh = sir_llh + np.log(np.mean(weights[t, :])) + np.max(logw)

            # Normalize weigths
            weights[t, :] = weights[t, :] / np.sum(weights[t, :])

        return sir_llh
        
