import numpy as np
from models.ngc_srsv.matlab_direct_translation.utils import *
from models.ngc_srsv.matlab_direct_translation.lstmSV import LSTMSV


class ParticleFilter:
    """
    A class to implement particle filter to estimate likelihood of a stochastic volatility model
    """
    def __init__(self, model: LSTMSV, data, num_particles=200, num_state_variables = 1, resampling_method='multinomial',
                 state_estimation_method='mean'):
        self.num_state_variables = num_state_variables
        self.num_particles = num_particles
        self.num_timesteps = len(data)
        self.resampling_method = resampling_method
        self.resampling_function = rs_multinomial  #TODO: obj.ResamplingFnc = @utils.rs_multinomial;
        self.state_estimation_method = state_estimation_method
        self.data = data
        self.model = model

        # Assign state transition function of the model
        # TODO: (isempty(obj.StateTransition) && ~isempty(Model.StateTransitionFcn))
        #if len(self.state_transition) == 0 and not len(model):
        #    self.state_transition = model.state_transition_fnc
        #else:
        #    raise ValueError("A handle of state transition must be specified")

        # TODO: prepisi ovo dolje
        # Assign measurement function of the model
        # if(isempty(obj.MeasurementLikelihood) && ~isempty(Model.MeasurementFcn))
        #     obj.MeasurementLikelihood = Model.MeasurementFcn;
        # else
        #     disp('A handle of measurement equation must be specified!')
        # end

    def state_proposal_fnc(self, old_state):
        """
        Equation to propose state particles in the next time step
        """
        theta = {p.name: p.prior.random_number_generator() for p in self.model.params_SV}
        # old_state = self.state[:, self.current_time]
        random_number = np.random.normal(1, self.num_particles)
        proposed_state = self.model.SVstateTransitionFnc(theta, old_state, random_number)

        return proposed_state

    def estimate(self):
        """
        Run a particle filter
        :param self: A particle filter object storing specifications to run a particle filter
        :param model: The model to run the particle filter
        :return sir_llh: The estimate of log-likelihood of the input model
        """
        # Setting
        N = self.num_particles
        T = self.num_timesteps
        P = self.model.num_params
        y = self.data

        # Preallocation
        self.state = np.zeros((N, T))
        self.weights = np.zeros((N, T))
        self.ancestor_index = np.zeros((N, T))

        # Sample particles at time t=1
        t = 0
        self.current_time = t
        self.state[:, t] = self.model.state_initialize(num_particle=N)

        # Calculate weights for particles at time t=1
        logw = self.model.SVmeasurementFnc(self.state[:, t], y[t])

        # Numerical stability
        self.weights[:, t] = np.exp(logw - np.max(logw))

        # Estimate marginal likelihood
        sir_llh = np.log(np.mean(self.weights[:, t])) + np.max(logw)

        # Normalize weights
        self.weights[:, t] = self.weights[:, t] / np.sum(self.weights[:, t])

        for t in range(1, T):
            self.current_time = t

            # Resampling
            self.ancestor_index[:, t] = self.resampling_function(self.weights[:, t - 1])
            self.state[:, t] = self.state_proposal_fnc(self.state[:, t-1])

            # Calculate weights of particles at the current time-step
            logw = self.model.SVmeasurementFnc(self.state[:, t], y[t])

            # Numerical stability
            self.weights[:, t] = np.exp(logw - np.max(logw))

            # Estimate marginal likelihood
            sir_llh = sir_llh + np.log(np.mean(self.weights[:, t])) + np.max(logw)

            # Normalize weights
            self.weights[:, t] = self.weights[:, t] / np.sum(self.weights[:, t])

        return sir_llh
