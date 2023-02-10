import numpy as np


class IS2:
    # Class to perform marginal likelihood estimation using IS2
    # TODO: add handles
    def __init__(self, model='lstmSV', num_particle=2_000, num_is_particle=5_000, proposal='@IS2proposal',
                 log_proposal='@IS2LogProposalPdf', random_generator='@IS2RandomGenerator',
                 burnin=0, marllh=0, std_marllh=0, seed=1):
        self.num_particle = num_particle
        self.num_is_particle = num_is_particle
        self.proposal = proposal
        self.log_proposal = log_proposal
        self.random_generator = random_generator
        self.burnin = burnin
        self.marllh = marllh
        self.std_marllh = std_marllh
        self.seed = seed

        if model == 'lstmSV':
            self.log_likelihood = '@lstmSVsmc'
        elif model == 'SV':
            self.log_likelihood = '@SVsmc'

        # TODO: translate down
        # Set user-specified settings
        # if nargin > 2
        #    obj = obj.setParams(obj, varargin);
        # end

        # Set random seed if specified
        # if(~isnan(obj.Seed))
        #     rng(obj.Seed);
        # end

        # Run IS2 algortihm
        self.marllh, self.st_marllh = IS2fit(self, model)

    # Calculate the log prior and log Jacobian of the current model
    def log_prior(self, model, theta):
        params = model.params
        num_params = model.num_params
        params_name = model.name_params
        log_prior = 0
        log_jac = 0

        for i in range(num_params):
            prior_i = getattr(params, params_name[i]).prior
            theta_i = getattr(theta, params_name[i])
            param_i = getattr(params, params[i])

            # For log-Jacobian
            log_jac_offset = param_i.JacobianOffset(theta_i)
            log_jac += prior_i.logJacobianRandomFnc(theta_i) + log_jac_offset

            # For log-prior
            theta_i = param_i.PriorTransform(theta_i)
            log_prior += prior_i.logPdfFnc(theta_i)

        return log_prior, log_jac

    # Calculate the log prior and log Jacobian of the current model
    # Theta is a matrix whose columns are mcmc samples of the corresponding parameters
    def transform(self, model, theta):
        params = model.params
        num_params = model.num_params
        params_name = model.name_params
        theta_trans = np.zeros(len(theta))

        for i in range(num_params):
            prior_i = getattr(params, params_name[i]).prior
            theta_i = theta[:, i]

            # Transformation for random-walk proposal
            theta_trans[:, i] = prior_i.transformForRandomWalkFnc(theta_i)

        return theta_trans

    # Inverse transform after random walk proposal
    # theta is output from random-walk proposal
    def inv_transform(self, model, theta):
        params = model.params
        num_params = model.num_params
        params_name = model.params_name
        theta_inv = np.zeros(shape=(num_params, 1))

        for i in range(num_params):
            prior_i = getattr(params, params_name[i]).prior
            theta_inv[i] = prior_i.invTransformAfterRandomWalkFnc(theta[i])

        return theta_inv

    # Convert parameter array to strcut if variable name is specifcied
    def to_struct(self, var_name, theta):
        for i in range(var_name):
            setattr(theta, var_name[i], theta[i])  # TODO: theta_struct.(var_name{i}) = theta(i);
        return theta









