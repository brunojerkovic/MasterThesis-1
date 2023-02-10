import math
import time
import random
import models.ngc_srsv.matlab_direct_translation.utils as utils
from models.ngc_srsv.matlab_direct_translation.lstmSV import LSTMSV
import numpy as np


class BPM:
    def _get_property_groups(self):
        proplist = ['model_to_fit', 'num_particles', 'num_MCMC', 'target_accept',
                    'num_covariance', 'block_size', 'save_file_name', 'save_after',
                    'verbose', 'initialize', 'seed']
        propgrp = 0 # TODO: propgrp = matlab.mixin.util.PropertyGroup
        return propgrp

    def _get_header(self, obj):
        if not isinstance(obj, (float, int)):
            header = 0 # TODO: header = getHeader@matlab.mixin.CustomDisplay(obj);
        else:
            new_header1 = 'Blocking Pseudo Marginal sampler\n'
            new_header2 = '--------------------------------\n'
            header = new_header1 + new_header2
        return header

    def __init__(self, num_particles=200, num_mcmc=100_000, target_accept=0.25,
                 num_covariance=1_000, save_filename='', save_after=5_000,
                 seed=1, block_size=5, verbose=True):
        # Set default values
        self.num_particles = num_particles
        self.num_mcmc = num_mcmc
        self.target_accept = target_accept
        self.num_covariance = num_covariance
        self.save_filename = save_filename
        self.save_after = save_after
        self.params_init = []
        self.seed = seed
        self.post = utils.DotDict({'theta': [], 'scale': [], 'time': float('nan')})
        self.initialize = 'Prior'
        self.num_block = 0
        self.block_size = block_size
        self.verbose = verbose

        if self.params_init:
            self.initialize = 'Custom'

        # Parameter values to track
        self.theta = []

    def log_prior_fnc(self, model: LSTMSV, theta):
        """
        Calculate the log prior and log jacobian of the current model.
        """
        #theta = theta if isinstance(theta, np.ndarray) else np.array(theta)
        params = model.params
        num_params = model.num_params
        params_name = model.name_params

        log_prior = 0
        log_jac = 0
        theta_trans = np.zeros((num_params,))

        for i, param in enumerate(params):
            prior_i = param.prior
            theta_i = theta[params_name[i]][0]
            param_i = param

            # Transformation for random-walk proposal
            theta_trans[i] = prior_i.transform_for_random_walk(theta_i)

            # For log-jacobian
            log_jac_offset = param_i.jacobian_offset(theta_i)
            log_jac = log_jac + prior_i.log_jacobian_random(theta_i) + log_jac_offset

            # For log-prior
            theta_i = param_i.prior_transform(theta_i)
            log_prior = log_prior + prior_i.log_pdf(theta_i)

        return log_prior, log_jac, theta_trans

    def inv_transfrom(self, model: LSTMSV, theta):
        """
        Inverse transform after random-walk proposal.
        """
        # Turn theta to numpy if it is not numpy
        theta = theta if isinstance(theta, np.ndarray) else np.array(theta)

        # Calculate the inverse of theta
        theta_inv = [p.prior.inv_transform_after_random_walk(t) for (p,t) in zip(model.params, theta)]
        theta_inv = np.array(theta_inv)

        return theta_inv

    def to_struct(self, names, theta):
        """
        Convert parameter array to struct if variable name is specified
        """
        theta_s = {}
        for n, t in zip(names, theta):
            theta_s[n] = np.array([t])
        return utils.DotDict(theta_s)

    def get_block(self, u, block_idx, N_block, type_):
        u_star = u
        n_rows, n_col = u.shape[0], u.shape[1]
        block_size = math.floor(abs(n_col / N_block))
        idx_start = (block_idx-1) * block_size + 1
        idx_stop = block_idx * block_size

        if type_ == 'normal':
            if block_idx == N_block:
                idx_stop = n_col
                u_star[:, idx_start:idx_stop+1] = np.random.normal(size=(n_rows, idx_stop-idx_start+1))
            else:
                u_star[:, idx_start:idx_stop+1] = np.random.normal(size=(n_rows, idx_stop - idx_start+1))
        elif type_ == 'uniform':
            if block_idx == N_block:
                idx_stop = n_col
                u_star[:, idx_start:idx_stop+1] = np.random.normal(size=(n_rows, idx_stop - idx_start + 1))
            else:
                u_star[:, idx_start:idx_stop+1] = np.random.normal(size=(n_rows, idx_stop - idx_start + 1))
        else:
            raise ValueError("You must specify random number type")

        return u_star


    def estimate(self, model: LSTMSV, data):
        """
        Bayesian estimation.
        """

        self.log_likelihood = model.lstmSVcorrSMC

        # Check that data is a row vector

        # Initialize parameters
        self.params_init = {p.name: p.prior_inv_transform(p.prior.random_number_generator()) for p in model.params}

        # Run BPM sampler
        self.num_block = round(len(data)) / self.block_size
        self.__bpm_estimate(model, data) # This is the main part for estimation
        model.Post = self.post # TODO: ovo zavrsi

    def __bpm_estimate(self, model: LSTMSV, y):
        """
        Bayesian estimation for LSTM-SV using BPM
        """
        # Model setting
        theta = utils.DotDict(self.params_init)

        # Static setting
        T = len(y)
        Sigma = 0.01 * np.eye(model.num_params)
        scale = 1

        # Likelihood estimation
        u_pro = np.random.normal(size=(self.num_particles, T))  # Random numbers for proposal
        u_res = np.random.uniform(size=(self.num_particles, T))  # Random numbers for resampling
        log_lik = self.log_likelihood(theta, self.num_particles, y, u_pro, u_res)

        # Prepare for first iteration
        log_prior, log_jac, theta_proposal = self.log_prior_fnc(model, theta)
        log_post = log_prior + log_lik

        # Store parameters to calculate adaptive proposal covariance matrix
        thetasave = np.zeros((self.num_mcmc, model.num_params))
        tic = time.time()

        # Prepare results for saving
        self.post.theta = np.zeros((self.num_mcmc, len(theta)))
        self.post.scale = np.zeros((self.num_mcmc))

        for i in range(self.num_mcmc):
            # Update U using random blocks
            if self.verbose:
                model.print_msg(theta, i)

            # Update U using random blocks
            block_idx = random.randint(0, self.num_block)  # Get random block index
            u_pro_star = self.get_block(u_pro, block_idx, self.num_block, 'normal')
            u_res_star = self.get_block(u_res, block_idx, self.num_block, 'uniform')

            # Propose new parameters with random walk proposal
            theta_temp = np.random.multivariate_normal(theta_proposal, scale * Sigma)
            theta_temp_inv = self.inv_transfrom(model, theta_temp)

            # Calculate acceptance probability
            theta_star = self.to_struct(model.name_params, theta_temp_inv)
            log_lik_star = self.log_likelihood(theta_star, self.num_particles, y, u_pro_star, u_res_star)
            log_prior_star, log_jac_star, theta_star_proposal = self.log_prior_fnc(model, theta_star)
            log_post_star = log_prior_star + log_lik_star
            r1 = np.exp(log_post_star - log_post + log_jac_star - log_jac)

            # Rejection decision
            A1 = np.random.uniform()  # Use this uniform random number to accept a proposal sample
            C1 = min(1, r1)

            # If accept the new proposal sample
            if A1 <= C1:
                theta_proposal = theta_star_proposal
                theta = theta_star
                log_post = log_post_star
                log_jac = log_jac_star
                u_pro = u_pro_star
                u_res = u_res_star

            # Adaptive random walk covariance
            thetasave[i, :] = theta_proposal

            # Adaptive scale for proposal distribution
            if i > 50:
                scale = utils.update_scale(scale, C1, self.target_accept, i, model.num_params)
                if i > self.num_covariance:
                    sigma = np.cov(thetasave[i - self.num_covariance + 1:i, :])
                else:
                    sigma = np.cov(thetasave[1:i, :])

            # Store the output
            self.post.theta[i, :] = np.array([t[0] for t in theta.values()])
            self.post.scale[i] = scale

            if self.save_filename != '':
                if i % self.save_after == 0:
                    # TODO: save(savename, 'obj') # Spremi object 'obj' u filename s imenom 'savename'
                    pass
        self.post.time = time.time() - tic