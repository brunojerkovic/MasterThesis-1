import numpy as np


def IS2_fit(obj, model):
    y = obj.data
    burnin = obj.burnin
    M = obj.num_is_particle
    N = obj.num_particle
    theta_mcmc = model.Post.theta[burnin+1:, :]

    # Convert all parameters to real value
    theta_proposal = obj.transform(model, theta_mcmc)

    # Fit a proposal model
    mdl_proposal = obj.proposal(theta_proposal, 'GM')

    log_w = np.zeros(shape=(M, 1))

    # For each sample of theta
    print('Starting...\n')

    # TODO: 'parfor' is used down
    for i in range(M):
        print(f'Iteration: {i}')

        # Transform the proposed theta to parameter space
        theta_inv = obj.inv_transform(model, theta)  # TODO: I think that 'theta_mcmc' goes instead of 'theta'

        # Convert vector of params to a struct
        theta_struct = obj.toStruct(model.NameParams, theta_inv)

        # Estimate log-likelihood using particle filter
        log_lik = obj.LogLikelihood(theta_struct, N, y)

        # Calculate log-prior contribution
        log_prior, log_jac = obj.logPrior(model, theta_struct)

        # Calculate log of proposal density contribution
        proposal_log_density = obj.LogProposal(mdl_proposal, theta)  # TODO: I think that 'theta_mcmc' goes instead of theta

        # Calculate weight for the current proposed sample of theta
        log_w[i] = log_prior + log_jac + log_lik - proposal_log_density

    # Numerical stability
    max_lw = max(log_w)
    weights = np.exp(log_w - max_lw)

    # Estimate log of marginal likelihood and its variance (using Delta method)
    llh = np.log(np.mean(weights)) + max_lw
    variance_llh = (np.mean(np.exp(2 * (log_w - max_lw))) / (np.mean(weights)) ^ 2 - 1) / len(y)
    std_llh = np.sqrt(variance_llh);

    print('End!')

