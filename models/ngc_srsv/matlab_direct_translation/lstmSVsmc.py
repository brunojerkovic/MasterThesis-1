import numpy as np


def lstmSVsmc(mdl, N, y):
    T = len(y)
    z = np.zeros((N, T))
    h = np.zeros((N, T))
    eta = np.zeros((N, T))
    memory_cell = np.zeros((N, T))
    weights = np.zeros((N, T))
    indx = np.zeros((N, T), dtype=int)  # Store N resampling index in each of T time step

    # Sample particles at time t = 0
    t = 0

    # Calculate weights for particles at time = 0
    eta[:, t] = mdl.beta0 + np.random.normal(0, np.sqrt(mdl.sigma2), N) # TODO: mozda ova jedinica ne treba
    z[:, t] = eta[:, t] + mdl.phi * np.log(np.var(y))
    logw = -0.5 * np.log(2 * np.pi) - 0.5 * z[:, t] - 0.5 * np.power(y[t], 2) * np.exp(-z[:, t])

    # Numerical stabability
    weights[:, t] = np.exp(logw - np.max(logw))

    # Estimate marginal likelihood
    sir_llh = np.log(np.mean(weights[:, t])) + np.max(logw)

    # Normalize weigths
    weights[:, t] = weights[:, t] / np.sum(weights[:, t])

    for t in range(1, T):
        # Calculate resampling index
        indx[:, t] = utils.rs_multinomial(weights[:, t - 1])

        # Resampling
        z[:, t-1] = z[indx[:, t], t-1]
        eta[:, t-1] = eta[indx[:, t], t-1]
        h[:, t-1] = h[indx[:, t], t-1]
        memory_cell[:, t-1] = memory_cell[indx[:, t], t-1]

        # Generate particles at time t>=2.
        x_d = utils.activation(mdl.v_d @ eta[:, t-1] + mdl.w_d @ h[:, t-1] + mdl.b_d, 'Tanh')  # data input
        g_i = utils.activation(mdl.v_i @ eta[:, t-1] + mdl.w_i @ h[:, t-1] + mdl.b_i, 'Sigmoid')  # input gate
        g_o = utils.activation(mdl.v_o @ eta[:, t-1] + mdl.w_o @ h[:, t-1] + mdl.b_o, 'Sigmoid')  # output gate
        g_f = utils.activation(mdl.v_f @ eta[:, t-1] + mdl.w_f @ h[:, t-1] + mdl.b_f, 'Sigmoid') # output gate

        memory_cell[:, t] = g_i * x_d + g_f * memory_cell[:, t-1] # Update recurrent cell
        h[:, t] = g_o * np.tanh(memory_cell[:, t])

        # Calculate weights for particles at time t>=1
        eta[:, t] = mdl.beta0 + mdl.beta1 @ h[:, t] + np.random.normal(0, np.sqrt(mdl.sigma2), N)
        z[:, t] = eta[:, t] + mdl.phi @ z[:, t-1]
        logw = -0.5 * np.log(2 * np.pi) - 0.5 * z[:, t] - 0.5 * np.pow(y[t], 2) * np.exp(-z[:, t])

        # Numerical stability
        weights[:, t] = np.exp(logw - np.max(logw))

        # Estimate marginal likelihood
        sir_llh += np.log(np.mean(weights[:, t])) + np.max(logw)

        # Normalize weights
        weights[:, t] = weights[:, t] / np.sum(weights[:, t])

    return sir_llh

