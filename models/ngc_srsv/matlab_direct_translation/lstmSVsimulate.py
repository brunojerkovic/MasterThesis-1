import numpy as np
import random

def lstmSVsimulate(model, numObs, num_paths, z0, theta):
    # Set default values
    NumPaths = 1
    z0 = []
    theta = []

    # Set-up user defined settings
    NumPaths = num_paths
    z0 = z0
    theta = theta

    # Check if theta is specified and is a list, then convert theta to dict
    # with keys are model parameter names
    if theta and not isinstance(theta, dict): # TODO: struct is dict in pytho
        theta = dict(zip(model["NameParams"], theta))

    # If model parameters are not specified then use prior to generate a sample
    # of model parameters
    if not theta:
        params = model["Params"]
        num_params = model["NumParams"]
        params_name = model["NameParams"]

        for i in range(num_params):
            dist = params[params_name[i]]["prior"] # TODO: ovo popravi
            theta[params_name[i]] = random.random_generator(dist) # TODO: ovo popravi

    if not z0:
        z0 = np.zeros((NumPaths, 1))

    # Create a LSTM object
    obj_lstm = LSTM(NumPaths, numObs, theta)

    # Pre-allocation
    y = np.zeros((NumPaths, numObs))
    z = np.zeros((NumPaths, numObs))

    # For t=1
    t = 0
    z[:, t] = obj_lstm.eta[:, t] + theta["phi"]*z0
    y[:, t] = np.exp(0.5*z[:, t])*np.random.randn(NumPaths, 1)

    # Simulation from t = 2
    for t in range(1, numObs):
        obj_lstm.forward(theta, t-1)
        z[:, t] = obj_lstm.eta[:, t] + theta["phi"] @ z[:, t-1]
        y[:, t] = np.exp(0.5*z[:, t])*np.random.randn(NumPaths, 1)

    return y, z, theta
