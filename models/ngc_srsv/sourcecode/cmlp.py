import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from models.ngc_noise.sourcecode.model_helper import activation_helper
from models.ngc_noise.utils import data_splitter


class MLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation, model_choice='ngc', logvar=None):
        super(MLP, self).__init__()
        self.num_series = num_series
        self.lag = lag
        self.activation = activation_helper(activation)
        self.logvar = logvar
        model_choice = 'ngc'
        p = 0.05
        dropouts = []

        # Set up network.
        layer = nn.Conv1d(num_series, hidden[0], lag) if model_choice == 'ngc' else nn.Conv1d(num_series, 1, lag, bias=False)
        modules = [layer]

        if model_choice == 'ngc':
            for d_in, d_out in zip(hidden, hidden[1:] + [1]):
                #dropouts.append(nn.Dropout(p=p))

                layer = nn.Conv1d(d_in, d_out, 1)
                modules.append(layer)

        # Register parameters.
        self.dropouts = nn.ModuleList(dropouts)
        self.layers = nn.ModuleList(modules)

    def forward(self, X, eps: torch.tensor, order: int):
        X = X.transpose(2, 1)
        for i, fc in enumerate(self.layers):
            if i != 0:
                #X = self.dropouts[i-1](X)
                X = self.activation(X)
            X = fc(X)

        # Reparametrization trick
        # X = X + noise*std; std = e^(0.5 * log(var)) = e^(log sqrt(var)) = sqrt(var) = L (or cholesky decomposition of L)
        X_ = X
        if self.logvar is not None:
            std = torch.exp(self.logvar) # TODO: maka san 0.5
            X_ = X + (std @ eps)[:, order, :][None, :, :]
        return X_.transpose(2, 1)

    # TODO: obrisi_me
    def forward_distribution(self, X, n_samples:int=100):
        predictions = []
        for i in range(n_samples):
            predictions.append(self(X, 0, 0).detach().cpu().numpy())

        return predictions


class cMLP(nn.Module):
    def __init__(self, num_series, lag, hidden, activation='relu', model_choice=None):
        '''
        cMLP model with one MLP per time series.

        Args:
          num_series: dimensionality of multivariate time series.
          lag: number of previous time points to use in prediction.
          hidden: list of number of hidden units per layer.
          activation: nonlinearity at each layer.
        '''
        super(cMLP, self).__init__()
        self.p = num_series
        self.lag = lag
        self.activation = activation_helper(activation)

        # Reparametrization trick params
        # TODO: mozda trebam logvar pomnozit sa 2, mozda ne triba bit cholesky??
        # self.logvar = torch.log(torch.tensor([[0.1, 0.05], [0.05, 0.1]])) #, requires_grad=True)
        # self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        #self.logvar = nn.Parameter(torch.rand((num_series, num_series)), requires_grad=True)
        self.logvar = None

        # Set up networks.
        self.networks = nn.ModuleList([
            MLP(num_series, lag, hidden, activation, model_choice, self.logvar)
            for _ in range(num_series)])

    def forward(self, X):
        '''
        Perform forward pass.

        Args:
          X: torch tensor of shape (batch, T, p).
        '''
        return torch.cat([network(X) for network in self.networks], dim=2)

    def get_variance(self) -> np.ndarray:
        if self.logvar is None:
            return np.array([[0,0], [0,0]])
        logvar_np = self.logvar.detach().cpu().numpy()
        var_np = np.exp(logvar_np)
        return var_np

    def GC(self, threshold=True, ignore_lag=True):
        '''
        Extract learned Granger causality.

        Args:
          threshold: return norm of weights, or whether norm is nonzero.
          ignore_lag: if true, calculate norm of weights jointly for all lags.

        Returns:
          GC: (p x p) or (p x p x lag) matrix. In first case, entry (i, j)
            indicates whether variable j is Granger causal of variable i. In
            second case, entry (i, j, k) indicates whether it's Granger causal
            at lag k.
        '''
        if ignore_lag:
            GC = [torch.norm(net.layers[0].weight, dim=(0, 2))
                  for net in self.networks]
        else:
            GC = [torch.norm(net.layers[0].weight, dim=0)
                  for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0.01).int()
        else:
            return GC


def prox_update(network, lam, lr, penalty):
    '''
    Perform in place proximal update on first layer weight matrix.

    Args:
      network: MLP network.
      lam: regularization parameter.
      lr: learning rate.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'GSGL':
        norm = torch.norm(W, dim=0, keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
        norm = torch.norm(W, dim=(0, 2), keepdim=True)
        W.data = ((W / torch.clamp(norm, min=(lr * lam)))
                  * torch.clamp(norm - (lr * lam), min=0.0))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        for i in range(lag):
            norm = torch.norm(W[:, :, :(i + 1)], dim=(0, 2), keepdim=True)
            W.data[:, :, :(i+1)] = (
                (W.data[:, :, :(i+1)] / torch.clamp(norm, min=(lr * lam)))
                * torch.clamp(norm - (lr * lam), min=0.0))
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def regularize(network, lam, penalty):
    '''
    Calculate regularization term for first layer weight matrix.

    Args:
      network: MLP network.
      penalty: one of GL (group lasso), GSGL (group sparse group lasso),
        H (hierarchical).
    '''
    W = network.layers[0].weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(W, dim=(0, 2)))
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(W, dim=(0, 2)))
                      + torch.sum(torch.norm(W, dim=0)))
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2)))
                          for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def ridge_regularize(network, lam):
    '''Apply ridge penalty at all subsequent layers.'''
    return lam * sum([torch.sum(fc.weight ** 2) for fc in network.layers[1:]])


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params


def train_model_ista(cmlp, X_train, X_valid, lr, max_iter, lam=0, lam_ridge=0, penalty='H',
                     lookback=5, check_every=100, verbose=1, device=None, use_valid=True):
    '''Train model with Ista.'''
    # Initial values
    lag = cmlp.lag
    N_train = X_train.shape[1]
    N_valid = X_valid.shape[1]
    p = X_train.shape[-1]
    train_loss_list, valid_loss_list = [], []

    # Definitions
    loss_fn = nn.MSELoss(reduction='mean')
    cov_mat = torch.tensor([[1., 0.], [0., 1.]])
    mean = torch.zeros(2)
    mvn = torch.distributions.MultivariateNormal(mean, cov_mat)
    eps_train = mvn.sample((N_train-lag,)).T[None, :, :]#.cuda() # TODO: cuda warning here
    eps_valid = mvn.sample((N_valid-lag,)).T[None, :, :]#.cuda() # TODO: cuda warning here
    # eps = torch.randn((p, N_train-p)).to(device)

    # For early stopping.
    best_it_train, best_it_valid = None, None
    best_loss_train, best_loss_valid = np.inf, np.inf
    best_model = None

    # Calculate smooth error.
    loss = sum([loss_fn(cmlp.networks[i](X_train[:, :-1], eps_train, i), X_train[:, lag:, i:i+1]) for i in range(p)])

    ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
    smooth = loss + ridge

    for it in range(max_iter):
        # Take gradient step. (all of the layers of the model)
        smooth.backward()
        for param in cmlp.parameters():
            param.data = param - lr * param.grad

        # Take prox step. (only first layer of the model)
        if lam > 0:
            for net in cmlp.networks:
                prox_update(net, lam, lr, penalty)

        cmlp.zero_grad()

        # Calculate smoooth loss for next iteration.
        loss = sum([loss_fn(cmlp.networks[i](X_train[:, :-1], eps_train, i), X_train[:, lag:, i:i + 1]) for i in range(p)])

        # Calculate smooth_loss = mse + ridge (all other than 1st layer)
        ridge = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
        smooth = loss + ridge

        # Check progress.
        if (it + 1) % check_every == 0:
            # Add nonsmooth penalty. (only first layer)
            nonsmooth = sum([regularize(net, lam, penalty)
                             for net in cmlp.networks])
            mean_loss = (smooth + nonsmooth) / p
            train_loss_list.append(mean_loss.detach())

            # Check for early stopping. (VALIDATION SET)
            if use_valid:
                loss_valid = sum([loss_fn(cmlp.networks[i](X_valid[:, :-1], eps_valid, i), X_valid[:, lag:, i:i + 1]) for i in range(p)])
                ridge_loss_valid = sum([ridge_regularize(net, lam_ridge) for net in cmlp.networks])
                nonsmooth_valid = sum([regularize(net, lam, penalty) for net in cmlp.networks])
                mean_loss = (loss_valid + ridge_loss_valid + nonsmooth_valid) / p
                valid_loss_list.append(mean_loss.detach())

            # Tracking progress
            if verbose > 0:
                print(('-' * 10 + 'Iter = %d' + '-' * 10) % (it + 1))
                print('Train Loss = %f' % train_loss_list[-1])
                if use_valid:
                    print('Validation loss = %f' % valid_loss_list[-1])
                #print('Variable usage = %.2f%%'
                #      % (100 * torch.mean(cmlp.GC().float())))

            if mean_loss < best_loss_valid:
                best_loss_valid = mean_loss
                best_it_valid = it
                best_model = deepcopy(cmlp)
            elif best_it_valid is not None and (it - best_it_valid) == lookback * check_every:
                if verbose:
                    print('Stopping early')
                break


    # Restore best model.
    restore_parameters(cmlp, best_model)

    return train_loss_list, valid_loss_list

def test_model(cmlp, X_test):
    '''Train model with Ista.'''
    # Initial values
    lag = cmlp.lag
    N_test = X_test.shape[1]
    p = X_test.shape[-1]

    # Definitions
    loss_fn = nn.MSELoss(reduction='mean')
    eps_test = torch.distributions.MultivariateNormal(torch.zeros(2), torch.tensor([[1., 0.], [0., 1.]])).sample((N_test-lag,)).T[None, :, :]#.cuda() # TODO: cuda warning here
    # eps = torch.randn((p, N_train-p)).to(device)

    # Calculate smooth error.
    test_loss = sum([loss_fn(cmlp.networks[i](X_test[:, :-1], eps_test, i), X_test[:, lag:, i:i+1]) for i in range(p)])

    return test_loss.item()
