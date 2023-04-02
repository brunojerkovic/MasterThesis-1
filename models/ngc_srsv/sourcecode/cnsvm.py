import time
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from models.ngc_noise.sourcecode.model_helper import activation_helper
from models.ngc_noise.utils import data_splitter
from models.ngc_srsv.neural_SVM.early_stopper import EarlyStopper

device_global = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_forecast(model, train_loader, valid_loader, p):
    inputs_pr_train, outputs_pr_train = [], []
    inputs_pr_valid, outputs_pr_valid = [], []

    for i_batch, (x_prev, _) in enumerate(train_loader):
        # Forward prop
        x_prev = x_prev[0]
        x_new, _, _, _, _, _, _ = model(x_prev)

        outputs_pr_train += [x_new.squeeze().detach().cpu().numpy().tolist()]
        if i_batch == 0:
            i = x_prev.squeeze().detach().cpu().numpy().reshape((-1, p)).tolist()
        else:
            i = [x_prev.squeeze().detach().cpu().numpy().tolist()[-p:]]
        inputs_pr_train += i

    # region VALIDATION
    for i_batch, (x_prev, _) in enumerate(valid_loader):
        # Forward prop
        x_prev = x_prev[0]
        x_new, _, _, _, _, _, _ = model(x_prev)

        # Save info
        outputs_pr_valid += [x_new.squeeze().detach().cpu().numpy().tolist()]
        if i_batch == 0:
            i = x_prev.squeeze().detach().cpu().numpy().reshape((-1, p)).tolist()
        else:
            i = [x_prev.squeeze().detach().cpu().numpy().tolist()[-p:]]
        inputs_pr_valid += i

    return inputs_pr_train, inputs_pr_valid, outputs_pr_train, outputs_pr_valid


# TODO: (later) makni cudu iz unutarnjih i cuda-iraj vanka na razini cijelog NSVM-a
# My MLP
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=10, S=10):
        super(MLP, self).__init__()
        self.n_samples = S
        self.dim = output_size

        self.input_mlp = nn.Linear(input_size, hidden_size)
        self.mean_mlp = nn.Linear(hidden_size, output_size)
        self.log_var_mlp = nn.Linear(hidden_size, int(output_size*(output_size+1)/2))
        # self.log_var_mlp = nn.Linear(hidden_size, int(output_size*2))

    def reparametrization(self, mean, var):
        # var = torch.exp(0.5 * log_var)
        var = torch.linalg.cholesky(var)
        samples = []
        for _ in range(self.n_samples):
            epsilon = torch.randn_like(mean)
            sample = mean + epsilon @ var # TODO: (later) popravi dimenzije ode
            samples.append(sample)
        return samples

    def get_var_shape(self, vals):
        m = torch.zeros((self.dim, self.dim), dtype=vals.dtype, device=device_global)
        xs, ys = torch.triu_indices(*m.shape, device=device_global)
        m[xs, ys] = vals[0]
        m[ys, xs] = vals[0]
        cov_matrix = torch.mm(m, m.t())
        cov_matrix += 1e-6 * torch.eye(self.dim, device=vals.device)
        # m = m + torch.eye(m.size(0), device=vals.device)
        # m = torch.linalg.cholesky(m, upper=False)
        # cov_matrix = torch.mm(m, m.t())
        # if not bool((cov_matrix == cov_matrix.T).all() and (torch.eig(cov_matrix)[0][:,0]>=0).all()):
        #     print("Not PSD!")
        return cov_matrix

    # TODO: ti koristis ove dvi doje funkcije
    def get_var_shape__(self, vals):
        m = torch.zeros((self.dim, self.dim), dtype=vals.dtype)
        xs, ys = torch.triu_indices(*m.shape)
        m[xs, ys] = vals[0]
        m[ys, xs] = vals[0]
        m = torch.mm(m, m.t())
        m = m + torch.eye(m.size(0), device=vals.device)
        eigvals, eigvecs = torch.symeig(m, eigenvectors=True)  # compute eigendecomposition
        eigvals = torch.clamp(eigvals, min=0)  # set negative eigenvalues to zero
        cov_matrix = torch.mm(eigvecs, torch.mm(torch.diag(eigvals), eigvecs.t()))  # reconstruct covariance matrix
        if not bool((cov_matrix == cov_matrix.T).all() and (torch.linalg.eig(cov_matrix)[0][:,0]>=0).all()):
            print("Not PSD!")
            print("This is the cov matrix")
            print(cov_matrix)
        return cov_matrix

    # TODO: ovo dolje je mozda ok, ali u drugom kodu je loss prevelik
    def get_var_shape_(self, vals):
        d, V = vals[:, :self.dim], vals[:, self.dim:]
        D = torch.eye(d.shape[1], device=vals.device)
        A = torch.linalg.inv(D)
        for k in range(1, V.shape[1]):
            gamma = (V @ A @ A.T @ V.T)[0,0]
            eta = 1 / (1 + gamma)
            A = A - ((1-torch.sqrt(eta)) / gamma) * A @ A.T @ V.T @ V @ A
        cov_matrix = A @ A.T

        # if not bool((cov_matrix == cov_matrix.T).all() and (torch.linalg.eig(cov_matrix)[0][:,0]>=0).all()):
        #    print("Not PSD!")
        #    print("This is the cov matrix")
        #    print(cov_matrix)
        return cov_matrix

    def get_var_shape__(self, vals):
        m = torch.zeros((self.dim, self.dim), dtype=vals.dtype)
        xs, ys = torch.triu_indices(*m.shape)
        m[xs, ys] = vals[0]
        m[ys, xs] = vals[0]
        m = torch.mm(m, m.t())
        m = m + torch.eye(m.size(0), device=vals.device)
        m = torch.linalg.cholesky(m, upper=False)
        cov_matrix = torch.mm(m, m.t())
        if not bool((cov_matrix == cov_matrix.T).all() and (torch.linalg.eig(cov_matrix)[0][:,0]>=0).all()):
            print("Not PSD!")
            print("This is the cov matrix")
            print(cov_matrix)
        return cov_matrix

    def get_var_shape_(self, cov):
        L = torch.zeros((self.dim, self.dim), device=cov.device)
        x_positions, y_positions = torch.tril_indices(row=self.dim, col=self.dim)
        L[x_positions, y_positions] = cov
        # L[y_positions, x_positions] = cov
        cov_matrix = L @ L.T
        # cov_matrix = torch.matmul(L, L.transpose(-1, -2))
        # cov_matrix += 1e-6 * torch.eye(self.dim, device=cov.device)
        if not bool((cov_matrix == cov_matrix.T).all() and (torch.linalg.eig(cov_matrix)[0][:,0] >= 0).all()):
            print("Not PSD!")
        return cov_matrix

    def get_var_shape__(self, vals):
        m = torch.zeros((self.dim, self.dim), dtype=vals.dtype)
        xs, ys = torch.triu_indices(*m.shape)
        m[xs, ys] = vals[0]
        m[ys, xs] = vals[0]
        m = torch.matrix_exp(m)
        m = m + 1e-6 * torch.eye(m.shape[0], device=vals.device)
        m = (m + m.T) / 2
        if bool((m == m.T).all() and (torch.eig(m)[0][:, 0] >= 0).all()):
            print("Not PSD!")
        return m

    def forward(self, x):
        h = self.input_mlp(x)
        mean = self.mean_mlp(h)
        var = self.get_var_shape(self.log_var_mlp(h))
        samples = None # self.reparametrization(mean, var)

        return samples, mean, var


# Inference model
class InferenceModel(nn.Module):
    def __init__(self, p, hidden_dim, latent_dim, dropout=0.2, seq_len=3, S=10):
        super(InferenceModel, self).__init__()
        self.S = S
        self.p, self.z_dim, self.h_dim = p, latent_dim, hidden_dim
        self.seq_len = seq_len

        self.gru_left = nn.GRU(input_size=p, hidden_size=hidden_dim, dropout=dropout)
        self.gru_right = nn.GRU(input_size=p, hidden_size=hidden_dim, dropout=dropout)
        self.gru = nn.GRU(input_size=latent_dim + hidden_dim * 2, hidden_size=hidden_dim, dropout=dropout)

        self.mlp = MLP(input_size=hidden_dim, output_size=latent_dim, hidden_size=hidden_dim, S=S)

    def forward(self, x_left_right, z_old):
        """
        :param x: x_(t)
        :param z: z_(t-1)
        :return: z_t
        """
        # Flip tensors
        x_left, x_right = torch.split(x_left_right, (self.seq_len, self.seq_len), dim=0)
        x_right_rev = torch.flip(x_right, dims=[0])

        self.gru_left.flatten_parameters()
        self.gru_right.flatten_parameters()
        self.gru.flatten_parameters()

        # Pass
        _, h_left = self.gru_left(x_right_rev)
        _, h_right = self.gru_right(x_left)
        _, h = self.gru(torch.cat([z_old, h_left, h_right], dim=1))
        z_samples, z_mu, z_var = self.mlp(h)

        return z_samples, z_mu, z_var


# Generative model
class GenerativeModelX(nn.Module):
    def __init__(self, p=2, hidden_dim=10, latent_dim=4, dropout=0.2, S=100):
        super(GenerativeModelX, self).__init__()

        self.gru = nn.GRU(input_size=p + latent_dim, hidden_size=hidden_dim, dropout=dropout)
        self.mlp = MLP(input_size=hidden_dim, output_size=1, hidden_size=hidden_dim, S=S)

    def forward(self, x_old, z):
        """
        p_Phi (x_t | x_(t-1), z_t )
        :param x: x_(t-1)
        :param z: z_(t)
        :return:
        """
        self.gru.flatten_parameters()

        n_repetitions = x_old.shape[0]
        z = z.repeat(n_repetitions, 1)
        _, h = self.gru(torch.concat([x_old, z], dim=1))  # TODO: (later) umjesto da saljes z (i tako S puta), salji mu_z, log_var_z
        x_sample, x_mean, x_var = self.mlp(h)

        return x_sample, x_mean, x_var


class GenerativeModelZ(nn.Module):
    def __init__(self, hidden_dim=4, latent_dim=10, dropout=0.2):
        super(GenerativeModelZ, self).__init__()

        self.gru_z = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, dropout=dropout)
        self.mlp_z = MLP(input_size=hidden_dim, output_size=latent_dim)

    def forward(self, z_old):
        """
        p_Phi (z_t | z_(t-1) )
        :param z_old: z_(t-1)
        :return: z_t
        """
        self.gru_z.flatten_parameters()

        _, h_z = self.gru_z(z_old)
        z, z_mu, z_var = self.mlp_z(h_z)

        return z, z_mu, z_var


class JoinNet(nn.Module):
    def __init__(self, p):
        super(JoinNet, self).__init__()
        self.cnn = nn.Conv1d(p, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(1, 0)
        x = self.cnn(x)
        x = self.relu(x)
        x = x.permute(1, 0)
        return x


class NSVM(nn.Module):
    def __init__(self, p=2, hidden_dim=4, latent_dim=10, S=50, seq_len=3, device=None):
        super(NSVM, self).__init__()
        self.S = S
        self.seq_len = seq_len

        self.join_net = JoinNet(p=p).to(device)
        self.inference_net = InferenceModel(p=1, hidden_dim=hidden_dim, latent_dim=latent_dim, seq_len=seq_len, S=S).to(device)
        self.genx_net = GenerativeModelX(p=1, hidden_dim=hidden_dim, latent_dim=latent_dim, S=S).to(device)
        #self.genz_net = GenerativeModelZ(hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

        rnd = torch.randn(size=(1, latent_dim))
        self.z_old = nn.Parameter(rnd, requires_grad=True).to(device)
        self.z_buffer = self.z_old.data.clone().detach()

    def forward(self, x):
        """
        :param x: x_(t-1)
        :return: x_t
        """
        self.z_old.data = self.z_buffer.data
        x = self.join_net(x)

        x_old, _ = torch.split(x, (self.seq_len, self.seq_len), dim=0)

        # Inference net
        z_samples, mutilda_z, sigmatilda_z = self.inference_net(x, self.z_old)
        z_new = mutilda_z

        # GenX net
        _, mu_x, sigma_x = self.genx_net(x_old, z_new)
        x_new = mu_x

        # GenZ net
        #_, mu_z, sigma_z = self.genz_net(self.z_old)

        self.z_buffer = z_new.detach().data
        #self.z_new = nn.Parameter(z_new, requires_grad=True)
        return x_new, mutilda_z, sigmatilda_z, mu_x, sigma_x, None, None #, mu_z, sigma_z

    def load_model(self):
        self.load_state_dict(torch.load('./nsvm_weights.pth'))
    def save_model(self):
        torch.save(self.state_dict(), './nsvm_weights.pth')


class cNSVM(nn.Module):
    def __init__(self, p, hidden_dim, latent_dim, seq_len, device):
        super(cNSVM, self).__init__()
        self.lag = 1

        self.networks = nn.ModuleList([
            NSVM(p=p, hidden_dim=hidden_dim, latent_dim=latent_dim, seq_len=seq_len, device=device)
        for _ in range(p)])

    def forward(self, X):
        '''
        Perform a forward pass
        :return: X tensor of shape (T, p)
        '''
        outputs = []
        for network in self.networks:
            outputs.append(network(X))
        return outputs
        # return torch.cat([network(X) for network in self.networks], dim=1)

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
        GC = [torch.norm(net.join_net.cnn.weight, dim=0) for net in self.networks]
        GC = torch.stack(GC)
        if threshold:
            return (GC > 0.1).int()
        else:
            return GC

    def load_model(self):
        self.load_state_dict(torch.load('./c_nsvm_weights.pth'))
    def save_model(self):
        torch.save(self.state_dict(), './c_nsvm_weights.pth')


class ElboLossUnit(nn.Module):
    def __init__(self):
        super(ElboLossUnit, self).__init__()
        self.loss_mse = nn.MSELoss(reduction='mean').to(device_global)

    def forward(self, params, x_gt):
        cum_loss = 0
        params = [params] if not isinstance(params, list) else params

        for var_idx, (x_hat, mutilda_z, sigmatilda_z, mu_x, sigma_x, mu_z, sigma_z) in enumerate(params):
            rec_loss = self.loss_mse(x_hat, x_gt[:, var_idx:(var_idx+1)].detach()) # TODO: provjeri ovo (greska!)
            #d1 = torch.distributions.MultivariateNormal(loc=mutilda_z, covariance_matrix=sigmatilda_z)
            #d2 = torch.distributions.MultivariateNormal(loc=torch.zeros(4, device=device_global),
            #                                            covariance_matrix=torch.eye(4, device=device_global))
            #kl_loss = torch.distributions.kl_divergence(d1, d2)[0]
            cum_loss += rec_loss #+ kl_loss

        return cum_loss


class ElboLossZ(nn.Module):
    def __init__(self):
        super(ElboLossZ, self).__init__()
        self.loss_mse = nn.MSELoss()

    def forward(self, params, x_hat, x):
        N = len(params)
        cum_loss = 0

        for (x, mutilda_z, sigmatilda_z, mu_x, sigma_x, mu_z, sigma_z) in params:
            epstilda_z = torch.randn_like(mutilda_z)

            part1 = torch.log(torch.det(sigma_z))
            part2 = (mutilda_z + epstilda_z @ sigmatilda_z - mu_z) @ torch.linalg.inv(sigma_z) @ (
                        mutilda_z + epstilda_z @ sigmatilda_z - mu_z).T
            part3 = torch.log(torch.det(sigma_x))
            part4 = (x - mu_x).T @ torch.linalg.inv(sigma_x) @ (x - mu_x)
            part5 = torch.log(torch.det(sigmatilda_z))

            cum_loss += part1 + part2[0, 0] + part3 + part4[0, 0] - part5

        return 1 / (2 * N) * cum_loss


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
    W = network.join_net.cnn.weight  # There is also 'bias' in join_net, but we do not want to use prox update on it
    # W = network.layers[0].weight
    hidden, p, lag = W.shape
    kernel_size = lag
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
        for i in range(kernel_size):
            norm = torch.norm(W[:, :, :(i + 1)], dim=(0, 2), keepdim=True, p='fro')
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
    W = network.join_net.cnn.weight
    hidden, p, lag = W.shape
    if penalty == 'GL':
        return lam * torch.sum(torch.norm(W, dim=(0, 2))) # TODO: ovo nije promjenjeno
    elif penalty == 'GSGL':
        return lam * (torch.sum(torch.norm(W, dim=(0, 2)))
                      + torch.sum(torch.norm(W, dim=0))) # TODO: ovo nije promjenjeno
    elif penalty == 'H':
        # Lowest indices along third axis touch most lagged values.
        return torch.sum(torch.abs(W)).item()
        #return sum([torch.norm(p).item() for p in network.parameters()][:2])
        #return lam * sum([torch.sum(torch.norm(W[:, :, :(i+1)], dim=(0, 2)))
        #                  for i in range(lag)])
    else:
        raise ValueError('unsupported penalty: %s' % penalty)


def ridge_regularize(network, lam):
    '''Apply ridge penalty at all subsequent layers.'''
    return lam * sum([torch.sum(p**2) for p in network.parameters()][2:])
    # return lam * sum([torch.sum(param**2).item() for m in list(model.named_modules())[2:] for param in m[1].parameters()])
    # return lam * sum([torch.sum(fc.weight ** 2) for fc in network.layers[1:]])


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for params, best_params in zip(model.parameters(), best_model.parameters()):
        params.data = best_params

def model_predict(model, data_loader, loss_fn):
    loss = None
    # Calculate smoooth loss for next iteration.
    for i_batch, (x_prev, x_next) in enumerate(data_loader):
        # Forward prop
        x_prev, x_next = x_prev[0], x_next[0]
        outputs = model(x_prev)

        if i_batch == 0:
            loss = loss_fn(outputs, x_next)
        else:
            loss += loss_fn(outputs, x_next)

    return loss

# TODO: promijeni train funkciju jos (sta si na #)
def train_model_ista(model, train_loader, valid_loader, lr, max_iter, lam=0.1, lam_ridge=0.1, penalty='H',
                    verbose=1, device=None):
    '''Train model with Ista.'''
    # Initial values
    lag = model.lag
    N_train = len(train_loader.dataset)
    N_valid = len(valid_loader.dataset)
    p = train_loader.dataset[0][0].shape[-1]
    train_loss_list, valid_loss_list = [], []

    # Definitions
    loss_fn = ElboLossUnit()
    loss_mse = nn.MSELoss().to(device_global)
    early_stopper = EarlyStopper(patience=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Calculate smoothing error (error of output + L2 on non-conv layers)
    first_sample_x, first_sample_y = next(iter(train_loader))
    first_sample_x, first_sample_y = first_sample_x[0], first_sample_y[0]
    loss = sum([loss_fn(model.networks[i](first_sample_x), first_sample_y) for i in range(p)])
    ridge = sum([ridge_regularize(net, lam_ridge) for net in model.networks])
    smooth = loss + ridge

    times = [time.time()]

    prev_time = time.time()
    for epoch in range(max_iter):
        print("TIME", (time.time() - prev_time)*50)
        prev_time = time.time()
        # Take gradient step. (all of the layers of the model)
        #smooth.backward()
        #optimizer.step()


        #optimizer.zero_grad()


        # TRAIN REGION
        #times = [time.time()]
        # Calculate smooth_loss = mse (+elbo) + ridge (all other than 1st layer)
        #times.append(time.time())
        #print(times[-1] - times[-2], 1)

        train_loss = model_predict(model, train_loader, loss_fn) # MODEL PREDICTION ON TRAIN SET
        #times.append(time.time())
        #print(times[-1] - times[-2], 2)

        ridge = sum([ridge_regularize(net, lam_ridge) for net in model.networks])
        smooth = train_loss + ridge
        #times.append(time.time())
        #print(times[-1] - times[-2], 3)

        # Add nonsmooth penalty. (only first layer)
        nonsmooth = sum([regularize(net, lam, penalty) for net in model.networks])
        mean_loss = (smooth + nonsmooth) / p
        train_loss_list.append(mean_loss.detach())
        #times.append(time.time())
        #print(times[-1] - times[-2], 4)

        mean_loss.backward()
        #times.append(time.time())
        #print(times[-1] - times[-2], 5)

        optimizer.step()
        # Take prox step. (only first layer of the model)
        if lam > 0:
            for net in model.networks:
                prox_update(net, lam, lr, penalty)
        #times.append(time.time())
        #print(times[-1] - times[-2], 6)


        optimizer.zero_grad()
        #times.append(time.time())
        #print(times[-1] - times[-2], 7)

        # VALIDATION REGION
        valid_loss = model_predict(model, valid_loader, loss_fn) # MODEL PREDICTION ON VALID SET
        #times.append(time.time())
        #print(times[-1] - times[-2], 8)

        ridge_loss_valid = sum([ridge_regularize(net, lam_ridge) for net in model.networks])
        #times.append(time.time())
        #print(times[-1] - times[-2], 9)

        nonsmooth_valid = sum([regularize(net, lam, penalty) for net in model.networks])
        #times.append(time.time())
        #print(times[-1] - times[-2], 10)

        mean_loss = (valid_loss + ridge_loss_valid + nonsmooth_valid) / p
        valid_loss_list.append(mean_loss.detach())

        # Print progress
        if verbose:
            print(f"{'-'*10} Iter={epoch+1} {'-'*10}")
            print(f"Train loss = {train_loss_list[-1]}")
            print(f"Validation loss = {valid_loss_list[-1]}")
            print(f"Variable usage = {100 * torch.mean(model.GC().float())}%")
            print("Weights: ", np.array([net.join_net.cnn.weight.detach().cpu().numpy() for net in model.networks]))
            times.append(time.time())
            print("Time", times[epoch+1]-times[epoch])

        # Check for early stopping
        es_flag = early_stopper.early_stop(valid_loss_list[-1])
        #times.append(time.time())
        #print(times[-1] - times[-2], 11)

        if es_flag == EarlyStopper.PATIENCE_ENDED:
            model.load_model()
            print("Early stopping!")
            break
        elif es_flag == EarlyStopper.EVERYTHING_OK:
            model.save_model()
        else:
            print(f"Patience: {early_stopper.counter}")

    # restore_parameters(model, best_model)
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
