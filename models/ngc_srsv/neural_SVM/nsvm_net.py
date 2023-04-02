import math
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.optim import Adam, RMSprop
from torch.utils.data import TensorDataset, DataLoader
from models.ngc_srsv.neural_SVM.early_stopper import EarlyStopper
from matplotlib import pyplot as plt # TODO: makni odavde


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
            i = x_prev.squeeze().detach().cpu().numpy().tolist()[-1:]
            i = [i] if not isinstance(i[0], list) else i
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
            i = x_prev.squeeze().detach().cpu().numpy().tolist()[-1:]
            i = [i] if not isinstance(i[0], list) else i
        inputs_pr_valid += i

    return inputs_pr_train, inputs_pr_valid, outputs_pr_train, outputs_pr_valid

class GradientAmplifier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.scale * grad_output, None

def amplify_gradients(tensor, scale):
    return GradientAmplifier.apply(tensor, scale)

# TODO: (later) makni cudu iz unutarnjih i cuda-iraj vanka na razini cijelog NSVM-a
# My MLP
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=10, S=10):
        super(MLP, self).__init__()
        self.n_samples = S = 1
        self.dim = output_size

        self.input_mlp = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.mean_mlp = nn.Linear(hidden_size, output_size)
        self.log_var_mlp = nn.Linear(hidden_size, int(output_size*(output_size+1)/2))
        # self.log_var_mlp = nn.Linear(hidden_size, int(output_size*2))

    def reparametrization(self, mean, log_var):
        # var = torch.exp(0.5 * log_var)
        var = torch.linalg.cholesky(log_var)
        # var = log_var
        samples = []
        for _ in range(self.n_samples):
            epsilon = torch.randn_like(mean).cuda()
            sample = mean + epsilon @ var
            samples.append(sample)
        return samples

    def get_var_shape(self, vals):
        m = torch.zeros((self.dim, self.dim), dtype=vals.dtype).cuda()
        xs, ys = torch.triu_indices(*m.shape)
        m[xs, ys] = vals[0]
        m[ys, xs] = vals[0]
        m = torch.matrix_exp(m)
        # m = m + 1e-6 * torch.eye(m.shape[0], device=vals.device)
        # m = torch.matmul(m, m.transpose(-1, -2))
        #m = torch.exp(0.5 * m)
        # cov_matrix += 1e-6 * torch.eye(self.dim, device=cov.device)
        return m

    def get_var_shape__(self, vals):
        cov = vals
        L = torch.zeros((self.dim, self.dim), device=cov.device)
        tril_indices = torch.tril_indices(row=self.dim, col=self.dim)
        L[tril_indices[0], tril_indices[1]] = cov
        L[tril_indices[1], tril_indices[0]] = cov
        cov_matrix = torch.matmul(L, L.transpose(-1, -2))
        cov_matrix += 1e-6 * torch.eye(self.dim, device=cov.device)
        return cov_matrix

    def get_var_shape__(self, output):
        n = self.dim
        L = torch.zeros(output.shape[0], n, n, device=output.device)
        k = 0
        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    L[:, i, j] = torch.exp(output[:, k])
                else:
                    L[:, i, j] = output[:, k]
                k += 1
        cov_matrix = L @ L.transpose(-2, -1)
        return cov_matrix

    def get_var_shape__(self, vals):
        d, V = vals[:, :self.dim], vals[:, self.dim:]
        D = torch.eye(d.shape[1], device=vals.device)
        A = torch.linalg.inv(D)
        for k in range(1, V.shape[1]):
            gamma = (V @ A @ A.T @ V.T)[0,0]
            eta = 1 / (1 + gamma)
            A = A - ((1-torch.sqrt(eta)) / gamma) * A @ A.T @ V.T @ V @ A
        cov_matrix = A @ A.T

        #if not bool((cov_matrix == cov_matrix.T).all() and (torch.linalg.eig(cov_matrix)[0][:,0]>=0).all()):
        #    print("Not PSD!")
        #    print("This is the cov matrix")
        #    print(cov_matrix)
        return cov_matrix

    def get_var_shape__(self, vals):
        m = torch.zeros((self.dim, self.dim), dtype=vals.dtype).cuda()
        xs, ys = torch.triu_indices(*m.shape)
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

    def get_var_shape_(self, cov):
        L = torch.zeros((self.dim, self.dim), device=cov.device)
        x_positions, y_positions = torch.tril_indices(row=self.dim, col=self.dim)
        L[x_positions, y_positions] = cov
        cov_matrix = torch.matmul(L, L.transpose(-1, -2))
        cov_matrix += 1e-6 * torch.eye(self.dim, device=cov.device)
        if bool((cov_matrix == cov_matrix.T).all() and (torch.eig(cov_matrix)[0][:,0]>=0).all()):
            print("Not PSD!")
        return cov_matrix

    def get_var_shape__(self, vals):
        dim = self.dim
        L = torch.zeros((dim, dim), device=vals.device)
        L[np.tril_indices(dim)] = vals
        cov_matrix = L @ L.T
        return cov_matrix

    def get_var_shape__(self, vals):
        vals = vals[0]
        # Compute diagonal matrix D and lower triangular matrix L
        D = torch.diag(torch.exp(vals[:self.dim]))
        L = torch.eye(D, device=vals.device)
        L = L.tril(-1)
        L[L == 0] = vals[self.dim:]

        # Compute the covariance matrix C
        C = L @ D @ L.T

        # Ensure that C is positive definite
        C = C + torch.eye(D) * 1e-6

        return C

    def get_var_shape_(self, vals):
        L = torch.zeros((self.dim, self.dim), dtype=vals.dtype).cuda()
        xs, ys = torch.triu_indices(*L.shape)
        L[xs, ys] = vals
        # set diagonal elements to be exp of half the log_var_mlp output
        L[torch.eye(self.dim, dtype=torch.bool).cuda()] = torch.exp(0.5 * vals[0, :self.dim])
        # construct covariance matrix as LL^T
        covariance = torch.matmul(L, L.t())
        # add small constant to diagonal to ensure positive definiteness
        covariance += torch.eye(self.dim).cuda() * 1e-6
        return covariance

    def get_var_shape__(self, cov):
        L = torch.zeros((self.dim, self.dim), device=cov.device)
        tril_indices = torch.tril_indices(row=self.dim, col=self.dim)
        L[tril_indices[0], tril_indices[1]] = cov
        cov_matrix = torch.matmul(L, L.transpose(-1, -2))
        cov_matrix += 1e-6 * torch.eye(self.dim, device=cov.device)
        return cov_matrix

    def forward(self, x):
        h = self.relu(self.input_mlp(x))

        mean = self.mean_mlp(h)
        var = self.get_var_shape(self.log_var_mlp(h))
        # samples = self.reparametrization(mean, var)
        samples = None

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
        # Flip tensors # TODO: napravi ovo u preprocessanju (manje zahtjevno)
        x_left, x_right = torch.split(x_left_right, (self.seq_len, self.seq_len), dim=0)
        x_right_rev = torch.flip(x_right, dims=[0])

        # Pass
        _, h_left = self.gru_left(x_right_rev)
        _, h_right = self.gru_right(x_left)
        _, h = self.gru(torch.cat([z_old, h_left, h_right], dim=1))
        z_samples, z_mean, z_log_var = self.mlp(h)

        return z_samples, z_mean, z_log_var


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
        n_repetitions = x_old.shape[0]
        z = z.repeat(n_repetitions, 1)
        _, h = self.gru(torch.concat([x_old, z], dim=1))  # TODO: (later) umjesto da saljes z (i tako S puta), salji mu_z, log_var_z
        x_sample, x_mean, x_log_var = self.mlp(h)

        return x_sample, x_mean, x_log_var

class GenerativeModelZ(nn.Module):
    def __init__(self, hidden_dim=4, latent_dim=10, dropout=0.2):
        super(GenerativeModelZ, self).__init__()

        self.gru_z = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim, dropout=dropout)
        self.mlp_z = MLP(input_size=hidden_dim, output_size=latent_dim)

    def forward(self, z_old):
        """
        p_Phi (z_t | z_(t-1) )
        :param z: z_(t-1)
        :return: z_t
        """
        _, h_z = self.gru_z(z_old)
        z, mu_z, log_var_z = self.mlp_z(h_z)

        return z, mu_z, log_var_z

class JoinNet(nn.Module):
    def __init__(self, p):
        super(JoinNet, self).__init__()
        self.cnn = nn.Conv1d(p, 1, 1)

    def forward(self, x):
        x = x.permute(1, 0)
        x = self.cnn(x)
        x = x.permute(1, 0)
        return x

class NSVM(nn.Module):
    def __init__(self, p=2, hidden_dim=4, latent_dim=10, S=50, seq_len=3):
        super(NSVM, self).__init__()
        self.S = S
        self.seq_len = seq_len

        self.join_net = JoinNet(p=p).cuda()
        self.inference_net = InferenceModel(p=1, hidden_dim=hidden_dim, latent_dim=latent_dim, seq_len=seq_len, S=S).cuda()
        self.genx_net = GenerativeModelX(p=1, hidden_dim=hidden_dim, latent_dim=latent_dim, S=S).cuda()
        self.genz_net = GenerativeModelZ(hidden_dim=hidden_dim, latent_dim=latent_dim).cuda()

        rnd = torch.randn(size=(1, latent_dim)).cuda()
        self.z_old = nn.Parameter(rnd, requires_grad=True)
        self.z_buffer = self.z_old.data.clone().detach()

    def forward(self, x):
        """
        :param x: x_(t-1)
        :return: x_t
        """
        self.z_old.data = self.z_buffer.data
        x = self.join_net(x)
        #x_joined2 = self.join_net2(x)

        x_old, _ = torch.split(x, (self.seq_len, self.seq_len), dim=0)

        # Generate S paths
        #x_mids, x_means, x_log_vars = [], [], []
        #z_samples, z_mean, z_log_var = self.inference_net(x, self.z_old)

        #for z_sample in z_samples:
        #    x_mid, x_mean, x_log_var = self.genx_net(x_old, z_sample)

        #    x_mids.append(x_mid[0])
        #    x_means.append(x_mean)
        #    x_log_vars.append(x_log_var)

        #x_next = sum(x_mids) / len(x_mids)

        #self.z_old = nn.Parameter(z_mean, requires_grad=True)

        # Inference net
        _, mutilda_z, sigmatilda_z = self.inference_net(x, self.z_old)
        z_new = mutilda_z

        # GenX net
        _, mu_x, sigma_x = self.genx_net(x_old, z_new)
        x_new = mu_x

        # GenZ net
        _, mu_z, sigma_z = self.genz_net(self.z_old)

        self.z_buffer = z_new.detach().data
        return x_new, mutilda_z, sigmatilda_z, mu_x, sigma_x, mu_z, sigma_z

    def load_model(self):
        self.load_state_dict(torch.load('./nsvm_weights.pth'))
    def save_model(self):
        torch.save(self.state_dict(), './nsvm_weights.pth')

class cNSVM(nn.Module):
    def __init__(self, p, hidden_dim, latent_dim, seq_len):
        super(cNSVM, self).__init__()
        self.networks = nn.ModuleList([
            NSVM(p=p, hidden_dim=hidden_dim, latent_dim=latent_dim, seq_len=seq_len)
        for _ in range(p)])

    def forward(self, X):
        '''
        Perform a forward pass
        :return: X tensor of shape (T, p)
        '''
        return torch.cat([network(X) for network in self.networks], dim=2)

def elbo_loss_unit(params, x_hat, x, loss_mse) -> torch.tensor:
    N = len(params)
    cum_loss = 0

    for (_, mutilda_z, sigmatilda_z, mu_x, sigma_x, mu_z, sigma_z) in params:
        rec_loss = loss_mse(x, x_hat)
        d1 = torch.distributions.MultivariateNormal(loc=mutilda_z, covariance_matrix=sigmatilda_z)
        d2 = torch.distributions.MultivariateNormal(loc=torch.zeros(4).cuda(), covariance_matrix=torch.eye(4).cuda())
        kl_loss = torch.distributions.kl_divergence(d1, d2)[0]
        cum_loss += rec_loss + kl_loss

    return cum_loss

def elbo_loss_unit_(params, x_hat, x, loss_mse) -> torch.tensor:
    N = len(params)
    cum_loss = 0

    for (_, mutilda_z, sigmatilda_z, mu_x, sigma_x, mu_z, sigma_z) in params:
        rec_loss = loss_mse(x, x_hat)
        d1 = torch.distributions.MultivariateNormal(loc=mutilda_z, covariance_matrix=sigmatilda_z)
        d2 = torch.distributions.MultivariateNormal(loc=mu_z, covariance_matrix=sigma_z)
        kl_loss = torch.distributions.kl_divergence(d1, d2)[0]
        cum_loss += rec_loss + kl_loss

    return cum_loss

def elbo_loss_z(params) -> torch.tensor:
    N = len(params)
    cum_loss = 0

    for (x, mutilda_z, sigmatilda_z, mu_x, sigma_x, mu_z, sigma_z) in params:
        epstilda_z = torch.randn_like(mutilda_z).cuda()

        part1 = torch.log(torch.det(sigma_z))
        part2 = (mutilda_z + epstilda_z @ sigmatilda_z - mu_z) @ torch.linalg.inv(sigma_z) @ (mutilda_z + epstilda_z @ sigmatilda_z - mu_z).T
        part3 = torch.log(torch.det(sigma_x))
        part4 = (x - mu_x).T @ torch.linalg.inv(sigma_x) @ (x - mu_x)
        part5 = torch.log(torch.det(sigmatilda_z))

        cum_loss += part1 + part2[0,0] + part3 + part4[0,0] - part5

    return 1/(2*N) * cum_loss


def train_nsvm(data: torch.Tensor, n_points: int, train_perc=0.95, epochs=100, x_dim=2):
    # Params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 10
    latent_dim = 4
    lr = 1e-3
    batch_size = 1

    # Take metadata of the dataset - data info
    data = data[:n_points]  # Take only the number that user wants
    N, p = data.shape
    #data = data.ravel() # TODO: pazi da si ovo promijenija sada

    # Models
    seq_len = int((x_dim - 1) / 2)
    model = NSVM(p=p, hidden_dim=hidden_dim, latent_dim=latent_dim, seq_len=seq_len).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    early_stopper = EarlyStopper(patience=40)

    # Generate dataset
    idx_mids = [math.floor(x_dim / 2)+p_ for p_ in range(1)]
    inputs_np = [data[i:(i + x_dim * 1)] for i in range(0, N * 1, 1)] # TODO: 3 puta si 'p' promijenija u '1'
    inputs_np = [[x for (i,x) in enumerate(input) if i not in idx_mids] for input in inputs_np]
    inputs_np = np.array([d for d in inputs_np if len(d) == len(inputs_np[0])])
    outputs_np = np.array([data[i:(i+1), :1] for i in range(idx_mids[0], len(data)-idx_mids[0])])

    inputs = torch.tensor(inputs_np, dtype=torch.float32).to(device) # [:-1]
    outputs = torch.tensor(outputs_np, dtype=torch.float32).to(device)

    # Split dataset (and store into data loaders)
    N_train = int(inputs.shape[0] * train_perc)
    train_set = TensorDataset(inputs[:N_train], outputs[:N_train])
    valid_set = TensorDataset(inputs[N_train:], outputs[N_train:])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    # Training
    loss_history_train, loss_kld_history_train, loss_rec_history_train = [], [], []
    loss_history_valid, loss_kld_history_valid, loss_rec_history_valid = [], [], []
    x_hat = None
    loss_fn = elbo_loss_unit
    loss_mse = nn.MSELoss()

    max_epochs = 100 # 50
    for epoch in range(max_epochs):
        loss_ = []
        inputs_pr_train, outputs_pr_train = [], []
        inputs_pr_valid, outputs_pr_valid = [], []
        params = []
        for i_batch, (x_prev, x_next) in enumerate(train_loader):
            # Forward prop
            x_prev = x_prev[0]
            x_next = x_next[0]
            #x_prev = x_prev.T
            output = x_new, mutilda_z, sigmatilda_z, mu_x, sigma_x, mu_z, sigma_z = model(x_prev)

            # Calculate loss
            loss = loss_fn([output], x_new, x_next, loss_mse)
            rec_loss = loss_mse(x_new, x_next)

            # Save info
            loss_.append((loss.item(), rec_loss.item()))
            if epoch == max_epochs-1:
                outputs_pr_train += [x_new.squeeze().detach().cpu().numpy().tolist()]
                if i_batch == 0:
                    i = x_prev.squeeze().detach().cpu().numpy().reshape((-1, p)).tolist()
                else:
                    i = [x_prev.squeeze().detach().cpu().numpy().tolist()[-p:]]
                inputs_pr_train += i

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save cumulated loss
        loss_history_train.append(sum(l[0] for l in loss_) / len(loss_))
        loss_rec_history_train.append(sum(l[1] for l in loss_) / len(loss_))

        # region VALIDATION
        loss_ = []
        for i_batch, (x_prev, x_next) in enumerate(valid_loader):
            # Forward prop
            #x_prev = x_prev.T
            x_prev, x_next = x_prev[0], x_next[0]
            output = x_new, mutilda_z, sigmatilda_z, mu_x, sigma_x, mu_z, sigma_z = model(x_prev)
            loss = loss_fn([output], x_new, x_next, loss_mse)
            rec_loss = loss_mse(x_new, x_next)

            # Save info
            loss_.append((loss.item(), rec_loss.item()))
            if epoch == max_epochs-1:
                outputs_pr_valid += [x_new.squeeze().detach().cpu().numpy().tolist()]
                if i_batch == 0:
                    i = x_prev.squeeze().detach().cpu().numpy().reshape((-1, p)).tolist()
                else:
                    i = [x_prev.squeeze().detach().cpu().numpy().tolist()[-p:]]
                inputs_pr_valid += i

        # Save cumulated loss
        loss_history_valid.append(sum(l[0] for l in loss_) / len(loss_))
        loss_rec_history_valid.append(sum(l[1] for l in loss_) / len(loss_))

        # Check for early stopping
        es_flag = early_stopper.early_stop(loss_history_valid[-1])  # loss_rec_history_train
        patience_msg = ''
        if es_flag == EarlyStopper.PATIENCE_ENDED:
            model.load_model()
            print("Early stopping!")
            break
        elif es_flag == EarlyStopper.EVERYTHING_OK:
            model.save_model()
        else:
            patience_msg += f'Patience: {early_stopper.counter}'
        # endregion

        # Log progress
        patience_msg = ""
        print(f"Epoch: [{epoch+1}/{max_epochs}]")
        print(f"Total loss (train): {loss_history_train[-1]} | Rec loss (train): {loss_rec_history_train[-1]}")  # | Total loss (valid): ") #{loss_history_valid[-1]}" + patience_msg)
        print(f"Total loss (valid): {loss_history_valid[-1]} | Rec loss (valid): {loss_rec_history_valid[-1]}")
        print(patience_msg)
        print(f"Weights: {model.join_net.cnn.weight}")
        print("="*10)

    # Save outputs
    inputs_pr_train, inputs_pr_valid, outputs_pr_train, outputs_pr_valid = get_forecast(model, train_loader, valid_loader, p)


    print("Finished!")
    print(f"Weights: {model.join_net.cnn.weight}")

    #plt.plot(loss_history_train, label='lik. loss')
    plt.plot(loss_rec_history_train, label='rec loss train')
    plt.plot(loss_rec_history_valid, label='rec loss valid')
    plt.legend()
    plt.show()
    return {
        'loss_train': loss_history_train,
        'loss_valid': loss_history_valid,

        'outputs_train': np.array(outputs_pr_train)[:, None],
        'outputs_valid': np.array(outputs_pr_valid)[:, None],
        'inputs_train': np.array(inputs_pr_train),
        'inputs_valid': np.array(inputs_pr_valid)
    }

# AKO GLEDAMO VAL LOSS; NE MOZE DOBRO PREDVIDIT (STD JE PREMAL DA BI SE VIDILO KOLIKO DOBRO PREDVIDJA)
# AKO GLEDAMO TRAIN LOSS (OVERFITTANJE); TODO: ?