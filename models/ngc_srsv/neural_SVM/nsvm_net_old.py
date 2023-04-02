import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


# My MLP
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=10):
        super(MLP, self).__init__()

        self.input_mlp = nn.Linear(input_size, hidden_size)
        self.mean_mlp = nn.Linear(hidden_size, output_size)
        self.log_var_mlp = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = self.input_mlp(x)
        mean = self.mean_mlp(h)
        log_var = self.log_var_mlp(h)

        return mean, log_var


# Generative model
class GenerativeModel(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=10, latent_dim=4, dropout=0.2):
        super(GenerativeModel, self).__init__()

        self.gru_x = nn.GRU(input_size=input_dim + latent_dim, hidden_size=hidden_dim, dropout=dropout)
        self.mlp_x = MLP(input_size=hidden_dim, output_size=input_dim, hidden_size=hidden_dim)

    def reparametrization(self, mean, log_var):
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(mean).cuda()
        z = mean + var * epsilon  # TODO: provjeri je li ovo dobro mnozenje
        return z

    def forward(self, x, z):
        """
        :param x: x_(t-1)
        :param z: z_(t)
        :return:
        """
        _, h_x = self.gru_x(torch.cat([x, z], dim=-1))
        mean_x, log_var_x = self.mlp_x(h_x)
        x = self.reparametrization(mean_x, log_var_x)

        return mean_x


# Inference model
class InferenceModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.2):
        super(InferenceModel, self).__init__()
        self.input_dim, self.hidden_dim, self.latent_dim = input_dim, hidden_dim, latent_dim

        self.rnn_bi = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, bidirectional=True, dropout=dropout)
        self.rnn = nn.GRU(input_size=latent_dim+hidden_dim*2, hidden_size=hidden_dim, dropout=dropout)

        self.mlp = MLP(hidden_dim, latent_dim)

    def reparametrization(self, mean, log_var):
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(mean).cuda()
        z = mean + var * epsilon
        return z

    def forward(self, x, z):
        """
        :param x: x_(t)
        :param z: z_(t-1)
        :return: z_t
        """
        _, h_ = self.rnn_bi(x)

        z_ = torch.cat([z, h_[0][None, :], h_[1][None, :]], dim=-1)
        _, h_z = self.rnn(z_)
        mean_z, log_var_z = self.mlp(h_z)

        z = self.reparametrization(mean_z, log_var_z)

        return z, mean_z, log_var_z

class NSVM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=4, latent_dim=10):
        super(NSVM, self).__init__()
        self.encoder = InferenceModel(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).cuda()
        self.decoder = GenerativeModel(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).cuda()

        rnd = torch.randn(size=(1, latent_dim)).cuda()
        self.z = nn.Parameter(rnd, requires_grad=True)

    def forward(self, x):
        z_, mean_z, log_var_z = self.encoder(x, self.z)
        self.z = nn.Parameter(z_, requires_grad=True)
        x_ = self.decoder(x, self.z)

        return x_, mean_z, log_var_z


def train_nsvm(X: torch.Tensor, n_points: int) -> dict:
    # Params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_dim = 1
    hidden_dim = 10
    latent_dim = 4
    lr = 1e-3
    epochs = 100 # 30
    batch_size = 10

    # Models
    model = NSVM(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim).to(device)

    # Loss and optimizer
    def vae_loss(x, x_hat, mean, log_var):
        mse_loss = nn.MSELoss()
        reconstruction_loss = mse_loss(x_hat, x)
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())  # TODO: provjeri je li ovo ima smisla

        return reconstruction_loss, kld, reconstruction_loss + kld

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Create dataset
    X = X[0, :, 0][:n_points]
    inputs_np = np.array([X[i:(i+x_dim)].detach().cpu().numpy() for i in range(X.shape[0])][:-x_dim])
    inputs = torch.tensor(inputs_np).to(device)
    outputs = torch.tensor([[X[i]] for i in range(x_dim, X.shape[0])]).to(device)
    dataset = TensorDataset(inputs, outputs)
    loader = DataLoader(dataset, batch_size=1)

    # Training
    loss_history, rec_loss_history, kld_loss_history = [], [], []
    outputs = []
    x_hat = None
    for epoch in range(epochs):
        loss_ = []
        for i_batch, (x, y) in enumerate(loader):
            # Forward prop
            x_hat, mean_z, log_var_z = model(x)
            rec_loss, kld_loss, loss = vae_loss(x, x_hat, mean_z, log_var_z)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save info (losses and outputs)
            loss_.append((loss.item(), rec_loss.item(), kld_loss.item()))
            if epoch == epochs - 1:
                o = x_hat.squeeze().detach().cpu().numpy().tolist()
                outputs += (o if i_batch == 0 else o[-1:])

        # Save losses of this epoch
        loss_history.append(sum(l[0] for l in loss_) / len(loss_))
        rec_loss_history.append(sum(l[1] for l in loss_) / len(loss_))
        kld_loss_history.append(sum(l[2] for l in loss_) / len(loss_))

        print(
            f"Epoch: {epoch + 1} | Reconstruction loss: {rec_loss_history[-1]} | KL loss: {kld_loss_history[-1]} | Total loss: {loss_history[-1]}")

    print("Finished!")
    return {
        'loss_total': loss_history,
        'loss_kld': kld_loss_history,
        'loss_rec': rec_loss_history,
        'outputs': outputs
    }
