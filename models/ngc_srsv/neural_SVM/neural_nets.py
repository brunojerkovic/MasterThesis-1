import torch
import torch.nn as nn
from torch.optim import Adam

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.fc_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h_ = self.leaky_relu(self.fc_input(x))
        h_ = self.leaky_relu(self.fc_input2(h_))
        mean = self.fc_mean(h_)
        log_var = self.fc_var(h_)

        return mean, log_var

# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc_hidden = nn.Linear(latent_dim, hidden_dim)
        self.fc_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_output = nn.Linear(hidden_dim, output_dim)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.leaky_relu(self.fc_hidden(x))
        h = self.leaky_relu(self.fc_hidden2(h))

        x_hat = torch.sigmoid(self.fc_output(h))
        return x_hat

# Model
class Model(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.device = device

    def reparametrization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparametrization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)

        return x_hat, mean, log_var

# Generative model
class Generator(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=10, latent_dim=4):
        super(Generator, self).__init__()
        self.gru_z = nn.GRU(input_size=latent_dim, hidden_size=hidden_dim)
        self.mlp_z = nn.Linear(in_features=hidden_dim, out_features=latent_dim*2)

        self.gru_g = nn.GRU(input_size=input_dim+latent_dim, hidden_size=hidden_dim)
        self.mlp_g = nn.Linear(in_features=hidden_dim, out_features=input_dim*2)

    def reparametrization(self, mean, log_var):
        var = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(var).to(self.device)
        z = mean + var * epsilon
        return z

    def forward(self, x, z):
        _, h_z = self.gru_z(z)
        mean, log_var = self.mlp_z(h_z)

        z = self.reparametrization(mean, log_var)









def train(X: torch.Tensor):
    # Params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(size=(784,)).to(device)
    x_dim = 784
    hidden_dim = 400
    latent_dim = 200
    lr = 1e-3
    epochs = 30

    # Models
    encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
    model = Model(encoder=encoder, decoder=decoder, device=device)

    # Loss and optimizer
    def vae_loss(x, x_hat, mean, log_var):
        bce_loss = nn.BCELoss()
        reconstruction_loss = bce_loss(x_hat, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

        return reconstruction_loss, kld, reconstruction_loss + kld
    optimizer = Adam(model.parameters(), lr=lr)

    # Training
    loss_history = []
    for epoch in range(epochs):
        x_hat, mean, log_var = model(x)
        rec_loss, kld_loss, loss = vae_loss(x, x_hat, mean, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        print(f"Epoch: {epoch+1} | Reconstruction loss: {rec_loss.item()} | KL loss: {kld_loss.item()} | Total loss: {loss.item()}")

    print("Finished!")
    return loss_history



