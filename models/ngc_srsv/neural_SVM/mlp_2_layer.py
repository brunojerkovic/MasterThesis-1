import numpy as np
import torch
import torch.nn as nn
import os
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from models.ngc_srsv.neural_SVM.early_stopper import EarlyStopper


class MLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=4, latent_dim=10, p=2, path='./model_weights'):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(input_dim*p, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, p)
        self.relu = nn.ReLU()

        if not os.path.exists(path):
            os.makedirs(path)
        self.path = os.path.join(path, 'mlp.pth')

    def save_model(self):
        torch.save(self.state_dict(), self.path)

    def load_model(self):
        self.load_state_dict(torch.load(self.path))

    def forward(self, x):
        x = self.relu(self.mlp1(x))
        x = self.mlp2(x)
        return x

def train_mlp(data: torch.Tensor, n_points: int, train_perc=0.8, epochs = 100, x_dim=2):
    # Take data meta-info
    data = data[:n_points]  # Take only the number that user wants
    N, p = data.shape
    data = data.ravel()

    # Params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 10
    latent_dim = 4
    lr = 1e-3
    batch_size = 1

    # Models
    model = MLP(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, p=p).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    early_stopper = EarlyStopper()

    # Split dataset
    inputs_np = [data[i:(i+x_dim*p)] for i in range(0, N*p, p)]
    inputs_np = np.array([d for d in inputs_np if len(d) == len(inputs_np[0])])
    inputs = torch.tensor(inputs_np, dtype=torch.float32).to(device)[:-1]

    outputs_np = np.array([data[i:i+p] for i in range(p*x_dim, N*p, p)])
    outputs = torch.tensor(outputs_np, dtype=torch.float32).to(device)

    N_train = int(inputs.shape[0] * train_perc)
    train_set = TensorDataset(inputs[:N_train], outputs[:N_train])
    valid_set = TensorDataset(inputs[N_train:], outputs[N_train:])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    # Training
    loss_history_train = []
    loss_history_valid = []
    inputs_pr_train, outputs_pr_train = [], []
    inputs_pr_valid, outputs_pr_valid = [], []
    x_hat = None
    for epoch in range(epochs):
        loss_ = []
        inputs_pr_train, outputs_pr_train = [], []
        inputs_pr_valid, outputs_pr_valid = [], []
        for i_batch, (x, y) in enumerate(train_loader):
            # Forward prop
            x_hat = model(x)
            loss = loss_fn(x_hat, y)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Save info
            loss_.append((loss.item(),))
            outputs_pr_train += [x_hat.squeeze().detach().cpu().numpy().tolist()]
            if i_batch == 0:
                i = x.squeeze().detach().cpu().numpy().reshape((-1, p)).tolist()
                inputs_pr_train += i
            else:
                i = x.squeeze().detach().cpu().numpy().tolist()
                inputs_pr_train += [i[-p:]]

        # Save cumulated loss
        loss_history_train.append(sum(l[0] for l in loss_) / len(loss_))

        # region VALIDATION
        loss_ = []
        for i_batch, (x, y) in enumerate(valid_loader):
            # Forward prop
            x_hat = model(x)
            loss = loss_fn(x_hat, y)

            # Save info
            loss_.append((loss.item(),))
            outputs_pr_valid += [x_hat.squeeze().detach().cpu().numpy().tolist()]
            if i_batch == 0:
                i = x.squeeze().detach().cpu().numpy().reshape((-1, p)).tolist()
                inputs_pr_valid += i
            else:
                i = x.squeeze().detach().cpu().numpy().tolist()
                inputs_pr_valid += [i[-p:]]

        # Save cumulated loss
        loss_history_valid.append(sum(l[0] for l in loss_) / len(loss_))

        # Check for early stopping
        patience_msg = ''
        es_flag = early_stopper.early_stop(loss_history_valid[-1])
        if es_flag == EarlyStopper.PATIENCE_ENDED:
            model.load_model()
            print("Early stopping!")
            break
        elif es_flag == EarlyStopper.EVERYTHING_OK:
            model.save_model()
        else:
            patience_msg += f' | Patience: {early_stopper.counter}'
        # endregion

        # Log progress
        print(
            f"Epoch: {epoch + 1} | Total loss (train): {loss_history_train[-1]} | Total loss (valid): {loss_history_valid[-1]}" + patience_msg)



    print("Finished!")
    return {
        'loss_train': loss_history_train,
        'loss_valid': loss_history_valid,
        'outputs_train': np.array(outputs_pr_train),
        'outputs_valid': np.array(outputs_pr_valid),

        'inputs_train': np.array(inputs_pr_train),
        'inputs_valid': np.array(inputs_pr_valid)
    }
