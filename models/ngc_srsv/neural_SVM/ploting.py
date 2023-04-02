from matplotlib import pyplot as plt
import numpy as np


def plot_losses(**losses):
    plt.title('Loss functions')
    for loss_name, loss_value in losses.items():
        if not loss_name.startswith('loss'):
            continue
        plt.plot(loss_value, label=loss_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.clf()

def plot_predictions(X_hat_train, X_hat_valid, X_train, X_valid, x_view=200, x_dim=1, p=2):
    N_train, N_valid = X_train.shape[0], X_valid.shape[0]

    if x_dim > 1:
        output_begin = np.zeros((x_dim-1, p))
        X_hat_train = np.concatenate((output_begin, X_hat_train))
        X_hat_valid = np.concatenate((output_begin, X_hat_valid))

    X_hat = np.concatenate((X_hat_train, X_hat_valid))[x_dim-1:, :]
    X = np.concatenate((X_train, X_valid))

    for i in range(p):
        plt.title(f'Predictions for variable {i+1}')
        plt.plot(X[:, i], linewidth=2., label=f'X_{i+1}')
        plt.plot(X_hat[:, i], label=f'X_hat_{i+1}')

        plt.axvline(x=N_train, color='red')
        plt.xlabel('Epoch')
        plt.legend()
        #plt.xlim((len(X)-x_view, len(X)))
        plt.show()
        plt.clf()