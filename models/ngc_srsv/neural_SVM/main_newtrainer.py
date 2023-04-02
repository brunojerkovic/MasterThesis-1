from models.ngc_srsv.neural_SVM.nsvm_net import train_nsvm
from models.ngc_srsv.neural_SVM.mlp_2_layer import train_mlp
from models.ngc_srsv.neural_SVM.ploting import plot_predictions, plot_losses


def main_newtrainer(X):
    n_points = 1_000
    train_perc = 0.8
    epochs = 2
    x_view = 200
    x_dim = 21  # Assumption: x_dim has to be odd number
    p = X.shape[1]

    results = train_nsvm(X, n_points, train_perc, epochs, x_dim)

    plot_predictions(X_hat_train=results['outputs_train'],
                     X_hat_valid=results['outputs_valid'],
                     X_train=results['inputs_train'],
                     X_valid=results['inputs_valid'],
                     x_view=x_view,
                     x_dim=x_dim,
                     p=1)
    plot_losses(**results)

    return results
