import time

import torch
import numpy as np

import utils
from models.ngc.sourcecode.cmlp import cMLP, cMLPSparse, train_model_ista, train_unregularized, train_model_ista_stocks
from models.ngc.data_cleaner import clean_data
from models.model import Model
from result_saver import ResultSaver


class NGC(Model):
    def __init__(self, config: utils.dotdict, result_saver: ResultSaver):
        super().__init__(config, result_saver)

        self.verbose = config.verbose
        self.num_layers = config.num_layers
        self.model_choice = config.model
        self.lr = config.lr
        self.lam = config.lam
        self.lam_ridge = config.lam_ridge
        self.max_iter = config.max_iter

    def _algorithm(self, series, coef_mat, edges) -> tuple:
        # Set seeds
        self.set_seeds()

        # Dataset creation
        X_np, beta, GC, coef_mat = clean_data(self.config, series, coef_mat, edges)
        X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=self.device)

        # Set up model
        layers = [self.config.layers_size] * self.config.num_layers
        start_time = time.time()
        cmlp = cMLP(X.shape[-1], lag=2, hidden=layers, model_choice=self.model_choice).to(self.device)

        # Train with ISTA
        train_function = train_model_ista if self.config['loader'] != 'stocks' else train_model_ista # train_model_ista_stocks
        train_loss_list = train_function(
            cmlp, X, lam=self.lam, lam_ridge=self.lam_ridge, lr=self.lr, penalty='H', max_iter=self.max_iter,
            check_every=100, verbose=self.config.verbose)
        train_losses = [loss.cpu().item() for loss in train_loss_list]

        # Verify learned Granger causality
        GC_est = cmlp.GC().cpu().data.numpy()
        accuracy = self._calculate_accuracy(coef_mat, GC_est)

        # For plotting predictions
        predictions = np.zeros_like(X.cpu().data.numpy()[0, :, :])
        for i in range(X.shape[-1]):
            out = cmlp.networks[i](X[:, :-1]).cpu().data.numpy()
            predictions[2:, i] = out[0, :, 0]

        if self.verbose:
            print('True variable usage = %.2f%%' % (100 * np.mean(GC)))
            print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
            print('Accuracy = %.2f%%' % accuracy)

        # Save the results
        results = {
            'beta': beta,
            'GC': GC.tolist(),
            'GC_est': GC_est.tolist(),
            'accuracy': accuracy,
            'train_losses': train_losses,
            'predictions': predictions.tolist(),
            'coef_mat': coef_mat.tolist(),
            'time': time.time() - start_time
        }

        return accuracy, results
