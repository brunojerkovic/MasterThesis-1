import time

import torch
import numpy as np

import utils
from models.ngc.sourcecode.cmlp import cMLP, cMLPSparse, train_model_ista, train_unregularized
from models.ngc.sourcecode.clstm import cLSTM
from models.ngc.sourcecode.clstm import train_model_ista as train_model_ista_lstm
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

    def _algorithm(self, series, coef_mat, edges) -> float:
        # Set seeds
        self.set_seeds()

        # Dataset creation
        X_np, beta, GC, coef_mat = clean_data(self.config, series, coef_mat, edges)
        X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=self.device)

        # Set up model
        layers = [self.config.layers_size] * self.config.num_layers
        start_time = time.time()
        #cmlp = cMLP(X.shape[-1], hidden=layers[0]).to(device=self.device)
        train_loss_list, cmlp = None, None
        if self.config.model in ['ngc', 'ngc0']:
            cmlp = cMLP(X.shape[-1], lag=2, hidden=layers, model_choice=self.model_choice).to(device=self.device)

            # Calculate total number of parameters
            params = []
            for param in cmlp.parameters():
                params.append(param.shape)
            lens = 0
            for param in params:
                if len(param) == 2:
                    lens += param[0] * param[1]
                if len(param) == 1:
                    lens += param[0]
                if len(param) == 3:
                    lens += param[0] * param[1] * param[2]

            # Train with ISTA
            train_loss_list = train_model_ista(
                cmlp, X, lam=self.lam, lam_ridge=self.lam_ridge, lr=self.lr, penalty='H', max_iter=self.max_iter,
                check_every=100, verbose=self.config.verbose)

        elif self.config.model == 'ngc_lstm':
            cmlp = cLSTM(X.shape[-1], hidden=layers[0]).to(device=self.device)

            # Calculate total number of parameters
            params = []
            for param in cmlp.parameters():
                params.append(param.shape)
            lens = 0
            for param in params:
                if len(param) == 2:
                    lens += param[0] * param[1]
                if len(param) == 1:
                    lens += param[0]
                if len(param) == 3:
                    lens += param[0] * param[1] * param[2]

            # Train with ISTA
            train_loss_list = train_model_ista_lstm(
                cmlp, X, context=self.config.context, lam=self.lam, lam_ridge=self.lam_ridge, lr=self.lr, max_iter=self.max_iter,
                check_every=100, verbose=self.config.verbose)

        train_losses = [loss.cpu().item() for loss in train_loss_list]

        # Verify learned Granger causality
        GC_est = cmlp.GC().cpu().data.numpy()
        accuracy = self._calculate_binary_accuracy(coef_mat, GC_est)

        # For plotting predictions
        predictions = np.zeros_like(X.cpu().data.numpy()[0, :, :])
        if self.config.loader != 'stocks':
            for i in range(X.shape[-1]):
                if self.config.model == 'ngc_lstm':
                    out = cmlp.networks[i](X[:, :-1])[0].cpu().data.numpy()
                    predictions[1:, i] = out[0, :, 0]
                else:
                    out = cmlp.networks[i](X[:, :-1]).cpu().data.numpy()
                    predictions[2:, i] = out[0, :, 0]

        noise_cov_mat_est = np.cov((predictions - series[:len(predictions)]).T)
        if self.verbose:
            print('True variable usage = %.2f%%' % (100 * np.mean(GC)))
            print('Estimated variable usage = %.2f%%' % (100 * np.mean(GC_est)))
            print('Accuracy = %.2f%%' % accuracy)
            print("Time taken: ", time.time() - start_time)
            print("="*50)

        # Save the results
        results = {
            'beta': beta,
            'GC': GC.tolist(),
            'GC_est': GC_est.tolist(),
            'accuracy': accuracy,
            'train_losses': train_losses,
            'coef_mat': coef_mat.tolist(),
            'time': time.time() - start_time,
            'noise_cov_mat': [self.config.sigma_eta_diag, self.config.sigma_eta_off_diag, self.config.sigma_eta_off_diag, self.config.sigma_eta_diag],
            'noise_cov_mat_est': noise_cov_mat_est.ravel().tolist()
        }

        return accuracy, results
