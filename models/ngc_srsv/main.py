import math
import time
import random
import matplotlib.pyplot as plt
import torch
import numpy as np

import utils
#from models.ngc_srsv.sourcecode.cmlp import cMLP, train_model_ista, test_model
from models.ngc_srsv.sourcecode.clstm import cLSTM, train_model_ista as s
from models.ngc_srsv.data_cleaner import clean_data, clean_data2
from models.model import Model
from result_saver import ResultSaver
from models.ngc_srsv.sourcecode.cnsvm import cNSVM, train_model_ista, test_model


from models.ngc_srsv.neural_SVM.main_newtrainer import main_newtrainer


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

    # IF YOU WANT TO USE OLD CODE
    # #results = main_newtrainer(X_np)
    #         ## print(results)
    #         #exit(0)

    def _algorithm(self, series, coef_mat, edges) -> tuple:
        # Dataset creation
        X_np, beta, GC, coef_mat = clean_data(self.config, series, coef_mat, edges)

        # TODO: DELETE ME
        #a = main_newtrainer(X_np)
        #exit(0)

        x_dim = 7
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_loader, valid_loader = clean_data2(X_np, self.device, train_perc=self.config.tvt_split, x_dim=x_dim)

        # Set up model
        start_time = time.time()
        cnsvm = cNSVM(p=X_np.shape[-1], hidden_dim=10, latent_dim=4, seq_len=math.floor(x_dim / 2), device=self.device)

        # Calculate total number of parameters
        params = []
        for param in cnsvm.parameters():
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
        train_function = train_model_ista if self.config['loader'] != 'stocks' else train_model_ista

        train_loss_list, valid_loss_list = train_function(
            cnsvm,
            train_loader,
            valid_loader,
            lam=self.lam,
            lam_ridge=self.lam_ridge,
            lr=self.lr,
            penalty='H',
            max_iter=self.max_iter,
            verbose=self.config.verbose,
            device=self.device
        )

        train_losses = [loss.cpu().item() for loss in train_loss_list]
        valid_losses = [loss.cpu().item() for loss in valid_loss_list]

        # Verify learned Granger causality
        GC_est = cnsvm.GC(threshold=False).cpu().data.numpy()
        accuracy = self._calculate_binary_accuracy(coef_mat, GC_est)
        # region FOR DEBUGGING
        #accuracy_non_binary = self._calculate_accuracy(coef_mat, GC_est)
        #print("="*50)
        #print("GC", GC_est)
        #print("accuracy", accuracy)
        #print("accuracy_non_binary", accuracy_non_binary)
        #print("Weights: ", [net.join_net.cnn.weight for net in cnsvm.networks])
        #print("="*50)
        #exit(0)
        # endregion

        # For plotting predictions
        predictions = []
        data = []
        for x_prev,x_next in train_loader:
            x_prev,x_next = x_prev[0], x_next[0]
            outputs = cnsvm(x_prev)
            outputs = torch.cat([o[0] for o in outputs], axis=1)

            predictions.append(outputs.detach().cpu().numpy())
            data.append(x_next.detach().cpu().numpy())
        predictions = np.array(predictions)[:, 0, :]
        data = np.array(data)[:, 0, :]
        noise_cov_mat_est = np.cov((predictions - data).T)

        #for i in range(X_np)
        #for i in range(X.shape[-1]):
        #    out = cmlp.networks[i](X[:, :-1], eps, i).cpu().data.numpy()
        #    predictions[2:, i] = out[0, :, 0]

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
            'time': time.time() - start_time,
            'noise_cov_mat': [self.config.sigma_eta_diag, self.config.sigma_eta_off_diag, self.config.sigma_eta_off_diag, self.config.sigma_eta_diag],
            'noise_cov_mat_est': noise_cov_mat_est.ravel().tolist()
        }

        return accuracy, results
