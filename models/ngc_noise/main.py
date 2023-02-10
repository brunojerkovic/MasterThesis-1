import time

import matplotlib.pyplot as plt
import torch
import numpy as np

import utils
from models.ngc_noise.sourcecode.cmlp import cMLP, train_model_ista as s, test_model
from models.ngc_noise.sourcecode.clstm import cLSTM, train_model_ista
from models.ngc_noise.data_cleaner import clean_data
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
        # X_np = np.load('X_ae.npy')
        X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=self.device)

        # Set up model
        layers = [self.config.layers_size] * self.config.num_layers
        start_time = time.time()
        cmlp = cMLP(X.shape[-1], lag=self.config.lag, hidden=layers, model_choice=self.model_choice).to(self.device)
        cmlp = cLSTM(X.shape[-1], hidden=100).to(self.device)

        # Get main stats of the data
        N = X.shape[1]
        p = X.shape[-1]

        # Split data into tvt splits
        tvt_split = self.config.tvt_split, (1-self.config.tvt_split)/2, (1-self.config.tvt_split)/2
        N_train, N_valid, N_test = int(N * tvt_split[0]), int(N * tvt_split[1]), int(N * tvt_split[2])
        X_train = X[:, :N_train, :]
        X_valid = X[:, N_train:(N_train+N_valid), :]
        X_test = X[:, (N_train+N_valid):, :]

        #mvn = torch.distributions.MultivariateNormal(torch.zeros(2), torch.tensor([[0.10157133, -0.00103958], [-0.00103958, 0.10146144]]))
        #eps_train = mvn.sample((N_train,)).T[None, :, :].cuda()  # TODO: cuda warning here
        #X_train = X_train - eps_train.transpose(2, 1)

        # Train with ISTA
        train_function = train_model_ista if self.config['loader'] != 'stocks' else train_model_ista # train_model_ista_stocks

        #noise = np.load('del_me.npy')
        #y_t_covs = np.cov([c for c in np.array([cmlp.networks[i].forward_distribution(X_train[:, :-1], 100) for i in range(p)])[:,:,:]])
        #y_t_covs_mean = np.mean(y_t_covs)

        train_loss_list = train_function(
            cmlp,
            X_train,
            context=10, # X_valid,
            lam=self.lam,
            lam_ridge=self.lam_ridge,
            lr=self.lr,
            # penalty='H',
            max_iter=20_000, # 30_000
            check_every=100,
            # verbose=self.config.verbose,
            # device=self.device
        )

        train_losses = [loss.cpu().item() for loss in train_loss_list]
        # valid_losses = [loss.cpu().item() for loss in valid_loss_list]

        # Verify learned Granger causality
        GC_est = cmlp.GC(threshold=False).cpu().data.numpy()
        accuracy = self._calculate_binary_accuracy(coef_mat, GC_est)
        accuracy_non_binary = self._calculate_accuracy(coef_mat, GC_est)
        print("="*50)
        print("GC", GC_est)
        print("accuracy", accuracy)
        print("accuracy_non_binary", accuracy_non_binary)
        print("="*50)

        # region DELETE (delete after debugging)
        #var = cmlp.get_variance()
        #print("VARIANCE", var)

        #noise = np.array([c for c in np.array([cmlp.networks[i].forward_distribution(X_train[:, :-1], 100) for i in range(p)])[:, :, :]])
        #estimated_cov_1 = np.mean(np.array([np.cov(signal) for signal in noise.squeeze().transpose(1,0,2)]), axis=0)
        #print("ESTIMATED COV (technique 1)", estimated_cov_1)


        # var = cmlp.get_variance()
        # plt.plot(train_losses, label='train_loss')
        # plt.plot(valid_losses, label='valid_loss')
        # plt.legend()
        # plt.show()

        mvn = torch.distributions.MultivariateNormal(torch.zeros(2), torch.tensor([[1., 0.], [0., 1.]]))
        eps_train = mvn.sample((N_train,)).T[None, :, :]#.cuda()  # TODO: cuda warning here
        y_train_t1 = np.array([cmlp.networks[i](X_train, eps_train, i).detach().cpu().numpy() for i in range(p)])
        y_train_t1 = y_train_t1.squeeze().T[:-1, :]
        #y_t_1_train_dp = np.array([[cmlp.networks[i](X_train[:, :-1], eps_train, i).detach().cpu().numpy() for i in range(p)] for j in range(100)])
        X_train_np = X_train.detach().cpu().numpy().squeeze()[self.config.lag:]

        # Plot predictions
        plt.plot(X_train_np[:, 0][:800], label='X_train')
        plt.plot(y_train_t1[:, 0][:800], label='y_train_t1')
        plt.legend()
        plt.show()

        # EPSILON ESTIMATED
        epsilons = (y_train_t1 - X_train_np).T
        epsilon_cov_est = np.cov(epsilons)
        print("ESTIMATED COV (technique 1)", epsilon_cov_est)

        # DIRECTLY ESTIMATE COV (repr. trick)
        cov_est = cmlp.get_variance()
        print("ESTIMATED COV (technique 2 - modeling)", cov_est)

        # TEST LOSS CALCULATED
        test_loss = test_model(cmlp, X_test)
        print("TEST LOSS", test_loss)
        exit(0)

        y_t_1_test = np.array([cmlp.networks[i](X_test[:, :-2], 0, i).detach().cpu().numpy() for i in range(p)])
        y_t_train = np.array([cmlp.networks[i](X_train[:, 1:-1], 0, i).detach().cpu().numpy() for i in range(p)])
        eps = (y_t_train - y_train_t1).squeeze().T
        mean = np.mean(eps, axis=0)
        cov = (1 / eps.shape[1]) * (eps - mean).T @ (eps - mean)

        cov_mat_rmse = self._calculate_accuracy(coef_mat, GC_est)
        y_train_rmse = self._calculate_predictions_accuracy(X_train[:, 1:-1], y_train_t1)
        y_test_rmse = self._calculate_predictions_accuracy(X_test[:, 1:-1], y_t_1_test)
        # endregion

        # For plotting predictions
        predictions = np.zeros_like(X.cpu().data.numpy()[0, :, :])

        eps = torch.randn((X.shape[-1], X.shape[1] - X.shape[-1])).cuda()  # TODO: cuda warning here
        for i in range(X.shape[-1]):
            out = cmlp.networks[i](X[:, :-1], eps, i).cpu().data.numpy()
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
