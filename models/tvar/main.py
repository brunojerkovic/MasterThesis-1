import os
#os.environ["R_HOME"] = "../software/R-4.2.2"
os.environ["R_HOME"] = "C:\Program Files\R\R-4.2.2"
import time

import git
import numpy as np
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri, r
from rpy2 import robjects

import utils
from models.model import Model
from result_saver import ResultSaver

pandas2ri.activate()


class TVAR(Model):
    def __init__(self, config: utils.dotdict, result_saver: ResultSaver):
        super().__init__(config, result_saver)

        self.P = config.P
        self.lambda1_opt = config.lambda1_opt
        self.gamma1_opt = config.gamma1_opt

        self.__install_package()

    def _algorithm(self, series: np.ndarray, coef_mat: np.ndarray, edges: np.ndarray) -> float:
        # Install package in R if it is not installed
        robjects.r('''
            if (!require("tVAR")) install.packages("tvar/sourcecode/t-VAR", repos=NULL, type="source")
        ''')

        # Train the model
        data = robjects.r.matrix(series, nrow=series.shape[0])
        var_package = rpackages.importr("tVAR")
        start_time = time.time()
        results = var_package.EM_VAR(data, self.P, nu=15, lambda1_OPT=self.lambda1_opt, gamma1_OPT=self.gamma1_opt)
        #results = var_package.Large_tVAR(Data=data, P=self.P, lambda1_OPT=self.lambda1_opt, gamma1_OPT=self.gamma1_opt)
        coef_mat_hat = results[1].squeeze().T
        print("TIME", time.time() - start_time)

        # Get accuracy of the model
        accuracy = self._calculate_binary_accuracy(coef_mat, coef_mat_hat)
        noise_cov_mat_est = np.cov(((data @ coef_mat_hat)[1:] - data[:-1]).T)

        print("NOISE COV MAT", noise_cov_mat_est.ravel())
        print("ACCURACY", accuracy)
        print("=" * 50)

        # Save the results
        results = {
            'accuracy': accuracy,
            'GC_est': coef_mat_hat.tolist(),
            'time': time.time()-start_time,
            'noise_cov_mat_est': noise_cov_mat_est.ravel().tolist(),
            'noise_cov_mat': [self.config.sigma_eta_diag, self.config.sigma_eta_off_diag, self.config.sigma_eta_off_diag, self.config.sigma_eta_diag]
        }

        return accuracy, results

    def __install_package(self):
        if len(os.listdir('models/tvar/sourcecode')) == 0:
            git.Git('models/tvar/sourcecode').clone("https://github.com/lucabarbaglia/t-VAR.git")
