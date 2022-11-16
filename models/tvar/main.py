import os
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
        results = var_package.Large_tVAR(Data=data, P=self.P, lambda1_OPT=self.lambda1_opt, gamma1_OPT=self.gamma1_opt)
        coef_mat_hat = results[1].squeeze().T

        # Get accuracy of the model
        accuracy = self._calculate_accuracy(coef_mat, coef_mat_hat)

        # Save the results
        results = {
            'accuracy': accuracy,
            'coef_mat_hat': list(coef_mat_hat.ravel()),
            'time': time.time()-start_time
        }

        return accuracy, results

    def __install_package(self):
        if len(os.listdir('models/tvar/sourcecode')) == 0:
            git.Git('models/tvar/sourcecode').clone("https://github.com/lucabarbaglia/t-VAR.git")
