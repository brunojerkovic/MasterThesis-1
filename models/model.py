from abc import ABC, abstractmethod
import numpy as np
import torch

import utils
from result_saver import ResultSaver


class Model(ABC):
    def __init__(self, config: utils.dotdict, result_saver: ResultSaver):
        self.config = config
        self.seed = config.seed
        self.result_saver = result_saver
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def set_seeds(self):
        """
        Set the seeds for reproducibility.
        :return: None
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device:
            torch.cuda.manual_seed(self.seed)

    @abstractmethod
    def _algorithm(self, series: np.ndarray, coef_mat: np.ndarray, edges: np.ndarray) -> float:
        pass

    def algorithm(self, series: np.ndarray, coef_mat: np.ndarray, edges: np.ndarray) -> float:
        """
        Training and testing the algorithm
        :param series: data for training, evaluation and testing
        :param coef_mat: coefficient matrix of the data (the first label)
        :param edges: binary edge matrix of the data (the second label)
        :return: Accuracy of the model on the data
        """
        accuracy, results = self._algorithm(series, coef_mat, edges)
        self.result_saver.add_results_to_buffer(self.config, results, data_flag=False)
        return accuracy

    def _calculate_binary_accuracy(self, coef_mat: np.ndarray, coef_mat_hat: np.ndarray, threshold=0.1) -> float:
        coef_mat_r = np.copy(coef_mat.ravel())
        coef_mat_hat_r = np.copy(coef_mat_hat.ravel())

        coef_mat_r[coef_mat_r > threshold] = 1
        coef_mat_r[coef_mat_r < threshold] = 0
        coef_mat_hat_r[coef_mat_hat_r > threshold] = 1
        coef_mat_hat_r[coef_mat_hat_r < threshold] = 0

        return float(sum(coef_mat_r == coef_mat_hat_r) / len(coef_mat_r))

    def _calculate_accuracy(self, coef_mat: np.ndarray, coef_mat_hat: np.ndarray) -> float:
        coef_mat_r = coef_mat.ravel()
        coef_mat_hat_r = coef_mat_hat.ravel()

        return np.sqrt(np.mean((coef_mat_r - coef_mat_hat_r) ** 2))

    def _calculate_predictions_accuracy(self, y: np.ndarray, y_hat: np.ndarray):
        y_r = y.ravel()
        y_hat_r = y_hat.ravel()

        return np.sqrt(np.mean((y_r - y_hat_r) ** 2))


