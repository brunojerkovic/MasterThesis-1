from abc import ABC, abstractmethod
import numpy as np
import random

from result_saver import ResultSaver


class DataGenerator(ABC):
    def __init__(self, config):
        self.seed = config.seed
        self.tvt_split = config.tvt_split
        self.timesteps = config.timesteps
        self.lag = 1
        self.n_data = config.n_data
        self.burn_in = config.burn_in
        self.normalize_flag = config.normalize_data
        self.norm_range = (config.norm_range_min, config.norm_range_max)
        self.stationarity_radius = config.stationarity_radius
        self.config = config

        self.trainset_size = config.trainset_size
        self.valid_size = int(config.trainset_size * config.tvt_split)
        self.eval_size = int(config.trainset_size * config.tvt_split)
        self.time = self.trainset_size + self.valid_size + self.eval_size + 2 * self.timesteps

    @abstractmethod
    def _generate_series(self) -> tuple:
        '''
        :return: A tuple of numpy array of series and numpy array of coef_mat
        '''
        pass

    def generate(self, result_saver: ResultSaver):
        # Create seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Generate datagenerators and adjacency matrix
        series, coef_mat = self._generate_series()

        # Normalize datagenerators if required
        series = self._normalize(series) if self.normalize_flag else series

        # Create binary edges matrix from coefficient matrix
        edges = self._generate_edges(coef_mat)

        # Save data to the results
        result_saver.add_results_to_buffer(self.config, {
            'edges': edges.tolist(),
            'coef_mat': coef_mat.tolist()
        })

        return series, coef_mat, edges

    def _make_coef_stationary(self, coef_mat):
        eigvals = np.linalg.eigvals(coef_mat)
        max_eigs = np.abs(eigvals).max()
        if max_eigs > self.stationarity_radius:
            self._make_coef_stationary(0.95 * coef_mat)
        else:
            return coef_mat

    def _normalize(self, series):
        norm_min, norm_max = self.norm_range
        feats_min, feats_max = series.flatten().min(), series.flatten().max()

        feats_unit = (series - feats_min) / (feats_max - feats_min)
        feats_norm = feats_unit * (norm_max - norm_min) + norm_min

        return feats_norm

    def _generate_edges(self, coef_mat):
        """
        Generate adjacency matrices for node features
        n_sims: number of simulations
        num_nodes: number of nodes
        """
        edges = np.array(coef_mat)
        edges[edges > 0.] = 1.

        return edges
