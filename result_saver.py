import os
import json
from datetime import datetime

import utils


class ResultSaver():
    def __init__(self, config):
        self.config = config
        self.storage_buffer = {} # lists have exp_id:dict_of_results_per_experiment

    def add_results_to_buffer(self, experiment_config: utils.dotdict, results: dict):
        '''
        Adds results of the current experiment to the buffer
        :param experiment_config: Config parameters for current experiment
        :param results: Results of the current experiment
        :return: None
        '''
        self.storage_buffer[experiment_config.exp_id] = experiment_config
        results.update({'date': datetime.now().strftime("%d-%m-%Y %H:%M:%S")})
        self.storage_buffer[experiment_config.exp_id]['model_results'] = results

    def save(self, experiment_config):
        # Save results to dir
        if self.config.save_in_dir:
            # Create directory if it does not exist already
            result_path = f"results/{experiment_config.experiment}/{experiment_config.model}"
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            # Save buffer to json
            with open(result_path + '/results.json', "w") as f:
                json.dump(self.storage_buffer, f, indent=4)

        # Empty out the buffer
        self.storage_buffer = {}
