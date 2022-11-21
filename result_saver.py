import os
import json
from datetime import datetime

import utils


class ResultSaver:
    def __init__(self, config):
        self.config = config
        self.storage_buffer = {} # lists have exp_id:dict_of_results_per_experiment

    def add_results_to_buffer(self, experiment_config: utils.dotdict, results: dict, data_flag=True):
        '''
        Adds results of the current experiment to the buffer
        :param experiment_config: Config parameters for current experiment
        :param results: Results of the current experiment
        '''
        if experiment_config.exp_id not in self.storage_buffer.keys():
            self.storage_buffer[experiment_config.exp_id] = {}
            self.storage_buffer[experiment_config.exp_id]['experiment_config'] = experiment_config
        results.update({'date': datetime.now().strftime("%d-%m-%Y %H:%M:%S")})
        self.storage_buffer[experiment_config.exp_id]['data' if data_flag else 'model_results'] = results

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

    def empty_buffer(self):
        self.storage_buffer = {}
