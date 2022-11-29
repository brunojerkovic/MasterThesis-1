import utils
from datagenerators.generator_selector import generator_selector
from models.model_selector import model_selector
from result_saver import ResultSaver


def main():
    config = utils.parse_config()
    parameter_combinations = utils.generate_parameters(config) # When there are multiple parameter options
    saving_idxs = utils.get_saving_idxs(parameter_combinations)
    result_saver = ResultSaver(config)

    for exp_id, exp_params in enumerate(parameter_combinations):
        # Parameters for this experiment
        experiment_config = utils.deepcopy_lvl1(config)
        experiment_config.update(exp_params)
        experiment_config = utils.get_nested_config(experiment_config)
        experiment_config.exp_id = exp_id

        # Choose a data generator and get the data
        generator = generator_selector(experiment_config)
        series, coef_mat , edges = generator.generate(result_saver)

        # Choose a model and get the results
        model = model_selector(experiment_config, result_saver)
        accuracy = model.algorithm(series, coef_mat, edges)

        print(f"Accuracy for experiment id {exp_id} is {accuracy}")

        # Save the results (only at saving idxs)
        result_saver.save(experiment_config)
        if exp_id in saving_idxs:
            result_saver.empty_buffer()

if __name__ == '__main__':
    main()
