import yaml
import itertools

keyword = 'FOLLOW_'


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def deepcopy_lvl1(d1: dict):
    d2 = {k1: dotdict(v1) if type(v1) == dict else v1 for (k1,v1) in d1.items()}
    return dotdict(d2)

def parse_config():
    with open('configs/config.yaml') as f:
        config = dotdict(yaml.load(f, Loader=yaml.FullLoader))
    with open('configs/config_main.yaml') as f:
        config_main = dotdict(yaml.load(f, Loader=yaml.FullLoader))
    config.update(config_main)

    return config

def generate_parameters(config):
    params = {k:v for k,v in config.items() if type(v) == list and k != 'model'}
    params_m = config.model if type(config.model) == list else [config.model] # So that the model is on the end
    if len(params) == 0:
        return params
    lenghts = [len(params_m)] + [len(param) for param in params.values()]
    max_len = max(lenghts)

    # Change it for 'model' to be at first place
    params_keys = ['model'] + list(params.keys())
    params_values = [params_m] + list(params.values())

    # Generate pointers
    pointers_ = list(itertools.product(list(range(max_len)), repeat=len(lenghts)))
    pointers = [p for p in pointers_ if all([idx < len(param) for (idx,param) in zip(p, params_values)])]

    # Generate all possibe hyperparameter values
    param_combinations = []
    for pointer in pointers:
        combination = {k:(params[k][i] if k!='model' else params_m[i]) for (k,i) in zip(params_keys, pointer)}
        combination.update({k:combination[v[len(keyword):]] for (k,v) in combination.items() if type(v) == str and v.startswith(keyword)}) # UPDATE: key: [val1, val2, FOLLOW_k2]
        combination.update({k:combination[v[len(keyword):]] for (k,v) in config.items() if type(v) == str and v.startswith(keyword)}) # UPDATE: key:FOLLOW_key2
        param_combinations.append(combination)

    return param_combinations

def get_nested_config(experiment_config: dotdict):
    '''
    To get the FOLLOWing values of the nested config
    :param experiment_config: config for current experiment
    :return: updated config for the current experiment
    '''

    # Follow the values inside the dict of models
    for k1, v1 in experiment_config.items():
        if type(v1) == dict or type(v1) == dotdict:
            for k2, v2 in v1.items():
                if type(v2) == str and v2.startswith(keyword):
                    experiment_config[k1][k2] = experiment_config[v2[len(keyword):]]
    return experiment_config

def get_saving_idxs(param_combinations):
    saving_idxs = {len(param_combinations) - 1} # You will want to save at the last index

    # Create a list of indexes at which the results will be saved (the ones before the model chagnes)
    curr_model = param_combinations[0]['model']
    for idx, param_combination in enumerate(param_combinations):
        if param_combination['model'] != curr_model:
            saving_idxs.add(idx-1)
            curr_model = param_combination['model']

    return saving_idxs
