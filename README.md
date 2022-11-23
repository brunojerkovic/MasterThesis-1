# Thesis

## Prerequisites
In _config.yaml_, change the _r_bin_path_ to the location where R is installed on your computer. \
This project was implemented with Python 3.9, but it should work for Python 3.6 and higher. \
For the library dependencies, install dependencies from _requirements.txt_.

## Config files
The entire project can be ran through the 2 _yaml_ config files saved in the _config_ directory. \
There are 2 config files so that the parameters do not get cluttered inside one file. Parameters from _config_main.yaml_ file are more likely to be changed a lot through experiments, and parameters from _config.yaml_ file mostly stay the same regardless of the experiment.
To create different experiments, change the values of the _yaml_ file. \
You can run multiple experiments if the values of the parameters are inside square brackets. \
Ex: \
\
**c11**: [0.9, 0.8] \
\
However, if there are multiple parameters with more than 1 value, the programme will run all possible experiments. \
Meaning, for the following parameter setting: \
\
**c11**: [0.9, 0.8] \
**seed**: [1,2,3,4,5] \
\
the programme will run 2*5=10 experiments. \

# Parameter Explanation
To use different models and data generators, change the attributes: _loader_ and _model_ from _config_main.yaml_ file. \
Parameters: _c11,c12,c21,c22_ are elements of the 2-D coefficient matrix for the data generation of 2 variables. They are located inside _config_main.yaml_ file. \
Other parameters such as the size of the training set (_trainset_size_), train-validation-test split (_tvt_split_), or values of the noise coefficient matrices (_noise_eta_off_diag, _noise_eta_on_diag_, _noise_sigma_off_diag_, _noise_sigma_on_diag_) are located inside _config.yaml_ file. \

# Make Parameter Follow a Value of Another Parameter
If you want to make a parameter follow the value of another parameter, write _FOLLOW_paramname_ as a value of the parameter that has to follow another parameter. \
Example: \
**c11**: [0.1, 0.2, 0.3] \
**c22**: FOLLOW_c11 \
\
Example 2: \
**sigma_eta_diag**: [0.01, 0.02] \
**sigma_eta_off_diag**: [0, FOLLOW_sigma_eta_diag] \
\

# Config values for each experiment
## Experiment 0
Inside _config.yaml_: \
**trainset_size**: [500,1000,2000,3000,4000,5000] \
**tvt_split**: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] \
**sigma_eta_diag**: 0.01 \
**sigma_eta_off_diag**: 0 \
**n_data**: 2

Inside _config_main.yaml_: \
**experiment**: 0 \
**c11**: 0.9 \
**c12**: 0.9 \
**c21**: 0 \
**c22**: 0.9 \
**seed**: [1, 101, 33, 12, 0] \
**loader**: var \
**model**: [nri, ngc, ngc0, tvar]

## Experiment 1a
Inside _config.yaml_: \
**trainset_size**: 3000 \
**tvt_split**: 0.8 \
**sigma_eta_diag**: 0.01 \
**sigma_eta_off_diag**: 0 \
**n_data**: 2

Inside _config_main.yaml_: \
**experiment**: 1a \
**c11**: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] \
**c12**: 0 \
**c21**: 0 \
**c22**: FOLLOW_c11 \
**seed**: [1, 101, 33, 12, 0] \
**loader**: var \
**model**: [nri, ngc, ngc0, tvar]

## Experiment 1b
Inside _config.yaml_: \
**trainset_size**: 3000 \
**tvt_split**: 0.8 \
**sigma_eta_diag**: 0.01 \
**sigma_eta_off_diag**: 0 \
**n_data**: 2

Inside _config_main.yaml_: \
**experiment**: 1b \
**c11**: [0.8] \
**c12**: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] \
**c21**: 0 \
**c22**: FOLLOW_c11 \
**seed**: [1, 101, 33, 12, 0] \
**loader**: var \
**model**: [nri, ngc, ngc0, tvar]

## Experiment 1c
Inside _config.yaml_: \
**trainset_size**: 3000 \
**tvt_split**: 0.8 \
**sigma_eta_diag**: 0.01 \
**sigma_eta_off_diag**: 0 \
**n_data**: 2

Inside _config_main.yaml_: \
**experiment**: 1c \
**c11**: [0.8] \
**c12**: 0 \
**c21**: [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] \
**c22**: FOLLOW_c11 \
**seed**: [1, 101, 33, 12, 0] \
**loader**: var \
**model**: [nri, ngc, ngc0, tvar]

## Experiment 1d
Inside _config.yaml_: \
**trainset_size**: 3000 \
**tvt_split**: 0.8 \
**sigma_eta_diag**: 0.01 \
**sigma_eta_off_diag**: 0 \
**n_data**: 2

Inside _config_main.yaml_: \
**experiment**: 1d \
**c11**: [0.4] \
**c12**: [0, 0.1, 0.2, 0.3, 0.4, 0.5] \
**c21**: FOLLOW_c12 \
**c22**: FOLLOW_c11 \
**seed**: [1, 101, 33, 12, 0] \
**loader**: var \
**model**: [nri, ngc, ngc0, tvar]

## Experiment 2
Inside _config.yaml_: \
**trainset_size**: 3000 \
**tvt_split**: 0.8 \
**sigma_eta_diag**: [0.01, 0.02] \
**sigma_eta_off_diag**: [0, FOLLOW_sigma_eta_diag] \
**n_data**: 2

Inside _config_main.yaml_: \
**experiment**: 2 \
**c11**: [0.8] \
**c12**: [0, 0.8] \
**c21**: [0] \
**c22**: FOLLOW_c11 \
**seed**: [1, 101, 33, 12, 0] \
**loader**: var \
**model**: [nri, ngc, ngc0, tvar]

## Experiment 3
Inside _config.yaml_: \
**trainset_size**: 3000 \
**tvt_split**: 0.8 \
**sigma_eta_diag**: 0.01 \
**sigma_eta_off_diag**: 0 \
**n_data**: [2,3,4,5,6,7,8,9,10]

Inside _config_main.yaml_: \
**experiment**: 3 \
**c11**: [0] \
**c12**: [0] \
**c21**: [0] \
**c22**: [0] \
**seed**: [1, 101, 33, 12, 0] \
**loader**: ngc_var \
**model**: [nri, ngc, ngc0, tvar]

## Experiment 4
Inside _config.yaml_: \
**trainset_size**: 3000 \
**tvt_split**: 0.8 \
**sigma_eta_diag**: 0.01 \
**sigma_eta_off_diag**: 0 \
**n_data**: 10 \
**sparsity**: [0.1, 0.2 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

Inside _config_main.yaml_: \
**experiment**: 4 \
**c11**: [0] \
**c12**: [0] \
**c21**: [0] \
**c22**: [0] \
**seed**: [1, 101, 33, 12, 0] \
**loader**: ngc_var \
**model**: [nri, ngc, ngc0, tvar]

## Experiment 5a
Inside _config.yaml_: \
**trainset_size**: 3000 \
**tvt_split**: 0.8 \
**sigma_eta_diag**: 0.01 \
**sigma_eta_off_diag**: 0 \
**n_data**: 2 \
**sigma_eps_diag**: 1 \
**sigma_eps_off_diag**: 0.6

Inside _config_main.yaml_: \
**experiment**: 5a \
**c11**: [0.9] \
**c12**: [0.4] \
**c21**: [0.9] \
**c22**: [0] \
**seed**: [1, 101, 33, 12, 0] \
**loader**: svm \
**model**: [nri, ngc, ngc0, tvar]

## Experiment 5b
Inside _config.yaml_: \
**trainset_size**: 3000 \
**tvt_split**: 0.8 \
**sigma_eta_diag**: 0.01 \
**sigma_eta_off_diag**: 0 \
**n_data**: 2 \
**sigma_eps_diag**: 1 \
**sigma_eps_off_diag**: 0.6

Inside _config_main.yaml_: \
**experiment**: 5b \
**c11**: [0.9] \
**c12**: [0] \
**c21**: [0.9] \
**c22**: [0.4] \
**seed**: [1, 101, 33, 12, 0] \
**loader**: svm \
**model**: [nri, ngc, ngc0, tvar]

## Experiment 5c
Inside _config.yaml_: \
**trainset_size**: 3000 \
**tvt_split**: 0.8 \
**sigma_eta_diag**: 0.01 \
**sigma_eta_off_diag**: 0 \
**n_data**: 2 \
**sigma_eps_diag**: 1 \
**sigma_eps_off_diag**: 0.6

Inside _config_main.yaml_: \
**experiment**: 5c \
**c11**: [0.9] \
**c12**: [0] \
**c21**: [0.9] \
**c22**: [0] \
**seed**: [1, 101, 33, 12, 0] \
**loader**: svm \
**model**: [nri, ngc, ngc0, tvar]

## Experiment 5d
Inside _config.yaml_: \
**trainset_size**: 3000 \
**tvt_split**: 0.8 \
**sigma_eta_diag**: 0.01 \
**sigma_eta_off_diag**: 0 \
**n_data**: 2 \
**sigma_eps_diag**: 1 \
**sigma_eps_off_diag**: 0.6

Inside _config_main.yaml_: \
**experiment**: 5d \
**c11**: [0.8] \
**c12**: [0.1] \
**c21**: [0.8] \
**c22**: [0.1] \
**seed**: [1, 101, 33, 12, 0] \
**loader**: svm \
**model**: [nri, ngc, ngc0, tvar]
