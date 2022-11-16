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

# Using different models and data generators
To use different models and data generators, change the attributes: _loader_ and _model_ from _config_main.yaml_ file.