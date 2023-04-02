import utils
from datagenerators import DataGenerator, VARGenerator, NGCVARGenerator, SVMGenerator, StockGenerator, NGCSVMGenerator


def generator_selector(config: utils.dotdict) -> DataGenerator:
    Generator = None
    if config.loader == 'var' or config.loader == 1:
        Generator = VARGenerator
    elif config.loader == 'svm' or config.loader == 2:
        Generator = SVMGenerator
    elif config.loader == 'ngc_var' or config.loader == 3:
        Generator = NGCVARGenerator
    elif config.loader == 'stocks' or config.loader == 4:
        Generator = StockGenerator
    elif config.loader == 'ngc_svm' or config.loader == 5:
        Generator = NGCSVMGenerator

    return Generator(config)
