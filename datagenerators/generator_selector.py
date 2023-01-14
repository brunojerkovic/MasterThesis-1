import utils
from datagenerators.generator import DataGenerator
from datagenerators.generate_var import VARGenerator
from datagenerators.generate_ngc_var import NGCVARGenerator
from datagenerators.generate_svm import SVMGenerator
from datagenerators.generate_stock_data import StockGenerator


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

    return Generator(config)
