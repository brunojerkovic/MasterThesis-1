from models.model import Model
from models.nri.main import NRI
from models.ngc.main import NGC
from models.tvar.main import TVAR

def model_selector(config, result_saver) -> Model:
    MyModel = None
    if config.model == 'nri' or config.model == 1:
        MyModel = NRI
        config.update(config.nri)
    elif config.model == 'ngc' or config.model == 2:
        MyModel = NGC
        config.update(config.ngc)
    elif config.model == 'ngc0' or config.model == 3:
        MyModel = NGC
        config.update(config.ngc)
    elif config.model == 'tvar' or config.model == 4:
        MyModel = TVAR
        config.update(config.tvar)

    return MyModel(config, result_saver)