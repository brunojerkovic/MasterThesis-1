from models.model import Model
from models.nri.main import NRI
from models.ngc.main import NGC
from models.tvar.main import TVAR
from models.ngc_noise.main import NGC as NGCNoise
from models.ngc_srsv.main import NGC as NGC_SVSR

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
    elif config.model == 'ngc_noise' or config.model == 5:
        MyModel = NGCNoise
        config.update(config.ngc)
    elif config.model == 'ngc_srsv' or config.model == 6:
        MyModel = NGC_SVSR
        config.update(config.ngc_svsr)

    return MyModel(config, result_saver)