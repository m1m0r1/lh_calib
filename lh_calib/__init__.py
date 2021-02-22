
def get_custom_objects():
    from .calibration import dirichlet_multinomial_nll
    from .calibration import VectorScaling
    from .calibration import TemperatureScaling
    from .calibration import OffDiagonalRegularizer
    #from .calibration import TrainOnlyRegularizer  # TODO
    # from .calibration import * not allowed
    obj = locals()
    obj['<lambda>'] = lambda : None  # dummy function for unused objects on prediction (hack)
    return obj
