from models.ngc_srsv.matlab_direct_translation.Distribution import Distribution


class Parameter:
    def __init__(self, prior: Distribution=None, prior_transform=None,
                 prior_inv_transform=None, jacobian_offset=None, name=None):
        self.name = name
        self.prior = Distribution(name='normal', parameters=[0, 0.1]) if prior is None else prior
        self.prior_transform = (lambda x: x) if prior_transform is None else prior_transform
        self.prior_inv_transform = (lambda x: x) if prior_inv_transform is None else prior_inv_transform
        self.jacobian_offset = (lambda _: 0) if jacobian_offset is None else jacobian_offset

