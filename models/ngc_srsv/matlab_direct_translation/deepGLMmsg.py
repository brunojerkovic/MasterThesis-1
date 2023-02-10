def deepGLMmsg(identifier):
    if identifier == 'deepglm:TooFewInputs':
        return 'At least two arguments are specified'
    elif identifier == 'deepglm:InputSizeMismatchX':
        return 'X and Y must have the same number of observations'
    elif identifier == 'deepglm:InputSizeMismatchY':
        return 'Y must be a single column vector'
    elif identifier == 'deepglm:ArgumentMustBePair':
        return 'Optinal arguments must be pairs'
    elif identifier == 'deepglm:ResponseMustBeBinary':
        return 'Two level categorical variable required'
    elif identifier == 'deepglm:DistributionMustBeBinomial':
        return 'Binomial distribution option required'
    elif identifier == 'deepglm:MustSpecifyActivationFunction':
        return 'Activation function type requied'