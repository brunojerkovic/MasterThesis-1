

def IS2_proposal(theta, distribution):
    if distribution == 'GM':
        options = statset('MaxIter', 1_000)
        GMModel = fitmdist(theta, 3, 'RegularizationValue', 0.1, 'Options', options)
    return GMModel