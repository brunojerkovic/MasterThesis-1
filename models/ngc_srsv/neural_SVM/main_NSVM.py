from models.ngc_srsv.neural_SVM.neural_nets import train


def main_nsvm(X):
    results = {}

    results = train(X)
    

    return results