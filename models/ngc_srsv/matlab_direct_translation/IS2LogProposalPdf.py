import numpy as np


def IS2LogProposalPdf(distribution, theta):
    log_pdf = np.log(pdf(distribution, theta))
    return log_pdf