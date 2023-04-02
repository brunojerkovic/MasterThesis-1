from scipy import io
from models.ngc_srsv.matlab_direct_translation.lstmSV import LSTMSV
from models.ngc_srsv.matlab_direct_translation.BPM import BPM
from models.ngc_srsv.matlab_direct_translation.ParticleFilter import ParticleFilter
import os
from datetime import datetime


def run():
    # Load data
    dataset_name = 'SP500_weekly'
    data = io.loadmat(f'data/{dataset_name}.mat')
    T = 1000
    y = data['y'][:T]

    # Define a save name to STORE the RESULTS during sampling phase (zapisi verziju i datum)
    date_now = [datetime.today().month, datetime.today().day, datetime.today().hour, datetime.today().minute, datetime.today().second]
    name = 'Results_lstmSV_' + dataset_name + '_' + '_'.join([str(x) for x in date_now])

    # Bayesian inference
    # Create a lstmSV object with default properties
    model = LSTMSV()

    # Create a Blocking Pseudo-Marginal object, setting random seed property
    # sampler = BPM(seed=1, save_after=100, num_mcmc=100_000)
    sampler = ParticleFilter(model, y)

    # Estimate using BPM
    # lstmSV_fit = sampler.estimate(model, y)
    lstmSV_fit = sampler.estimate()

    print("LOSS (negative likelihood)", lstmSV_fit)

    # Estimate marginal likelihood with IS2
    #lstmSV_fit.Post.IS2 = IS2(y, lstmSV_fit, num_particle=1_000, num_is_particle=5_000, burnin=10_000, seed=1)

    #print("Marginal likelihood", lstmSV_fit.Post.IS2.Marllh)

run()