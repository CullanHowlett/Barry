# The main driver routine for fitting the BAO feature to the average of some mock data. There are two examples here, one using the traditional method with a Polnomial model
# and a second using a FullShape model and more up to date method for computing the smooth power spectrum and using a fully Bayesian correction for the covariance matrix. 

import sys
from read_data import *
from powerspectrum import *
from fitting_routines import *
from plotting_routines import *
from models import *
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager
from matplotlib import gridspec

# Find the best fit parameters for the average of mock data using a more up-to-date method, with a dewiggled power spectrum calculated by smoothing the 
# input matter power spectrum and fully marginalising over the measured sample covariance matrix following Sellentin & Heavens 2016.
def plot_mockaverage(dataflag, matterfile, datafile, datafile_recon, covfile, covfile_recon, winfile, winmatfile, xmin, xmax, chainfile, chainfile_recon):

    # Read in the data (and window function if required)
    data = InputData(dataflag, nmocks=1000)
    data.read_data(datafile=datafile, covfile=covfile, xmin=xmin, xmax=xmax, winmatfile=winmatfile)
    if (dataflag):
        data.read_data(winfile=winfile, xmin=xmin, xmax=xmax, nconcat=binwidth)

    data_recon = InputData(dataflag, nmocks=1000)
    data_recon.read_data(datafile=datafile_recon, covfile=covfile_recon, xmin=xmin, xmax=xmax, winmatfile=winmatfile)
    if (dataflag):
        data_recon.read_data(winfile=winfile, xmin=xmin, xmax=xmax, nconcat=binwidth)

    # Read in the linear matter power spectrum and choose the type of smooth power spectrum model we want
    power = Hinton2017(matterfile=matterfile)

    # Set up the type of model we want. Can be one of "Polynomial" or "FullShape". We will add "LinearPoint" and "BAOExtractor" later.
    #model = Polynomial(dataflag, power, prepare_model_flag=False)
    #model_recon = Polynomial(dataflag, power, prepare_model_flag=False)
    model = FullShape(dataflag, power, prepare_model_flag=False, nonlinearterms="compute_pt_integrals_output.dat")
    model_recon = FullShape(dataflag, power, prepare_model_flag=False, nonlinearterms="compute_pt_integrals_output.dat")

    # Fit the data using either an MCMC ("MCMC") routine or iterating over a list of alpha values ("list")
    fitter = MCMC_emcee(data, model, outputfile=chainfile)
    fitter_recon = MCMC_emcee(data_recon, model_recon, outputfile=chainfile_recon)

    # Plot aspects of the MCMC chain
    params, max_params, walkers, samples, posterior = fitter.prepareforChainConsumer(burnin=1000)
    params_recon, max_params_recon, walkers_recon, samples_recon, posterior_recon = fitter_recon.prepareforChainConsumer(burnin=1000)

    print params

    free_params = model.get_all_params()
    model.prepare_model(fitter)
    for counter, i in enumerate(free_params):
        model.params[i][0] = max_params[counter]
    model.set_model(x=data.x)
    p = Plotter(data=data, model=model)

    free_params = model_recon.get_all_params()
    model_recon.prepare_model(fitter_recon)
    for counter, i in enumerate(free_params):
        model_recon.params[i][0] = max_params_recon[counter]
    model_recon.set_model(x=data_recon.x)
    p.add_data_to_plot(data=data_recon, markerfacecolor='w')
    p.add_model_to_plot(data_recon, model=model_recon, markerfacecolor='w', linecolor='r')
    p.display_plot() 

    c = ChainConsumer().add_chain(samples, parameters=params, walkers=walkers, posterior=posterior, cloud=True, name="Pre-reconstruction").configure(summary=True)
    c.add_chain(samples_recon, parameters=params_recon, walkers=walkers_recon, posterior=posterior_recon, cloud=True, name="Post-reconstruction").configure(summary=True)
    print c.analysis.get_summary(), max_params, max_params_recon
    #c.plotter.plot(display=True)

    return

if __name__ == "__main__":

    dataflag = int(sys.argv[1])    # Whether to fit the correlation function (0) or power spectrum (otherwise)
    binwidth = int(sys.argv[2])

    # Filenames
    winfile = []
    winmatfile = []
    matterfile = '/Volumes/Work/ICRAR/TAIPAN/camb_TAIPAN_matterpower_linear.dat'   # The linear matter power spectrum (i.e., from CAMB)
    if (dataflag == 0): 
        datafile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1.xi_%d_ave_30-200' % binwidth)
        covfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1.xi_%d_cov_30-200' % binwidth)       # The covariance matrix
        chainfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/BAOfits/BAO_MockAverage_taipan_year1_v1_xi_30-200_%d' % binwidth)   # The file in which to store the output MCMC chain
        datafile_recon = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1.xi_%d_ave_30-200_recon' % binwidth)
        covfile_recon = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1.xi_%d_cov_30-200_recon' % binwidth)       # The covariance matrix
        chainfile_recon = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/BAOfits/BAO_MockAverage_taipan_year1_v1_xi_30-200_%d_recon' % binwidth)   # The file in which to store the output MCMC chain
    elif (dataflag == 1):          
        datafile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1.lpow_%d_0p02-0p30_ave' % binwidth)                        # The data file
        covfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1.lpow_%d_0p02-0p30_cov' % binwidth)       # The covariance matrix 
        winfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/taipanmock_year1_mock_rand_cullan.lwin')                         #    Power spectrum (for the integral constraint)
        winmatfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/taipanmock_year1_mock_rand_cullan.winfit_%d' % binwidth)      #    Convolution matrix (for convolving the model)
        chainfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/BAOfits/BAO_MockAverage_FullShape_taipan_year1_v1_lpow_0p02-0p30_%d' % binwidth)   # The file in which to store the output MCMC chain
        datafile_recon = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1.lpow_%d_0p02-0p30_ave_recon' % binwidth)                        # The data file
        covfile_recon = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1.lpow_%d_0p02-0p30_cov_recon' % binwidth)       # The covariance matrix 
        chainfile_recon = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/BAOfits/BAO_MockAverage_FullShape_taipan_year1_v1_lpow_0p02-0p30_%d_recon' % binwidth)   # The file in which to store the output MCMC chain

    if (dataflag):
        xmin = 0.02
        xmax = 0.30
    else:
        xmin = 30.0
        xmax = 200.0
    plot_mockaverage(dataflag, matterfile, datafile, datafile_recon, covfile, covfile_recon, winfile, winmatfile, xmin, xmax, chainfile, chainfile_recon)



