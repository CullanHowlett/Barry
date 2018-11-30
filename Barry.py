# The main driver routine for fitting the BAO feature in some data. The idea is that you change this depending on you use case.
# It has a couple of examples for fitting some data either via the method used for SDSS BOSS (traditional) or out preferred method (new)

import sys
from read_data import *
from powerspectrum import *
from fitting_routines import *
from models import *
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager
from matplotlib import gridspec

# Fit alpha to some data using the SDSS style Polynomial method, with a list over different alpha values. For each fixed value of alpha, we find the best-fitting polynomial
# terms. We use an Eisenstein and Hu dewiggled power spectrum, CAMB value for the sound horizon and Gaussian likleihood corrected for a finite number of simulations. We fix
# the value of sigma_nl if fitting the correlation function, and allow it to vary with a Gaussian prior of width 2Mpc/h for the power spectrum
def fit_data_traditional(dataflag, binwidth, matterfile, datafile, covfile, winfile, winmatfile, chainfile, xmin, xmax):

    # Read in the data (and window function if required)
    data = InputData(dataflag, nmocks=1000)
    data.read_data(datafile, covfile, xmin=xmin, xmax=xmax, nconcat=binwidth, winfile=winfile, winmatfile=winmatfile)

    # Read in the linear matter power spectrum and choose the type of smooth power spectrum model we want
    power = EH98(matterfile,r_s=147.17)

    # Set up the type of model we want. Can be one of "Polynomial" or "FullShape". We will add "LinearPoint" and "BAOExtractor" later.
    if (dataflag):
        model = Polynomial(dataflag, power, prepare_model_flag=True)
        model.set_prior("sigma_nl", ["Gaussian", 11.32, 2.0])
    else:
        model = Polynomial(dataflag, power, free_sigma_nl=False, prepare_model_flag=True)
        model.sigma_nl = 11.7

    # Fit the data iterating over a list of alpha values
    fitter = List(data, model, 0.7, 1.3, 600, liketype="Hartlap2007", do_plot=1)
    fitter.fit_data()

    return

# Fit the data using an MCMC chain, with a dewiggled power spectrum calculated by smoothing the input matter power spectrum and fully 
# marginalising over the measured sample covariance matrix following Sellentin & Heavens 2016. We fix the value of sigma_nl if fitting 
# the correlation function, and allow it to vary with a Gaussian prior of width 2Mpc/h for the power spectrum
def fit_data_new(dataflag, binwidth, matterfile, datafile, covfile, winfile, winmatfile, chainfile, xmin, xmax):

    # Read in the linear matter power spectrum and choose the type of smooth power spectrum model we want
    power = Hinton2017(matterfile, r_s = 147.17)

    # Set up the type of data and model we want. Can be one of "Polynomial" or "FullShape". We will add "LinearPoint" and "BAOExtractor" later.
    if (dataflag == 0):
        data = CorrelationFunction(nmocks=1000).read_data(datafile=datafile, covfile=covfile, xmin=xmin, xmax=xmax)
        model = Polynomial("CorrelationFunction", power, free_sigma_nl=False, prepare_model_flag=True)
        #model = FullShape("CorrelationFunction", power, free_sigma_nl=False, nonlinearterms="compute_pt_integrals_output.dat")
        #model = LinearPoint(power, polyorder=9, lpoint=0.5, offset=0.0)
        model.sigma_nl = 11.7
    elif (dataflag == 1):
        data = PowerSpectrum(nmocks=1000, verbose=True).read_data(datafile=datafile, covfile=covfile, xmin=xmin, xmax=xmax, nconcat=binwidth, winfile=winfile, winmatfile=winmatfile)
        #model = Polynomial("PowerSpectrum", power, free_sigma_nl=True, prepare_model_flag=True, verbose=True)
        model = FullShape("PowerSpectrum", power, free_sigma_nl=True, nonlinearterms="compute_pt_integrals_output.dat", verbose=True)
        #model.set_prior("sigma_nl", ["Gaussian", 12.5, 2.0])
    elif (dataflag == 2):
        data = BAOExtract(nmocks=1000, verbose=True).read_data(datafile=datafile, covfile=covfile, xmin=xmin, xmax=xmax, nconcat=binwidth, winfile=winfile, winmatfile=winmatfile).extract_BAO(power.r_s)
        model = BAOExtractor(power, free_sigma_nl=True, nonlinearterms="compute_pt_integrals_output.dat", verbose=True)
    else:
        print "dataflag value not supported, ", dataflag
        exit()

    # Fit the data iterating over a list of alpha values
    fitter = MCMC_emcee(data, model, niterations=4000, liketype="SH2016", do_plot=1, startfrombestfit=False, outputfile=chainfile)
    fitter.fit_data()

    return

if __name__ == "__main__":

    # This routine is set up to do embarrasingly parallel runs of the BAO_fitter over a number of
    # mock galaxy catalogues. Each process (ID) does 'nfits' BAO fits starting with mock 'firstmock'

    dataflag = int(sys.argv[1])    # Whether to fit the correlation function (0), power spectrum (1), or BAO extract (otherwise)
    binwidth = int(sys.argv[2])
    mocknum = int(sys.argv[3])

    # Filenames
    winfile = []
    winmatfile = []
    matterfile = '/Volumes/Work/ICRAR/TAIPAN/camb_TAIPAN_matterpower_linear.dat'   # The linear matter power spectrum (i.e., from CAMB)
    if (dataflag == 0): 
        datafile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1_R%d.xi_%d' % (mocknum, binwidth))
        covfile =  str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1.xi_%d_cov' % binwidth)       # The covariance matrix
        chainfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/BAOfits/BAO_Mock_taipan_year1_v1_R%d_xi_%d' % (mocknum, binwidth))   # The file in which to store the output MCMC chain
    elif (dataflag == 1):          
        datafile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1_R%d.lpow_blake' % (mocknum))                        # The data file
        covfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1.lpow_%d_0p02-0p30_cov' % binwidth)       # The covariance matrix 
        winfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/taipanmock_year1_mock_rand_cullan.lwin')                         #    Power spectrum (for the integral constraint)
        winmatfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/taipanmock_year1_mock_rand_cullan.winfit_%d' % binwidth)      #    Convolution matrix (for convolving the model)
        chainfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/BAOfits/BAO_Mock_FullShape_FreeSigmaNL_taipan_year1_v1_R%d_lpow_0p02-0p30_%d' % (mocknum, binwidth))   # The file in which to store the output MCMC chain
    elif (dataflag == 2):          
        datafile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1_R%d.lpow_blake' % (mocknum))                        # The data file
        covfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1.rp_1_0p5_%d_0p02-0p30_cov' % binwidth)       # The covariance matrix 
        winfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/taipanmock_year1_mock_rand_cullan.lwin')                         #    Power spectrum (for the integral constraint)
        winmatfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/taipanmock_year1_mock_rand_cullan.winfit_%d' % binwidth)      #    Convolution matrix (for convolving the model)
        chainfile = str('/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/BAOfits/BAO_Mock_BAOExtractor_FreeSigmaNL_taipan_year1_v1_R%d_rp_1_0p5_0p02-0p30_%d' % (mocknum, binwidth))   # The file in which to store the output MCMC chain

    if (dataflag):
        xmin = 0.02
        xmax = 0.30
    else:
        xmin = 30.0
        xmax = 200.0
    #fit_data_traditional(dataflag, binwidth, matterfile, datafile, covfile, winfile, winmatfile, chainfile, xmin, xmax)
    fit_data_new(dataflag, binwidth, matterfile, datafile, covfile, winfile, winmatfile, chainfile, xmin, xmax)


