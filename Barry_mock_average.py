# The main driver routine for fitting the BAO feature to the average of some mock data. There are two examples here, one using the traditional method with a Polnomial model
# and a second using a FullShape model and more up to date method for computing the smooth power spectrum and using a fully Bayesian correction for the covariance matrix. 

import sys
sys.path.append("./src")
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

# Find the best fit parameters for the average of mock data using a traditional Polynomial method, with a Eisenstein and Hu dewiggled power spectrum, CAMB value for the 
# sound horizon and Gaussian likelihood corrected for a finite number of simulations. We allow both alpha and sigma_nl to vary, and this is used primarily to find the 
# value of sigma_nl to fix to when fitting the data/mocks individually.
def fit_mockaverage_traditional(dataflag, matterfile, datafile, covfile, winfile, winmatfile, xmin, xmax):

    # Read in the data (and window function if required)
    data = InputData(dataflag, nmocks=1000)
    data.read_data(datafile, covfile, xmin=xmin, xmax=xmax, nconcat=binwidth, winfile=winfile, winmatfile=winmatfile)

    # Read in the linear matter power spectrum and choose the type of smooth power spectrum model we want
    power = EH98(matterfile, r_s=147.17)

    # Set up the type of model we want. Can be one of "Polynomial" or "FullShape" or "LinearPoint". We will add "BAOExtractor" later.
    model = Polynomial(dataflag, power)

    # Fit the data using a single optimization ("Optimizer")
    fitter = Optimizer(data, model, liketype="Hartlap2007", do_plot=1)
    fitter.fit_data()

    return

# Find the best fit parameters for the average of mock data using a more up-to-date method, with a dewiggled power spectrum calculated by smoothing the 
# input matter power spectrum and fully marginalising over the measured sample covariance matrix following Sellentin & Heavens 2016.
def fit_mockaverage_new(dataflag, matterfile, datafile, covfile, winfile, winmatfile, xmin, xmax):

    power = Hinton2017(matterfile, r_s = 147.17)

    # Set up the type of data and model we want. Can be one of "Polynomial" or "FullShape". We will add "LinearPoint" and "BAOExtractor" later.
    if (dataflag == 0):
        data = CorrelationFunction(nmocks=1000).read_data(datafile=datafile, covfile=covfile, xmin=xmin, xmax=xmax)
        model = Polynomial("CorrelationFunction", power, free_sigma_nl=True, prepare_model_flag=True)
        #model = FullShape("CorrelationFunction", power, free_sigma_nl=False, nonlinearterms="compute_pt_integrals_output.dat")
        #model = LinearPoint(power, polyorder=9, lpoint=0.5, offset=0.0)
    elif (dataflag == 1):
        data = PowerSpectrum(nmocks=1000, verbose=True).read_data(datafile=datafile, covfile=covfile, xmin=xmin, xmax=xmax, winmatfile=winmatfile)
        data.read_data(winfile=winfile, xmin=xmin, xmax=xmax, nconcat=binwidth)
        model = Polynomial("PowerSpectrum", power, free_sigma_nl=True, prepare_model_flag=True, verbose=True)
        #model = FullShape("PowerSpectrum", power, free_sigma_nl=True, nonlinearterms="./files/compute_pt_integrals_output.dat", verbose=True)
    elif (dataflag == 2):
        data = BAOExtract(nmocks=1000, verbose=True).read_data(datafile=datafile, covfile=covfile, xmin=xmin, xmax=xmax, winmatfile=winmatfile)
        data.read_data(winfile=winfile, xmin=xmin, xmax=xmax, nconcat=binwidth)
        model = BAOExtractor(power, free_sigma_nl=True, nonlinearterms="./files/compute_pt_integrals_output.dat", verbose=True)
    else:
        print "dataflag value not supported, ", dataflag
        exit()

    # Fit the data using either an MCMC ("MCMC") routine or iterating over a list of alpha values ("list")
    fitter = MCMC_emcee(data, model, niterations=4000, liketype="SH2016", do_plot=0, startfrombestfit=True, outputfile=chainfile)
    fitter.fit_data()

    return

if __name__ == "__main__":

    dataflag = int(sys.argv[1])    # Whether to fit the correlation function (0), power spectrum (1), or BAO extract (2)
    binwidth = int(sys.argv[2])

    # Filenames
    winfile = []
    winmatfile = []
    matterfile = './files/test_files/camb_TAIPAN_matterpower_linear.dat'   # The linear matter power spectrum (i.e., from CAMB)
    if (dataflag == 0): 
        datafile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.xi_%d_ave_30-200' % binwidth)
        covfile =  str('./files/test_files/mock_average/Mock_taipan_year1_v1.xi_%d_cov_30-200' % binwidth)       # The covariance matrix
        chainfile = str('./files/test_files/BAOfits/BAO_MockAverage_taipan_year1_v1_xi_30-200_%d' % binwidth)   # The file in which to store the output MCMC chain
    elif (dataflag == 1):          
        datafile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.lpow_%d_0p03-0p25_ave_recon' % binwidth)                        # The data file
        covfile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.lpow_%d_0p03-0p25_cov_recon' % binwidth)       # The covariance matrix 
        winfile = str('./files/test_files/taipanmock_year1_mock_rand_cullan.lwin')                         #    Power spectrum (for the integral constraint)
        winmatfile = str('./files/test_files/taipanmock_year1_mock_rand_cullan.winfit_%d' % binwidth)      #    Convolution matrix (for convolving the model)
        chainfile = str('./files/test_files/BAOfits/BAO_MockAverage_Polynomial_taipan_year1_v1_lpow_0p03-0p25_%d_recon' % binwidth)   # The file in which to store the output MCMC chain
    elif (dataflag == 2):          
        datafile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.rp_1_0p5_%d_0p02-0p30_ave' % binwidth)                        # The data file
        covfile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.rp_1_0p5_%d_0p02-0p30_cov' % binwidth)       # The covariance matrix 
        winfile = str('./files/test_files/taipanmock_year1_mock_rand_cullan.lwin')                         #    Power spectrum (for the integral constraint)
        winmatfile = str('./files/test_files/taipanmock_year1_mock_rand_cullan.winfit_%d' % binwidth)      #    Convolution matrix (for convolving the model)
        chainfile = str('./files/test_files/BAOfits/BAO_MockAverage_BAOExtractor_taipan_year1_v1_rp_1_0p5_0p02-0p30_%d' % binwidth)   # The file in which to store the output MCMC chain
    else:
        print "dataflag value not supported, ", dataflag
        exit()

    if (dataflag):
        xmin = 0.03
        xmax = 0.25
    else:
        xmin = 30.0
        xmax = 200.0
    #fit_mockaverage_traditional(dataflag, matterfile, datafile, covfile, winfile, winmatfile, xmin, xmax)
    fit_mockaverage_new(dataflag, matterfile, datafile, covfile, winfile, winmatfile, xmin, xmax)



