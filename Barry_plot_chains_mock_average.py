# A code to read in the chains for fits to the mock average pre- and post-reconstruction, and produce plots of the best-fit models compared to the data,
# and the 2 sets of posterior samples

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

# Read in pre- and post-recon fits, set up some data and models and plot the best-fit model against the data. Then plot the posterior samples using ChainConsumer
def plot_mockaverage(dataflag, matterfile, datafile, datafile_recon, covfile, covfile_recon, winfile, winmatfile, xmin, xmax, chainfile, chainfile_recon):

    power = Hinton2017CAMB(redshift=0.11, mnu=0.0)

    # Set up the type of data and model we want. Can be one of "Polynomial" or "FullShape". We will add "LinearPoint" and "BAOExtractor" later.
    if (dataflag == 0):
        data = CorrelationFunction(nmocks=1000).read_data(datafile=datafile, covfile=covfile, xmin=xmin, xmax=xmax)
        data_recon = CorrelationFunction(nmocks=1000).read_data(datafile=datafile_recon, covfile=covfile_recon, xmin=xmin, xmax=xmax)
        model = Polynomial("CorrelationFunction", power, free_sigma_nl=True, prepare_model_flag=True)
        model_recon = Polynomial("CorrelationFunction", power, free_sigma_nl=True, prepare_model_flag=True)
    elif (dataflag == 1):
        data = PowerSpectrum(nmocks=1000, verbose=True).read_data(datafile=datafile, covfile=covfile, xmin=xmin, xmax=xmax, winmatfile=winmatfile)
        data.read_data(winfile=winfile, xmin=xmin, xmax=xmax, nconcat=binwidth)
        data_recon = PowerSpectrum(nmocks=1000, verbose=True).read_data(datafile=datafile_recon, covfile=covfile_recon, xmin=xmin, xmax=xmax, winmatfile=winmatfile)
        data_recon.read_data(winfile=winfile, xmin=xmin, xmax=xmax, nconcat=binwidth)
        model = FullShape("PowerSpectrum", power, free_sigma_nl=True, nonlinearterms="./files/compute_pt_integrals_output.dat", verbose=True)
        model_recon = FullShape("PowerSpectrum", power, free_sigma_nl=True, nonlinearterms="./files/compute_pt_integrals_output.dat", verbose=True)
    elif (dataflag == 2):
        data = BAOExtract(nmocks=1000, verbose=True).read_data(datafile=datafile, covfile=covfile, xmin=xmin, xmax=xmax, winmatfile=winmatfile)
        data.read_data(winfile=winfile, xmin=xmin, xmax=xmax, nconcat=binwidth)
        model = BAOExtractor(power, free_sigma_nl=True, nonlinearterms="./files/compute_pt_integrals_output.dat", verbose=True)
    else:
        print "dataflag value not supported, ", dataflag
        exit()

    # Read in the chains
    params, max_params, max_loglike, samples, loglike = prepareforChainConsumer(model, outputfile=chainfile, burnin=1000)
    params_recon, max_params_recon, max_loglike_recon, samples_recon, loglike_recon = prepareforChainConsumer(model_recon, outputfile=chainfile_recon, burnin=1000)

    free_params = model.get_all_params()
    prepare_model(data, model)
    for counter, i in enumerate(free_params):
        model.params[i][0] = max_params[counter]
    set_model(model, x=data.x)
    p = Plotter(data=data, model=model)

    free_params = model_recon.get_all_params()
    prepare_model(data_recon, model_recon)
    for counter, i in enumerate(free_params):
        model_recon.params[i][0] = max_params_recon[counter]
    set_model(model_recon, x=data_recon.x)
    p.add_data_to_plot(data=data_recon, markerfacecolor='w')
    p.add_model_to_plot(data_recon, model=model_recon, markerfacecolor='w', linecolor='r')
    p.display_plot() 

    c = ChainConsumer().add_chain(samples, parameters=params, posterior=loglike, cloud=True, name="Pre-reconstruction").configure(summary=True)
    c.add_chain(samples_recon, parameters=params_recon, posterior=loglike_recon, cloud=True, name="Post-reconstruction").configure(summary=True)
    print c.analysis.get_summary(), max_params, max_params_recon
    #c.plotter.plot(display=True)

    return

if __name__ == "__main__":

    dataflag = int(sys.argv[1])    # Whether to fit the correlation function (0), power spectrum (1), or BAO extract (2)
    binwidth = int(sys.argv[2])

    # Filenames
    winfile = []
    winmatfile = []
    matterfile = '/Volumes/Work/ICRAR/TAIPAN/camb_TAIPAN_matterpower_linear.dat'   # The linear matter power spectrum (i.e., from CAMB)
    if (dataflag == 0): 
        datafile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.xi_%d_ave_30-200' % binwidth)
        covfile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.xi_%d_cov_30-200' % binwidth)       # The covariance matrix
        chainfile = str('./files/test_files/BAOfits/BAO_MockAverage_taipan_year1_v1_xi_30-200_%d' % binwidth)   # The file in which to store the output MCMC chain
        datafile_recon = str('./files/test_files/mock_average/Mock_taipan_year1_v1.xi_%d_ave_30-200_recon' % binwidth)
        covfile_recon = str('./files/test_files/mock_average/Mock_taipan_year1_v1.xi_%d_cov_30-200_recon' % binwidth)       # The covariance matrix
        chainfile_recon = str('./files/test_files/BAOfits/BAO_MockAverage_taipan_year1_v1_xi_30-200_%d_recon' % binwidth)   # The file in which to store the output MCMC chain
    elif (dataflag == 1):          
        datafile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.lpow_%d_0p03-0p25_ave' % binwidth)                        # The data file
        covfile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.lpow_%d_0p03-0p25_cov' % binwidth)       # The covariance matrix 
        winfile = str('./files/test_files/taipanmock_year1_mock_rand_cullan.lwin')                         #    Power spectrum (for the integral constraint)
        winmatfile = str('./files/test_files/taipanmock_year1_mock_rand_cullan.winfit_%d' % binwidth)      #    Convolution matrix (for convolving the model)
        chainfile = str('./files/test_files/BAOfits/BAO_MockAverage_FullShape_taipan_year1_v1_lpow_0p03-0p25_%d' % binwidth)   # The file in which to store the output MCMC chain
        datafile_recon = str('./files/test_files/mock_average/Mock_taipan_year1_v1.lpow_%d_0p03-0p25_ave_recon' % binwidth)                        # The data file
        covfile_recon = str('./files/test_files/mock_average/Mock_taipan_year1_v1.lpow_%d_0p03-0p25_cov_recon' % binwidth)       # The covariance matrix 
        chainfile_recon = str('./files/test_files/BAOfits/BAO_MockAverage_FullShape_taipan_year1_v1_lpow_0p03-0p25_%d_recon' % binwidth)   # The file in which to store the output MCMC chain
    elif (dataflag == 2):  
        datafile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.rp_1_0p5_%d_0p02-0p30_ave' % binwidth)                        # The data file
        covfile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.rp_1_0p5_%d_0p02-0p30_cov' % binwidth)       # The covariance matrix 
        winfile = str('./files/test_files/taipanmock_year1_mock_rand_cullan.lwin')                         #    Power spectrum (for the integral constraint)
        winmatfile = str('./files/test_files/taipanmock_year1_mock_rand_cullan.winfit_%d' % binwidth)      #    Convolution matrix (for convolving the model)
        chainfile = str('./files/test_files/BAOfits/BAO_MockAverage_BAOExtractor_taipan_year1_v1_rp_1_0p5_0p02-0p30_%d' % binwidth)   # The file in which to store the output MCMC chain
        datafile_recon = str('./files/test_files/mock_average/Mock_taipan_year1_v1.rp_1_0p5_%d_0p02-0p30_ave_recon' % binwidth)                        # The data file
        covfile_recon = str('./files/test_files/mock_average/Mock_taipan_year1_v1.rp_1_0p5_%d_0p02-0p30_cov_recon' % binwidth)       # The covariance matrix 
        chainfile_recon = str('./files/test_files/BAOfits/BAO_MockAverage_BAOExtractor_taipan_year1_v1_rp_1_0p5_0p02-0p30_%d_recon' % binwidth)   # The file in which to store the output MCMC chain
    else:
        print "dataflag value not supported, ", dataflag
        exit()

    if (dataflag):
        xmin = 0.03
        xmax = 0.25
    else:
        xmin = 30.0
        xmax = 200.0
    plot_mockaverage(dataflag, matterfile, datafile, datafile_recon, covfile, covfile_recon, winfile, winmatfile, xmin, xmax, chainfile, chainfile_recon)



