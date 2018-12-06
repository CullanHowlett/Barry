import sys
sys.path.append("./src/")
import numpy as np
from models import *
from read_data import *
from powerspectrum import *
from fitting_routines import *
from plotting_routines import *
from chainconsumer import ChainConsumer

# Read in the chains for some mocks (pre- and post-recon), and compute some summary statistics (the mean alpha, the average error and the sample variance)
if __name__ == "__main__":

    dataflag = int(sys.argv[1])    # Whether to analyse the correlation function (0), power spectrum (1), or BAO extract (2)
    binwidth = int(sys.argv[2])
    ID = int(sys.argv[3])
    firstmock = int(sys.argv[4])
    nmocks = int(sys.argv[5])

    xmin = 0.03
    xmax = 0.25
    power = Hinton2017CAMB(redshift=0.11, mnu=0.0)

    alphamean = np.empty((nmocks,2))
    alphaerrlo = np.empty((nmocks,2))
    alphaerrhi = np.empty((nmocks,2))
    alphasig = np.empty((nmocks,2))
    for mock in range(nmocks):

        mocknum = ID*nmocks+firstmock+mock

        # Loop ove pre or post-recon data and models
        for i in range(2):

            if ((dataflag == 1) or (dataflag == 2)):
                winfile = str('./files/test_files/taipanmock_year1_mock_rand_cullan.lwin')                         #    Power spectrum (for the integral constraint)
                winmatfile = str('./files/test_files/taipanmock_year1_mock_rand_cullan.winfit_%d' % binwidth)      #    Convolution matrix (for convolving the model)

            if (i == 0):
                if (dataflag == 0): 
                    datafile = str('./files/test_files/mock_individual/Mock_taipan_year1_v1_R%d.xi_%d' % (mocknum, binwidth))
                    covfile =  str('./files/test_files/mock_average/Mock_taipan_year1_v1.xi_%d_cov' % binwidth)       # The covariance matrix
                    chainfile = str('./files/test_files/BAOfits/BAO_Mock_taipan_year1_v1_R%d_xi_%d' % (mocknum, binwidth))   # The file in which to store the output MCMC chain
                    model = Polynomial("CorrelationFunction", power, free_sigma_nl=False, prepare_model_flag=True)
                    model.sigma_nl = 11.7
                elif (dataflag == 1):          
                    datafile = str('/fred/oz074/clustering/results/Mock_v1/prerecon/Mock_taipan_year1_v1_R%d.lpow_blake' % (mocknum))                        # The data file
                    covfile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.lpow_%d_0p03-0p25_cov' % binwidth)       # The covariance matrix 
                    chainfile = str('/fred/oz074/clustering/results/BAOfits/Mock_v1/BAO_Mock_FullShape_SigmaNLprior_taipan_year1_v1_R%d_lpow_0p03-0p25_%d' % (mocknum, binwidth))   # The file in which to store the output MCMC chain
                    model = FullShape("PowerSpectrum", power, free_sigma_nl=True, nonlinearterms="./files/compute_pt_integrals_output.dat", verbose=True)
                    set_prior(model, "sigma_nl", ["Gaussian", 12.5, 4.0])
                elif (dataflag == 2):          
                    datafile = str('./files/test_files/mock_individual/Mock_taipan_year1_v1_R%d.lpow_blake' % (mocknum))                        # The data file
                    covfile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.rp_1_0p5_%d_0p02-0p30_cov' % binwidth)       # The covariance matrix 
                    chainfile = str('./files/test_files/BAOfits/BAO_Mock_BAOExtractor_FreeSigmaNL_taipan_year1_v1_R%d_rp_1_0p5_0p02-0p30_%d' % (mocknum, binwidth))   # The file in which to store the output MCMC chain
                    model = BAOExtractor(power, free_sigma_nl=True, nonlinearterms="./files/compute_pt_integrals_output.dat", verbose=True)
                else:
                    print "dataflag value not supported, ", dataflag
                    exit()
            elif (i == 1):
                if (dataflag == 0): 
                    datafile = str('./files/test_files/mock_individual/Mock_taipan_year1_v1_R%d.xi_%d' % (mocknum, binwidth))
                    covfile =  str('./files/test_files/mock_average/Mock_taipan_year1_v1.xi_%d_cov' % binwidth)       # The covariance matrix
                    chainfile = str('./files/test_files/BAOfits/BAO_Mock_taipan_year1_v1_R%d_xi_%d' % (mocknum, binwidth))   # The file in which to store the output MCMC chain
                    model = Polynomial("CorrelationFunction", power, free_sigma_nl=False, prepare_model_flag=True)
                    model.sigma_nl = 11.7
                elif (dataflag == 1):          
                    datafile = str('/fred/oz074/clustering/results/Mock_v1/postrecon/Mock_taipan_year1_v1_R%d.lpow_blake_recon' % (mocknum))                        # The data file
                    covfile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.lpow_%d_0p03-0p25_cov_recon' % binwidth)       # The covariance matrix 
                    chainfile = str('/fred/oz074/clustering/results/BAOfits/Mock_v1/BAO_Mock_FullShape_SigmaNLprior_taipan_year1_v1_R%d_lpow_0p03-0p25_%d_recon' % (mocknum, binwidth))   # The file in which to store the output MCMC chain
                    model = FullShape("PowerSpectrum", power, free_sigma_nl=True, nonlinearterms="./files/compute_pt_integrals_output.dat", remove_kaiser=True, verbose=True)
                    set_prior(model, "sigma_nl", ["Gaussian", 12.5, 4.0])
                elif (dataflag == 2):          
                    datafile = str('./files/test_files/mock_individual/Mock_taipan_year1_v1_R%d.lpow_blake' % (mocknum))                        # The data file
                    covfile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.rp_1_0p5_%d_0p02-0p30_cov' % binwidth)       # The covariance matrix 
                    chainfile = str('./files/test_files/BAOfits/BAO_Mock_BAOExtractor_FreeSigmaNL_taipan_year1_v1_R%d_rp_1_0p5_0p02-0p30_%d' % (mocknum, binwidth))   # The file in which to store the output MCMC chain
                    model = BAOExtractor(power, free_sigma_nl=True, nonlinearterms="./files/compute_pt_integrals_output.dat", verbose=True)
                else:
                    print "dataflag value not supported, ", dataflag
                    exit()

            # Set up the data class
            if (dataflag == 0):
                data = CorrelationFunction(nmocks=1000).read_data(datafile=datafile, covfile=covfile, xmin=xmin, xmax=xmax)
            elif (dataflag == 1):
                data = PowerSpectrum(nmocks=1000, verbose=True).read_data(datafile=datafile, covfile=covfile, xmin=xmin, xmax=xmax, nconcat=binwidth, winfile=winfile, winmatfile=winmatfile)
            elif (dataflag == 2):
                data = BAOExtract(nmocks=1000, verbose=True).read_data(datafile=datafile, covfile=covfile, xmin=xmin, xmax=xmax, nconcat=binwidth, winfile=winfile, winmatfile=winmatfile).extract_BAO(power.r_s)
            else:
                print "dataflag value not supported, ", dataflag
                exit()

            # Now read in the chain
            params, max_params, max_loglike, samples, loglike = prepareforChainConsumer(model, outputfile=chainfile, burnin=1000)

            # Find the marginalised maximum likelihood value of alpha and the 68% confidence interval about this point (bounded by the prior on alpha)
            c = ChainConsumer().add_chain(samples, parameters=params, posterior=loglike, num_eff_data_points=len(data.x), num_free_params=len(params))
            alphavals = c.analysis.get_summary()['$\\alpha$']
            print alphavals
            if (alphavals[0] == None):
                alphavals[0] = model.params["alpha"][2]
            if (alphavals[1] == None):
                alphavals[1] = 1.0
            if (alphavals[2] == None):
                alphavals[2] = model.params["alpha"][3]

            alphamean[mock,i] = alphavals[1] 
            alphaerrlo[mock,i] = alphavals[1]-alphavals[0]
            alphaerrhi[mock,i] = alphavals[2]-alphavals[1] 

            # Now find the best-fit smooth model at the best-fit value of alpha to determine the detection significance. We can do this 
            # using the List based fitter class with a single value of alpha. For the LinearPoint fitting, we use smooth model from the Polynomial class
            free_params = model.get_all_params()
            for counter, j in enumerate(free_params):
                model.params[j][0] = max_params[counter]
            print model.params
            fitter = List(data, model, alphavals[1], alphavals[1], 1, liketype="SH2016", do_plot=0, startfromprior=False, optimtype="Powell")
            alpha, chi_squared, likelihood, posterior, chi_squarednoBAO, likelihoodnoBAO, posteriornoBAO = fitter.fit_data()

            alphasig[mock,i] = chi_squarednoBAO-chi_squared
            if (alphasig[mock,i] > 0.0):
                alphasig[mock,i] = np.sqrt(alphasig[mock,i])
            else:
                alphasig[mock,i] = -np.sqrt(np.fabs(alphasig[mock,i]))

    strout = str("/fred/oz074/clustering/results/BAOfits/Mock_v1/BAO_Mock_FullShape_SigmaNLprior_taipan_year1_v1_lpow_0p03-0p25_%d_summary_%d" % (binwidth, ID))
    outfile = open(strout, 'w')
    for mock in range(nmocks):
        mocknum = ID*nmocks+firstmock+mock
        outfile.write("%d  %lf  %lf  %lf  %lf  %lf  %lf  %lf  %lf\n" % (mocknum, alphamean[mock,0], alphaerrlo[mock,0], alphaerrhi[mock,0], alphasig[mock,0], alphamean[mock,1], alphaerrlo[mock,1], alphaerrhi[mock,1], alphasig[mock,1]))
    outfile.close()

