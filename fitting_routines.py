# Classes associated with fitting over some data using a model. There are three classes:
#   "List", which performs a loop over a number of fixed alpha values and for each one finds the best-fit model
#   "Optimizer", which just fits all the free parameters of the model and returns the best-fitting model
#   "MCMC_emcee", which uses the emcee package to perform an MCMC over all free parameters. 
# For all of these we can use the normal Gaussian likelihood, apply the Hartlap 2007 correction, or we can use the modified 
# likelihood from Sellentin & Heavens 2016, which marginalises over the fact we evaluate the covariance matrix with a fixed number of realisations

import sys
import math
import numpy as np
from models import *
from scipy import optimize, special
from plotting_routines import Plotter
import emcee
import itertools

# A class for fitting the data using a list of fixed alpha values. For each fixed value of alpha we find fit for the other free parameters
# and output the best-fitting value for the chi-squared, likelihood and posterior with and without the BAO feature.
class List(object):

    def __init__(self, data, model, alphamin, alphamax, nalpha, outputfile=None, liketype="SH2016", do_plot=0, startfromprior=True, optimtype="Nelder-Mead", verbose=False):

        self.verbose = verbose
        self.data = data
        self.model = model
        self.alphamax = alphamax
        self.alphamin = alphamin
        self.nalpha = nalpha
        self.liketype = liketype
        self.chi_squared = 0.0
        self.likelihood = 0.0
        self.posterior = 0.0
        self.do_plot = do_plot
        self.startfromprior = startfromprior
        self.optimtype = optimtype

        if (self.do_plot):
            self.plotter = Plotter(fignum=1, data=self.data, interactive=True)

        if outputfile is None:
            self.outputfile = None
            self.outputstream = sys.stdout
            if (self.verbose):
                print "Outputting to stdout"
        else:
            self.outputfile = outputfile
            self.outputstream = None
            if (self.verbose):
                print "Outputting to file: ", self.outputfile

        checkfitterinputs(self) 

        if isinstance(self.model, LinearPoint):
            print "List-based fitting of alpha is not supported for LinearPoint fitting, please use class Optimizer or MCMC_emcee instead"
            exit()

    def fit_data(self, optimtype=None, startfromprior=None):

        if (startfromprior is not None):
            self.startfromprior = startfromprior

        if (optimtype is not None):
            self.optimtype = optimtype

        if (self.outputstream is None):
            self.outputstream = open(self.outputfile, 'w')

        if (self.verbose):
            print "Evaluating model for ",self.nalpha, "fixed alpha values between ", self.alphamin, self.alphamax, "inclusive: "

        # Do anything we need to do to prepare the model. I.e., for the Polynomial method this involves fitting the data without BAO to compute the
        # normalisation such that B is always close to 1, then modifying the B prior (which is rather recursive!). This function belongs to all
        # classes of model, but only does something if the model is set up to perform any preparatory steps (which is controlled by the self.model.prepare_model_flag)
        # parameter.
        if (self.model.prepare_model_flag):
            prepare_model(self.data, self.model, liketype=self.liketype, do_plot=self.do_plot, startfromprior=self.startfromprior, optimtype=self.optimtype, verbose=self.verbose)

        self.outputstream.write("           alpha      log(chi_squared)      log(likelihood)      log(posterior)      log(chi_squared) (no BAO)      log(likelihood) (no BAO)      log(posterior) (no BAO)\n")
        for i in range(self.nalpha):
            plt_handle = []
            if (self.nalpha == 1):
                alpha = (self.alphamax + self.alphamax)/2.0
            else:
                alpha = i*(self.alphamax-self.alphamin)/(self.nalpha-1) + self.alphamin
            self.model.params["alpha"][0] = alpha

            # Fit the data with the model without BAO
            self.model.BAO = False
            result = self.run_optimizer()
            chi_squarednoBAO = self.chi_squared
            likelihoodnoBAO = self.likelihood
            posteriornoBAO = self.posterior
            free_params = self.model.get_free_params()
            smooth_params = np.empty(len(free_params))
            for counter, i in enumerate(free_params):
                smooth_params[counter] = self.model.params[i][0]

            # Do the fit with BAO
            self.model.BAO = True
            result = self.run_optimizer()
            chi_squared = self.chi_squared
            likelihood = self.likelihood
            posterior = self.posterior

            if (self.do_plot):
                self.plotter.display_plot(plt_array=[self.plotter.add_model_to_plot_fixedsmooth(self.data, self.model, smooth_params)])

            self.outputstream.write("           %12.6g  %12.6g  %12.6g  %12.6g  %12.6g  %12.6g  %12.6g\n" % (alpha, chi_squared, likelihood, posterior, chi_squarednoBAO, likelihoodnoBAO, posteriornoBAO))

            if (self.outputfile is not None):
                self.outputstream.close()

        return alpha, chi_squared, likelihood, posterior, chi_squarednoBAO, likelihoodnoBAO, posteriornoBAO

    # A routine to optimize the model given some data
    def run_optimizer(self, optimtype=None, startfromprior=None):

        if (startfromprior is None):
            startfromprior = self.startfromprior

        if (optimtype is None):
            optimtype = self.optimtype

        # Create a simplex for Nelder-Mead minimization with a list of free parameter names. 
        # We start the simplex at wide values drawn from the prior for each parameters
        if (startfromprior):
            free_params = self.model.get_free_params()
            nfree_params = len(free_params)
            for i in free_params:
                if (i == "alpha"):
                    nfree_params -= 1
            x0 = np.empty((nfree_params+1,nfree_params))
            begin = np.empty(nfree_params)
            counter = 0
            for i in free_params:
                if (i == "alpha"):
                    continue
                if (self.model.params[i][1] == "Linear"):
                    lowval = self.model.params[i][2]
                    hival = self.model.params[i][3]
                elif (self.model.params[i][1] == "Log"): 
                    lowval = math.exp(self.model.params[i][2])
                    hival = math.exp(self.model.params[i][3])
                elif (self.model.params[i][1] == "Gaussian"): 
                    lowval = self.model.params[i][2]-4.0*self.model.params[i][3]
                    hival = self.model.params[i][2]+4.0*self.model.params[i][3]
                elif (self.model.params[i][1] == "LogGaussian"): 
                    lowval = math.exp(self.model.params[i][2]-4.0*self.model.params[i][3])
                    hival = math.exp(self.model.params[i][2]+4.0*self.model.params[i][3])
                else:
                    print "Prior type ", self.model.params[i][1], "for ", i, "not supported, must be one of: Linear, Log, Gaussian or LogGaussian"
                    exit()
                begin[counter] = self.model.params[i][0]
                x0[0:,counter] = np.repeat(lowval,nfree_params+1)
                x0[counter+1,counter] = hival       
                counter += 1

            # Run the Nelder-Mead minimization. For the correlation function we might be able to compute the integration over the power spectrum outside
            # this loop and pass this in as an argument to the minimizer to greatly speed things up
            nll = lambda *args: -lnpost(*args)
            result = optimize.minimize(nll, begin, method=optimtype, tol=1.0e-7, args=self, options={'maxiter':50000, 'initial_simplex':x0})

        else:

            free_params = self.model.get_free_params()
            nfree_params = len(free_params)
            for i in free_params:
                if (i == "alpha"):
                    nfree_params -= 1
            begin = np.empty(nfree_params)
            counter = 0
            for i in free_params:
                if (i == "alpha"):
                    continue
                begin[counter] = self.model.params[i][0] 
                counter += 1
  
            nll = lambda *args: -lnpost(*args)
            result = optimize.minimize(nll, begin, method=optimtype, tol=1.0e-7, args=self, options={'maxiter':50000})

        return result

# A class for fitting the data using a simple optimization routine. For all free parameters, we maximise the 
# posterior probablility of the parameters given the data and output the best-fitting values.
class Optimizer(object):

    def __init__(self, data, model, outputfile=None, liketype="SH2016", do_plot=0, startfromprior=True, optimtype="Nelder-Mead", verbose=False):

        self.verbose = verbose
        self.data = data
        self.model = model
        self.liketype = liketype
        self.chi_squared = 0.0
        self.likelihood = 0.0
        self.posterior = 0.0
        self.do_plot = do_plot
        self.startfromprior = startfromprior
        self.optimtype = optimtype

        if outputfile is None:
            self.outputfile = None
            self.outputstream = sys.stdout
            if (self.verbose):
                print "Outputting to stdout"
        else:
            self.outputfile = outputfile
            self.outputstream = None
            if (self.verbose):
                print "Outputting to file: ", self.outputfile

        checkfitterinputs(self) 

        if (isinstance(self.model, LinearPoint) and (self.startfromprior)):
            print "Warning starting optimization from a wide prior range is not recommended for LinearPoint model as it converges poorly."
            print "Convergence is much better if you set self.startfromprior == False and the initial values of the parameters to zeroes"

    def fit_data(self, optimtype=None, startfromprior=None):

        if (startfromprior is not None):
            self.startfromprior = startfromprior

        if (optimtype is not None):
            self.optimtype = optimtype

        if (self.outputstream is None):
            self.outputstream = open(self.outputfile, 'w')

        if (self.verbose):
            print "Optimizing to find the best-fitting model"

        # Do anything we need to do to prepare the model. I.e., for the Polynomial method this involves fitting the data without BAO to compute the
        # normalisation such that B is always close to 1, then modifying the B prior (which is rather recursive!). This function belongs to all
        # classes of model, but only does something if the model is set up to perform any preparatory steps (which is controlled by the self.model.prepare_model_flag)
        # parameter.
        if (self.model.prepare_model_flag):
            prepare_model(self.data, self.model, liketype=self.liketype, do_plot=self.do_plot, startfromprior=self.startfromprior, optimtype=self.optimtype, verbose=self.verbose)

        # Do the fit with BAO
        result = self.run_optimizer()
        self.outputstream.write("Best fit model: Parameters,       log(chi_squared),      log(likelihood),      log(posterior)\n")
        free_params = self.model.get_all_params()
        self.outputstream.write("%s" % ', '.join([str(e) for e in [[self.model.params[i][0] for i in free_params], self.chi_squared, self.likelihood, self.posterior]]))
        self.outputstream.write("\n")

        if (self.outputfile is not None):
            self.outputstream.close()

        if (self.do_plot):
            Plotter(data=data, model=model)

        return

    # A routine to optimize the model given some data
    def run_optimizer(self, optimtype=None, startfromprior=None):

        if (startfromprior is None):
            startfromprior = self.startfromprior

        if (optimtype is None):
            optimtype = self.optimtype

        if (isinstance(self.model, LinearPoint) and (startfromprior)):
            print "WARNING: starting optimization from a wide prior range is not recommended for LinearPoint model as the large dynamic range of"
            print "the polynomial terms means it converges very poorly and will almost certainly give nonsense. Convergence is much better if"
            print "you set self.startfromprior == False and use the default starting parameters for the polynomial terms"

        # Create a simplex for Nelder-Mead minimization with a list of free parameter names. 
        # We start the simplex at wide values drawn from the prior for each parameters
        if (startfromprior):
            free_params = self.model.get_free_params()
            x0 = np.empty((len(free_params)+1,len(free_params)))
            begin = np.empty(len(free_params))
            counter = 0
            for i in free_params:
                if (self.model.params[i][1] == "Linear"):
                    lowval = self.model.params[i][2]
                    hival = self.model.params[i][3]
                elif (self.model.params[i][1] == "Log"): 
                    lowval = math.exp(self.model.params[i][2])
                    hival = math.exp(self.model.params[i][3])
                elif (self.model.params[i][1] == "Gaussian"): 
                    lowval = self.model.params[i][2]-4.0*self.model.params[i][3]
                    hival = self.model.params[i][2]+4.0*self.model.params[i][3]
                elif (self.model.params[i][1] == "LogGaussian"): 
                    lowval = math.exp(self.model.params[i][2]-4.0*self.model.params[i][3])
                    hival = math.exp(self.model.params[i][2]+4.0*self.model.params[i][3])
                else:
                    print "Prior type ", self.model.params[i][1], "for ", i, "not supported, must be one of: Linear, Log, Gaussian or LogGaussian"
                    exit()
                begin[counter] = self.model.params[i][0]
                x0[0:,counter] = np.repeat(lowval,len(free_params)+1)
                x0[counter+1,counter] = hival       
                counter += 1

            # Run the Nelder-Mead minimization
            nll = lambda *args: -lnpost(*args)
            result = optimize.minimize(nll, begin, method=optimtype, tol=1.0e-8, args=self, options={'maxiter':100000, 'initial_simplex':x0, 'xatol':1.0e-8, 'fatol':1.0e-8})

        else:

            free_params = self.model.get_free_params()
            begin = np.empty(len(free_params))
            counter = 0
            for i in free_params:
                begin[counter] = self.model.params[i][0] 
                counter += 1

            nll = lambda *args: -lnpost(*args)
            result = optimize.minimize(nll, begin, method=optimtype, tol=1.0e-8, args=self, options={'maxiter':100000, 'xatol':1.0e-8, 'fatol':1.0e-8})

        return result

# A class for fitting the data using a simple optimization routine. For all free parameters, we maximise the 
# posterior probablility of the parameters given the data and output the best-fitting values.
class MCMC_emcee(object):

    def __init__(self, data, model, nwalkers=None, niterations=None, startfrombestfit=False, outputfile=None, liketype="SH2016", do_plot=0, verbose=False):

        self.verbose = verbose
        self.data = data
        self.model = model
        self.liketype = liketype
        self.chi_squared = 0.0
        self.likelihood = 0.0
        self.posterior = 0.0
        self.do_plot = do_plot
        self.nwalkers = nwalkers
        self.niterations = niterations
        self.startfrombestfit = startfrombestfit

        if (self.do_plot):
            self.plotter = Plotter(fignum=1, data=self.data, interactive=True)

        if outputfile is None:
            self.outputfile = None
            self.outputstream = sys.stdout
            if (self.verbose):
                print "Outputting to stdout"
        else:
            self.outputfile = outputfile
            self.outputstream = None
            if (self.verbose):
                print "Outputting to file: ", self.outputfile

        checkfitterinputs(self)

    def fit_data(self, nwalkers=None, niterations=None, startfrombestfit=None):

        if (self.verbose):
            print "MCMCing over the data using the given model"

        if (startfrombestfit is not None):
            self.startfrombestfit = startfrombestfit

        if (niterations is not None):
            self.niterations = nwalkers
        else:
            if (self.niterations is None):
                self.niterations = 2000

        if (self.outputstream is None):
            self.outputstream = open(self.outputfile, 'w')


        # Do anything we need to do to prepare the model. I.e., for the Polynomial method this involves fitting the data without BAO to compute the
        # normalisation such that B is always close to 1, then modifying the B prior (which is rather recursive!). This function belongs to all
        # classes of model, but only does something if the model is set up to perform any preparatory steps (which is controlled by the self.model.prepare_model_flag)
        # parameter.
        if (self.model.prepare_model_flag):
            prepare_model(self.data, self.model, liketype=self.liketype, do_plot=self.do_plot, verbose=self.verbose)

        # Set up the walkers
        free_params = self.model.get_free_params()
        nparams = len(free_params)
        if (nwalkers is not None):
            self.nwalkers = nwalkers
        else:
            if (self.nwalkers is None):
                self.nwalkers = 8*nparams

        # Find the starting points
        if (self.startfrombestfit):
            if (self.verbose):
                print "Starting from narrow region around best-fit:"
            result = Optimizer(self.data, self.model, liketype=self.liketype, do_plot=self.do_plot, verbose=self.verbose).run_optimizer()
            if (self.verbose):
                print "Best fit model with BAO feature: Parameters,       log(chi_squared),      log(likelihood),      log(posterior)"
                free_params = self.model.get_all_params()
                print [self.model.params[i][0] for i in free_params], self.chi_squared, self.likelihood, self.posterior
            begin = np.array([list(itertools.chain.from_iterable([[(0.001*(np.random.rand()-0.5)+1.0)*result['x'][j] for j in range(nparams)]])) for i in range(self.nwalkers)])
        else:
            if (self.verbose):
                print "Drawing random start points from prior:"
            begin = []
            for i in free_params:
                if (self.model.params[i][1] == "Linear"):
                    begin.append(np.random.uniform(self.model.params[i][2], self.model.params[i][3], self.nwalkers))
                elif (self.model.params[i][1] == "Log"): 
                    begin.append(np.exp(np.random.uniform(self.model.params[i][2], self.model.params[i][3], self.nwalkers)))
                elif (self.model.params[i][1] == "Gaussian"): 
                    begin.append(np.random.normal(self.model.params[i][2], self.model.params[i][3], self.nwalkers))
                elif (self.model.params[i][1] == "LogGaussian"): 
                    begin.append(np.exp(np.random.normal(np.log(self.model.params[i][2]), self.model.params[i][3], self.nwalkers)))
                else:
                    print "Prior type ", self.model.params[i][1], "for ", i, "not supported, must be one of: Linear, Log, Gaussian or LogGaussian"
                    exit()
            begin = np.array(begin).T

        # RELEASE THE CHAIN!!!
        sampler = emcee.EnsembleSampler(self.nwalkers, nparams, self.lnpost_emcee)
        pos = sampler.run_mcmc(begin, 1)[0]
        sampler.reset()

        # Run and print out the chain
        for counter, result in enumerate(sampler.sample(pos, iterations=self.niterations, storechain=False)):
            if (self.do_plot):
                if (counter % 20 == 0):
                    self.plotter.display_plot(plt_array=[self.plotter.add_model_to_plot(self.data, self.model)])
            if ((counter % 500) == 0):
                print "Niter: ", counter, (", (Mean acceptance fraction: {0:.3f})".format(np.mean(sampler.acceptance_fraction)))
            for k in range(self.nwalkers):
                self.outputstream.write("%4d  " % k)
                for m in range(len(result[3][k])):
                    self.outputstream.write("%g  " % result[3][k][m])
                self.outputstream.write("%g\n" % result[1][k])

        if (self.outputfile is not None):
            self.outputstream.close()

        return

    # Update the parameter values and calculate the posterior probability, this is used for emcee 
    # as it allows us to return additional parameters for each walker as a 'blob'
    def lnpost_emcee(self, params):

        posterior = lnpost(params, self)
        free_params = self.model.get_all_params()
        paramvals = [self.model.params[i][0] for i in free_params]
        return posterior, paramvals + [self.chi_squared, self.likelihood]

# ******************************** #
# Global Functions outside classes #
# ******************************** #

# A routine to check that the fitter inputs are valif whichever class we have used
def checkfitterinputs(fitter):

    if ((fitter.liketype != "LeastSquares") and (fitter.liketype != "Gaussian") and (fitter.liketype != "Hartlap2007") and (fitter.liketype != "SH2016")):
        print "Value for likelihood type unsupported, must be either LeastSquares, Gaussian, Hartap2007 or SH2016. Current value: ", liketype 
        exit()    

    if ((fitter.liketype == "Gaussian") or (fitter.liketype == "Hartlap2007") or (fitter.liketype == "SH2016")):
        if (fitter.data.cov_inv is None):
            print "No inverse covariance matrix found for data. Please use read_cov to read in a covariance matrix (which will be inverted automatically),"
            print "call data.compute_cov_inv() with data.cov != None, or set data.cov_det and data.cov_inv manually"
            exit()    

    if ((fitter.liketype == "Hartlap2007") or (fitter.liketype == "SH2016")):
        if (fitter.data.nmocks is None):
            print "For Hartlap2007 or SH2016 likelihood we require the number of mocks used to estimate the covariance, please change value associated with InputData class. Current value: ", fitter.data.nmocks
            exit()    

    if (fitter.data.__class__.__name__ != fitter.model.datatype):
        print "Datatype (denoting CorrelationFunction, PowerSpectrum, BAOExtract) not consistent between data (", fitter.data.__class__.__name__, ") and model (", fitter.model.datatype, ")"
        exit()

    return

# Do any preliminary preparation of the model that we need to do. This takes as input a data class and a
# model class and only actually does anything if the model class is Polynomial. In this case we generate an Optimizer class
# fitter to find the normalisation such that B is close to one, then put a loose Gaussian prior on log(B). This is very recursive
# but can be called from any other fitting class, or just generally
def prepare_model(data, model, liketype="SH2016", do_plot=0, startfromprior=True, optimtype="Nelder-Mead", verbose=False):

    if isinstance(model, Polynomial):

        # Set the normalisation of the model, such that the free parameter B is close to one. For the power spectrum we generate the 
        # fit without the BAO feature and use the best fit B as the normalisation. For the correlation function we compute the model at the cloest point to
        # 50 Mpc/h and use that to compute the normalisation
        oldBAO = model.BAO
        model.BAO = False
        if (model.datatype == "PowerSpectrum"):
            free_params = model.get_free_params()
            oldparams = np.empty(len(free_params))
            for counter, i in enumerate(free_params):
                oldparams[counter] = model.params[i][0]
            result = Optimizer(data, model, liketype=liketype, do_plot=do_plot, startfromprior=startfromprior, optimtype=optimtype, verbose=verbose).run_optimizer()
            model.norm = result["x"][0]
            for counter, i in enumerate(free_params):
                model.params[i][0] = oldparams[counter]
        else:
            index = np.where(data.x >= 50.0)[0]
            yval = model.compute_model(x=data.x[index[0]])
            model.norm = data.y[index[0]]/yval

        # Update the B prior
        model.BAO = oldBAO
        set_prior(model, "B", ["LogGaussian",1.0, 0.4])

    return 

# Update the parameter values and calculate the posterior probability
def lnpost(params, fitter):

    # Update parameters. For List based fitting, alpha has already been updated
    if isinstance(fitter, List):
        free_params = fitter.model.get_free_params()
        counter = 0
        for i in free_params:
            if (i == "alpha"):
                continue
            fitter.model.params[i][0] = params[counter]
            counter += 1
    else:
        free_params = fitter.model.get_free_params()
        counter = 0
        for i in free_params:
            fitter.model.params[i][0] = params[counter]
            counter += 1

    # Compute the prior
    prior = compute_prior(fitter.model)
    if not np.isfinite(prior):
        return -np.inf

    # Set the model using the new parameters
    if ((fitter.model.datatype == "PowerSpectrum") or (fitter.model.datatype == "BAOExtract")):
        yvals = fitter.model.compute_model(x=fitter.data.kwinmatin)
        p0 = np.sum(fitter.data.winmat[0,0:]*yvals)
        pkmod = np.zeros(len(fitter.data.winmat[0:])-1)
        for j in range(len(fitter.data.winmat[0:])-1):
            pkmod[j] = np.sum(fitter.data.winmat[j+1][0:]*yvals) - p0*fitter.data.pkwin[j]
        if (fitter.model.datatype == "PowerSpectrum"):
            fitter.model.x = fitter.data.kwinmatout[fitter.data.kwinmatoutindex]
            fitter.model.y = pkmod[fitter.data.kwinmatoutindex]
        else:
            fitter.model.x = fitter.data.kwinmatout[fitter.data.kwinmatoutindex]
            fitter.model.y = fitter.model.extract_BAO(fitter.data.kwinmatout, pkmod)[fitter.data.kwinmatoutindex]
    elif (fitter.model.datatype == "CorrelationFunction"):
        set_model(fitter.model,x=fitter.data.x)
    else:
        print "Datatype ", fitter.model.datatype, " not supported, must be either 'PowerSpectrum', 'CorrelationFunction' or 'BAOExtract'"
        exit()

    like = lnlike(len(params), fitter)
    fitter.posterior = prior+like

    return fitter.posterior

# Calculate the likelihood of the data given the model. This is the same for all classes
def lnlike(nparams, fitter):

    if (fitter.liketype == "LeastSquares"):
        fitter.likelihood = -np.sum((fitter.data.y-fitter.model.y)**2)
    else:
        fitter.chi_squared = 0.0
        for i in range(len(fitter.data.x)):
            fitter.chi_squared += (fitter.data.y[i]-fitter.model.y[i])*np.sum(fitter.data.cov_inv[i,0:]*(fitter.data.y-fitter.model.y))

        if (fitter.liketype == "Gaussian"):
            fitter.likelihood = -0.5*len(fitter.data.x)*math.log(2.0*math.pi) - 0.5*fitter.data.cov_det - 0.5*fitter.chi_squared
        elif (fitter.liketype == "Hartlap2007"):
            norm = (fitter.data.nmocks - nparams - 2.0)/(fitter.data.nmocks - 1.0)
            fitter.likelihood = -0.5*len(fitter.data.x)*math.log(2.0*math.pi) - 0.5*fitter.data.cov_det - 0.5*fitter.chi_squared*norm
        elif (fitter.liketype == "SH2016"):
            norm = special.gammaln(0.5*fitter.data.nmocks) - 0.5*nparams*math.log(math.pi*(fitter.data.nmocks)-1.0) - special.gammaln(0.5*(fitter.data.nmocks-nparams))
            fitter.likelihood = norm - 0.5*fitter.data.cov_det - 0.5*fitter.data.nmocks*np.log(1.0 + fitter.chi_squared/(fitter.data.nmocks-1.0))

    return fitter.likelihood

# This routine takes a model class and chains stored in outputfile and makes it suitable for passing to ChainConsumer by returning 
# the parameter names, the number of walkers, the chain organised so that each walker is contiguous, and the log posterior.
def prepareforChainConsumer(model, outputfile, burnin=0):

    free_params = model.get_latex_params()

    walkers=[]
    samples=[]
    loglike = []        
    outputstream = open(outputfile, 'r')
    for line in outputstream:
        ln=line.split()
        samples.append(map(float, ln[1:len(free_params)+1]))
        walkers.append(int(ln[0]))
        loglike.append(float(ln[-2]))
    outputstream.close()

    walkers = np.array(walkers)
    nwalkers = max(walkers)+1
    samples = np.array(samples).reshape(len(samples),len(free_params))
    loglike = np.array(loglike)

    bestid = np.argmax(loglike)
    max_params = samples[bestid,0:]

    nburnin = len(samples)-burnin*nwalkers
    newsamples = []
    newloglike = []
    for i in range(nwalkers):
        index = np.where(walkers == i)[0]
        if (len(index) == 0):
            continue
        x = [j for j in xrange(len(index))]
        index2 = np.where(np.asarray(x) >= burnin)[0]
        for k in range(len(index2+1)):
            newsamples.append(samples[index[index2[k]],0:])
            newloglike.append(loglike[index[index2[k]]])
    newsamples = np.array(newsamples).reshape(nburnin,len(free_params))
    newloglike = np.array(newloglike)

    return free_params, max_params, loglike[bestid], newsamples, newloglike

