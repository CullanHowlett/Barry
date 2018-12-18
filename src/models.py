# A class associated with all the models for the BAO feature. There are four classes of model each of which produces a model
# based on some free parameters and a smooth power spectrum (except LinearPoint, which doesn't require this). Each class also
# has routines that will evaluate the prior for the current parameter values.

import math
import numpy as np
from scipy import integrate, interpolate, optimize
from hankel import SymmetricFourierTransform
from subprocess import check_call
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
import matplotlib.font_manager
from matplotlib import gridspec

# The Polynomial based model. This is the model used for many of isotropic BAO analyses from SDSS. It consists of a dewiggled power spectrum, modulated by
# the alpha parameter and multiplied by some bias value. A polynomial term is then added to the model to marginalise over the broadband shape. The fitting procedure
# is quite finely tuned, so there are a number of obscurities with this model, in particular related to whether or not the width of the BAO feature is fixed or
# varied, and how the normalisation is dealt with. The class is flexible enough to allow all these choices to be explored by chnaging the parameters and flags below
class Polynomial(object):

    def __init__(self, datatype, powerspectrum, x=None, xnarrow=None, alpha=1.0, sigma_nl=10.0, B=1.0, a1=0.0, a2=0.0, a3=0.0, a4=0.0, a5=0.0, 
                       free_sigma_nl=True, BAO=True, norm=1.0, prepare_model_flag=False, verbose=False):

        if ((datatype != "PowerSpectrum") and (datatype != "CorrelationFunction") and (datatype != "BAOExtract")):
            print "Datatype ", datatype, " not supported for FullShape class, must be either 'PowerSpectrum', 'CorrelationFunction' or 'BAOExtract'"
            exit()

        self.verbose = verbose
        self.datatype = datatype
        self.power = powerspectrum

        if (self.verbose):
            print "Setting up Polynomial model:  "

        # The x and y values for the model
        if (x is not None):
            if (hasattr(x,'__len__')):
                self.x = np.array(x)
            else:
                self.x = np.array([x])
        else:
            if ((self.datatype == "PowerSpectrum") or (self.datatype == "BAOExtract")):
                self.x = np.linspace(0.0, 0.3, 40)
            elif (self.datatype == "CorrelationFunction"):
                self.x = np.linspace(0.0, 200.0, 40)
            else:
                self.datatype_error()
        self.y = np.zeros(len(self.x))

        # The parameters of the model. Self.params contains a dictionary of all the free parameters for the model. Each dictionary element contains:
        #   the current value
        #   the prior type, one of "Linear", "Log", "Gaussian" or "LogGaussian"
        #   the values of the prior. These are lower/upper limits for flat priors, central value and standard deviation for (Log)Gaussian priors.
        self.params = {"alpha":    [alpha, "Linear", 0.7, 1.3],
                       "sigma_nl": [sigma_nl, "Linear", 2.0, 20.0],
                       "B":        [B, "Linear", 0.01, 12.0]}
        if ((self.datatype == "PowerSpectrum") or (self.datatype == "BAOExtract")):
            self.params["a1"] = [a1, "Linear", -50000.0, 50000.0]
            self.params["a2"] = [a2, "Linear", -50000.0, 50000.0]
            self.params["a3"] = [a3, "Linear", -50000.0, 50000.0]
            self.params["a4"] = [a4, "Linear",  -1000.0,  1000.0]
            self.params["a5"] = [a5, "Linear",    -10.0,    10.0]
        elif (self.datatype == "CorrelationFunction"):
            self.params["a1"] = [a1, "Linear", -1000.0, 1000.0]
            self.params["a2"] = [a2, "Linear",   -10.0,   10.0]
            self.params["a3"] = [a3, "Linear",    -2.0,    2.0]
        else:
            self.datatype_error()

        # Some choices required for the model, a normalisation, whether or not to include the BAO feature, fix or vary sigma_nl, the total number of free parameters
        # An important choice in the model is also whether or not to perform some actions just prior to fitting data. In the literature
        # it seems to be the case that the data is fit with a flat prior on B, then the model is normalised, and a wide prior put on B to keep it close to 1
        # The self.prepare_model_flag allows us to choose whether or not we do these actions during fitting.
        self.norm = norm
        self.BAO = BAO
        self.free_sigma_nl = free_sigma_nl
        self.prepare_model_flag = prepare_model_flag
        
        if (self.verbose):
            print "         Parameters: ", self.get_all_params()

        # Recomputing the correlation function many times can slow down the fitting, so we only do this if we have too. 
        # If free_sigma_nl is True then we need to recompute xi every iteration, and so we evaluate this exactly at alpha*s. 
        # If free_sigma_nl is False, then we are better off precomputing xi in narrow bins (self.xnarrow) 
        if (xnarrow is not None):
            if (hasattr(xnarrow,'__len__')):
                self.xnarrow = np.array(xnarrow)
            else:
                self.xnarrow = np.array([xnarrow])
        else:
            self.xnarrow = np.linspace(0.0, 400.0, 800)

        if (self.datatype == "CorrelationFunction"):
            oldBAO = self.BAO
            self.BAO = False
            self.xismooth = self.precompute_xi()
            self.BAO = oldBAO
            if (not self.free_sigma_nl):
                self.xi = self.precompute_xi()

    # Return the names of the all parameters for the model. For this class, it is the same as all free parameters
    def get_all_params(self):
        return self.get_free_params()

    # Return the names of the free parameters for the model, depending on the data we are fitting, whether or not we include the BAO feature, 
    # and whether we are allowing sigma_nl to vary
    def get_free_params(self):

        if ((self.datatype == "PowerSpectrum") or (self.datatype == "BAOExtract")):
            if (self.BAO):
                if (self.free_sigma_nl):
                    return ["alpha","sigma_nl","B","a1","a2","a3","a4","a5"]
                else:
                    return ["alpha","B","a1","a2","a3","a4","a5"]
            else:
                return ["B","a1","a2","a3","a4","a5"]
        elif (self.datatype == "CorrelationFunction"):
            if (self.BAO):
                if (self.free_sigma_nl):
                    return ["alpha","sigma_nl","B","a1","a2","a3"]
                else:
                    return ["alpha","B","a1","a2","a3"]
            else:
                return ["alpha","B","a1","a2","a3"]
        else:
            self.datatype_error()

    # Return the names of the free parameters for the model in a way that is suitable for plotting
    def get_latex_params(self):

        if ((self.datatype == "PowerSpectrum") or (self.datatype == "BAOExtract")):
            if (self.BAO):
                if (self.free_sigma_nl):
                    return [r"$\alpha$",r"$\Sigma_{nl}$",r"$B$",r"$a_{1}$",r"$a_{2}$",r"$a_{3}$",r"$a_{4}$",r"$a_{5}$"]
                else:
                    return [r"$\alpha$",r"$B$",r"$a_{1}$",r"$a_{2}$",r"$a_{3}$",r"$a_{4}$",r"$a_{5}$"]
            else:
                return [r"$B$",r"$a_{1}$",r"$a_{2}$",r"$a_{3}$",r"$a_{4}$",r"$a_{5}$"]
        elif (self.datatype == "CorrelationFunction"):
            if (self.BAO):
                if (self.free_sigma_nl):
                    return [r"$\alpha$",r"$\Sigma_{nl}$",r"$B$",r"$a_{1}$",r"$a_{2}$",r"$a_{3}$"]
                else:
                    return [r"$\alpha$",r"$B$",r"$a_{1}$",r"$a_{2}$",r"$a_{3}$"]
            else:
                return [r"$\alpha$",r"$B$",r"$a_{1}$",r"$a_{2}$",r"$a_{3}$"]
        else:
            self.datatype_error()

    # This routine evaluates the model with and without the BAO feature at the values of x. If no x is provided it uses the x values stored in self.x.
    # We save time by optionally passing in a correlation function array, as we only need to compute this once for each value of alpha, but might want to 
    # marginalise over other parameters (i.e., if using "List" based fitting)
    def compute_model(self, x=None):

        if (x is None):
            x = self.x
        else:
            if (hasattr(x,'__len__')):
                x = np.array(x)
            else:
                x = np.array([x])

        if ((self.datatype == "PowerSpectrum") or (self.datatype == "BAOExtract")):
            pksmoothspline = interpolate.splrep(self.power.k, self.power.pksmooth)
            Apoly = self.params["a1"][0]*x + self.params["a2"][0] + self.params["a3"][0]/x + self.params["a4"][0]/x**2 + self.params["a5"][0]/x**3
            if (self.BAO):
                pkrat = self.power.pk/self.power.pksmooth
                dewiggled = 1.0 + (pkrat - 1.0)*np.exp(-0.5*self.power.k**2*self.params["sigma_nl"][0]**2)
                dewiggledspline = interpolate.splrep(self.power.k,dewiggled)
                return (self.params["B"][0]*self.norm*interpolate.splev(x,pksmoothspline) + Apoly)*interpolate.splev(x/self.params["alpha"][0],dewiggledspline)
            else:
                return self.params["B"][0]*self.norm*interpolate.splev(x,pksmoothspline) + Apoly
        elif (self.datatype == "CorrelationFunction"):
            Apoly = self.params["a1"][0]/x**2 + self.params["a2"][0]/x + self.params["a3"][0]
            if (self.BAO):
                if (self.free_sigma_nl):
                    xi = self.compute_xi(self.params["alpha"][0]*x)
                else:
                    xi = interpolate.splev(self.params["alpha"][0]*x,self.xi)
            else:
                xi = interpolate.splev(self.params["alpha"][0]*x,self.xismooth)
            return self.params["B"][0]*self.norm*xi + Apoly
        else:
            self.datatype_error()

    # Precomputes values for a correlation function and returns a spline
    def precompute_xi(self, x=None):

        if (x is None):
            x = self.xnarrow
        else:
            if (hasattr(x,'__len__')):
                x = np.array(x)
            else:
                x = np.array([x])

        if (self.verbose):
            if (self.BAO):
                print "         Precomputing BAO correlation function"
            else:
                print "         Precomputing smooth correlation function"

        return interpolate.splrep(x, self.compute_xi(x))

    # Separate routine just to compute the correlation function with BAO feature, but without adding bias or polynomial terms
    def compute_xi(self, x=None):

        if (x is None):
            x = self.x
        else:
            if (hasattr(x,'__len__')):
                x = np.array(x)
            else:
                x = np.array([x])

        ft = SymmetricFourierTransform(ndim=3, N = 3200, h = 0.001)

        if (x[0] < 1.0e-3):
            x[0] = 1.0e-3
        if (self.BAO):
            pkrat = self.power.pk/self.power.pksmooth
            dewiggled = 1.0 + (pkrat - 1.0)*np.exp(-0.5*self.power.k**2*self.params["sigma_nl"][0]**2)
            pkwiggled = self.power.pksmooth*dewiggled
            pkwiggledspline = interpolate.splrep(self.power.k, pkwiggled)
            f = lambda k: interpolate.splev(k, pkwiggledspline)
            xi = ft.transform(f,x,inverse=True, ret_err=False)
        else:
            pksmoothspline = interpolate.splrep(self.power.k, self.power.pksmooth)
            f = lambda k: interpolate.splev(k, pksmoothspline)
            xi = ft.transform(f,x,inverse=True, ret_err=False)

        return xi

    def datatype_error(self):
        print "Datatype ", self.datatype, " not supported for Polynomial class, must be either 'PowerSpectrum', 'CorrelationFunction' or 'BAOExtract'"
        exit()

# This class implements a Full Shape model of the power spectrum based on that in Appendix A of Howlett, in prep, which in turn is based on Vlah2012,2013; Okumura2014 and Saito2014
class FullShape(object):

    def __init__(self, datatype, powerspectrum, nonlinearterms=None, x=None, alpha=1.0, sigma_nl=10.0, b1sigma8=1.0, b2sigma8=0.0, fsigma8=0.527, sigma8=0.8340, 
                       sigmav=10.0, gamma=0.0, s0=1.0, free_sigma_nl=True, free_sigma8=True, BAO=True, prepare_model_flag=False, remove_kaiser=False, verbose=False):

        if ((datatype != "PowerSpectrum") and (datatype != "CorrelationFunction") and (datatype != "BAOExtract")):
            print "Datatype ", datatype, " not supported for FullShape class, must be either 'PowerSpectrum', 'CorrelationFunction' or 'BAOExtract'"
            exit()

        self.verbose = verbose
        self.datatype = datatype
        self.power = powerspectrum

        if (self.verbose):
            print "Setting up Full Shape model:  "

        if (nonlinearterms is None):
            print "No non-linear terms found, computing these from the smooth linear power spectrum. Note this may take some time, but they only need to be"
            print "computed once for a given linear power spectrum (regardless of sigma8, f etc.). They will be stored in the file compute_pt_integrals_output.dat for future reference"
            self.nonlinearterms = self.compute_nonlinearterms(self.power.k, self.power.pksmooth)
        else:
            self.nonlinearterms = self.read_nonlinearterms(nonlinearterms)

        # The x and y values for the model
        if (x is not None):
            if (hasattr(x,'__len__')):
                self.x = np.array(x)
            else:
                self.x = np.array([x])
        else:
            if ((self.datatype == "PowerSpectrum") or (self.datatype == "BAOExtract")):
                self.x = np.linspace(0.0, 0.3, 40)
            elif (self.datatype == "CorrelationFunction"):
                self.x = np.linspace(0.0, 200.0, 40)
            else:
                self.datatype_error()
        self.y = np.zeros(len(self.x))

        # The parameters of the model. Self.params contains a dictionary of all the free parameters for the model. Each dictionary element contains:
        #   the current value
        #   the prior type, one of "Linear", "Log", "Gaussian" or "LogGaussian"
        #   the values of the prior. These are lower/upper limits for flat priors, central value and standard deviation for (Log)Gaussian priors.
        self.params = {"alpha":     [alpha,     "Linear",  0.7,   1.3],
                       "sigma_nl":  [sigma_nl,  "Linear",  2.0,  20.0],
                       "b1sigma8":  [b1sigma8,  "Linear",  0.1,  12.0],
                       "fsigma8":   [fsigma8,   "Linear",  0.1,   2.0],
                       "sigma8":    [sigma8,    "Linear",  0.1,   2.0],
                       "sigmav":    [sigmav,    "Linear",  0.01, 30.0]}
        if ((self.datatype == "PowerSpectrum") or (self.datatype == "BAOExtract")):
            self.params["b2sigma8"] = [b2sigma8,  "Linear", -2.0, 30.0]
        elif (self.datatype == "CorrelationFunction"):
            self.params["gamma"]    = [gamma,  "Linear", -4.0,   4.0]
            self.params["s0"]       = [s0,     "Linear",  0.01, 10.0]
        else:
            self.datatype_error()

        # Some choices required for the model, whether or not to include the BAO feature and to fix or vary fsigma8 and sigma8.
        self.BAO = BAO
        self.free_sigma_nl = free_sigma_nl
        self.free_sigma8 = free_sigma8
        self.prepare_model_flag = prepare_model_flag
        self.remove_kaiser = remove_kaiser  # This divides the model by the Kaiser boost factor, which is often done for the data during reconstruction
        
        if (self.verbose):
            print "         Parameters: ", self.get_all_params()

        return

    # Return the names of the all parameters for the model. For this class, it is the same as all free parameters
    def get_all_params(self):
        return self.get_free_params()

    # Return the names of the free parameters for the model, depending on the data we are fitting, whether or not we include the BAO feature, 
    # and whether we are allowing sigma_nl to vary
    def get_free_params(self):

        if ((self.datatype == "PowerSpectrum") or (self.datatype == "BAOExtract")):
            if (self.BAO):
                if (self.free_sigma_nl):
                    if (self.free_sigma8):
                        return ["alpha","sigma_nl","b1sigma8","b2sigma8","fsigma8","sigma8","sigmav"]
                    else:
                        return ["alpha","sigma_nl","b1sigma8","b2sigma8","fsigma8","sigmav"]
                else:
                    if (self.free_sigma8):
                        return ["alpha","b1sigma8","b2sigma8","fsigma8","sigma8","sigmav"]
                    else:
                        return ["alpha","b1sigma8","b2sigma8","fsigma8","sigmav"]
            else:
                if (self.free_sigma8):
                    return ["alpha","b1sigma8","b2sigma8","fsigma8","sigma8","sigmav"]
                else:
                    return ["alpha","b1sigma8","b2sigma8","fsigma8","sigmav"]
        elif (self.datatype == "CorrelationFunction"):
            if (self.BAO):
                if (self.free_sigma_nl):
                    if (self.free_sigma8):
                        return ["alpha","sigma_nl","b1sigma8","s0","gamma","fsigma8","sigma8","sigmav"]
                    else:
                        return ["alpha","sigma_nl","b1sigma8","s0","gamma","fsigma8","sigmav"]
                else:
                    if (self.free_sigma8):
                        return ["alpha","b1sigma8","s0","gamma","fsigma8","sigma8","sigmav"]
                    else:
                        return ["alpha","b1sigma8","s0","gamma","fsigma8","sigmav"]
            else:
                if (self.free_sigma8):
                    return ["alpha","b1sigma8","s0","gamma","fsigma8","sigma8","sigmav"]
                else:
                    return ["alpha","b1sigma8","s0","gamma","fsigma8","sigmav"]
        else:
            self.datatype_error()

    # Return the names of the free parameters for the model in latex format suitable for plotting
    def get_latex_params(self):

        if ((self.datatype == "PowerSpectrum") or (self.datatype == "BAOExtract")):
            if (self.BAO):
                if (self.free_sigma_nl):
                    if (self.free_sigma8):
                        return [r"$\alpha$",r"$\Sigma_{nl}$",r"$b_{1}\sigma_{8}$",r"$b_{2}\sigma_{8}$",r"$f\sigma_{8}$",r"$\sigma_{8}$",r"$\sigma_{v}$"]
                    else:
                        return [r"$\alpha$",r"$\Sigma_{nl}$",r"$b_{1}\sigma_{8}$",r"$b_{2}\sigma_{8}$",r"$f\sigma_{8}$",r"$\sigma_{v}$"]
                else:
                    if (self.free_sigma8):
                        return [r"$\alpha$",r"$b_{1}\sigma_{8}$",r"$b_{2}\sigma_{8}$",r"$f\sigma_{8}$",r"$\sigma_{8}$",r"$\sigma_{v}$"]
                    else:
                        return [r"$\alpha$",r"$b_{1}\sigma_{8}$",r"$b_{2}\sigma_{8}$",r"$f\sigma_{8}$",r"$\sigma_{v}$"]
            else:
                if (self.free_sigma8):
                    return [r"$\alpha$",r"$b_{1}\sigma_{8}$",r"$b_{2}\sigma_{8}$",r"$f\sigma_{8}$",r"$\sigma_{8}$",r"$\sigma_{v}$"]
                else:
                    return [r"$\alpha$",r"$b_{1}\sigma_{8}$",r"$b_{2}\sigma_{8}$",r"$f\sigma_{8}$",r"$\sigma_{v}$"]
        elif (self.datatype == "CorrelationFunction"):
            if (self.BAO):
                if (self.free_sigma_nl):
                    if (self.free_sigma8):
                        return [r"$\alpha$",r"$\Sigma_{nl}$",r"$b_{1}\sigma_{8}$",r"$s_{0}$","\gamma",r"$f\sigma_{8}$",r"$\sigma_{8}$",r"$\sigma_{v}$"]
                    else:
                        return [r"$\alpha$",r"$\Sigma_{nl}$",r"$b_{1}\sigma_{8}$",r"$s_{0}$","\gamma",r"$f\sigma_{8}$",r"$\sigma_{v}$"]
                else:
                    if (self.free_sigma8):
                        return [r"$\alpha$",r"$b_{1}\sigma_{8}$",r"$s_{0}$","\gamma",r"$f\sigma_{8}$",r"$\sigma_{8}$",r"$\sigma_{v}$"]
                    else:
                        return [r"$\alpha$",r"$b_{1}\sigma_{8}$",r"$s_{0}$","\gamma",r"$f\sigma_{8}$",r"$\sigma_{v}$"]
            else:
                if (self.free_sigma8):
                    return [r"$\alpha$",r"$b_{1}\sigma_{8}$",r"$s_{0}$","\gamma",r"$f\sigma_{8}$",r"$\sigma_{8}$",r"$\sigma_{v}$"]
                else:
                    return [r"$\alpha$",r"$b_{1}\sigma_{8}$",r"$s_{0}$","\gamma",r"$f\sigma_{8}$",r"$\sigma_{v}$"]
        else:
            self.datatype_error()

    # This routine evaluates the model with and without the BAO feature at the values of x. If no x is provided it uses the x values stored in self.x.
    # We save time by optionally passing in a correlation function array, as we only need to compute this once for each value of alpha, but might want to 
    # marginalise over other parameters (i.e., if using "List" based fitting)
    def compute_model(self, x=None):

        if (x is None):
            x = self.x
        else:
            if (hasattr(x,'__len__')):
                x = np.array(x)
            else:
                x = np.array([x])

        pksmoothspline = self.compute_pksmooth_nl()
        pkrat = self.power.pk/self.power.pksmooth
        dewiggled = 1.0 + (pkrat - 1.0)*np.exp(-0.5*self.power.k**2*self.params["sigma_nl"][0]**2)
        dewiggledspline = interpolate.splrep(self.power.k,dewiggled)

        if ((self.datatype == "PowerSpectrum") or (self.datatype == "BAOExtract")):
            if (self.BAO):
                return interpolate.splev(x,pksmoothspline)*interpolate.splev(x/self.params["alpha"][0],dewiggledspline)#/self.params["alpha"][0]**3
            else:
                return interpolate.splev(x,pksmoothspline)#/self.params["alpha"][0]**3
        elif (self.datatype == "CorrelationFunction"):
            if (self.params["alpha"][0]*x[0] < 1.0e-3):
                x[0] = 1.0e-3
            ft = SymmetricFourierTransform(ndim=3, N = 3200, h = 0.001)
            if (self.BAO):
                f = lambda k: np.piecewise(k, [k<self.nonlinearterms[0,0], np.logical_and(k>=self.nonlinearterms[0,0],k<=self.nonlinearterms[-1,0]), k>self.nonlinearterms[-1,0]], [0.0, lambda y: interpolate.splev(y, pksmoothspline)*interpolate.splev(y, dewiggledspline), 0.0])
            else:
                f = lambda k: np.piecewise(k, [k<self.nonlinearterms[0,0], np.logical_and(k>=self.nonlinearterms[0,0],k<=self.nonlinearterms[-1,0]), k>self.nonlinearterms[-1,0]], [0.0, lambda y: interpolate.splev(y, pksmoothspline), 0.0])
            return (1.0 + (self.params["alpha"][0]*x/self.params["s0"][0])**self.params["gamma"][0])*ft.transform(f,self.params["alpha"][0]*x,inverse=True,ret_err=False)
        else:
            self.datatype_error()

    def __xiintegrand(self, k, x, pkspline):
        if ((k < self.nonlinearterms[0,0]) or (k > self.nonlinearterms[-1,0])):
            pk = 0.0
        else:
            pk = interpolate.splev(k, pkspline)
        return k*pk*np.sin(k*x)/x*np.exp(-k**2)

    # Routine to add all the PT integrals togther in the correct way and evaluate the FullShape power spectrum
    def compute_pksmooth_nl(self):

        if ((self.datatype == "PowerSpectrum") or (self.datatype == "BAOExtract")):

            b1 = self.params["b1sigma8"][0]/self.params["sigma8"][0]
            b2 = self.params["b2sigma8"][0]/self.params["sigma8"][0]
            f = self.params["fsigma8"][0]/self.params["sigma8"][0]
            sigma8rat = (self.params["sigma8"][0]/self.power.sigma8)**2
            bs = -4.0/7.0*(b1-1.0)
            b3nl = 32.0/315.0*(b1-1.0) 
            NL = self.nonlinearterms.T

            P_00_A =  b1**2*sigma8rat*(NL[1,0:] + 2.0*sigma8rat*(NL[2,0:] + 3.0*NL[0,0:]**2*NL[1,0:]*NL[18,0:]))
            P_00_B =  2.0*b1*sigma8rat**2*(b2*NL[24,0:] + bs*NL[25,0:] + b3nl*NL[1,0:]*NL[37,0:])
            P_00_C =  0.5*b2**2*sigma8rat**2*NL[26,0:] + 0.5*bs**2*sigma8rat**2*NL[27,0:] + b2*bs*sigma8rat**2*NL[28,0:]
            P_01_A =  2.0/3.0*f*b1*sigma8rat*(NL[1,0:] + 2.0*sigma8rat*(NL[3,0:] + b1*NL[6,0:] + 3.0*NL[0,0:]**2*NL[1,0:]*(NL[19,0:] + b1*NL[21,0:])))
            P_01_B =  2.0/3.0*f*sigma8rat**2*(b2*NL[29,0:] + bs*NL[30,0:] + b1*b2*NL[31,0:] + b1*bs*NL[32,0:] + b3nl*NL[1,0:]*NL[37,0:])
            P_02_A =  b1*f**2*sigma8rat**2*(NL[4,0:]/3.0 + NL[10,0:]/5.0 + 2.0*NL[0,0:]**2*NL[1,0:]*(NL[20,0:]/3.0 + NL[23,0:]/5.0))
            P_02_B =  0.5*b2*f**2*sigma8rat**2*(NL[33,0:]/3.0 + NL[35,0:]/5.0) + 0.5*bs*f**2*sigma8rat**2*(NL[34,0:]/3.0 + NL[36,0:]/5.0) 
            P_11_A =  f**2*sigma8rat*(NL[1,0:]/5.0 + sigma8rat*(2.0*NL[7,0:]/5.0 + 4.0*b1*NL[12,0:]/5.0 + b1**2*(NL[15,0:]/3.0 + NL[9,0:]/5.0) + 6.0*NL[0,0:]**2*NL[1,0:]/5.0*(NL[22,0:] + 2.0*b1*NL[21,0:])))
            P_12_A =  f**3*sigma8rat**2*(NL[8,0:]/5.0 + NL[11,0:]/7.0 - b1*(NL[5,0:]/5.0 + NL[14,0:]/7.0) + 2.0*NL[0,0:]**2*NL[1,0:]*(NL[20,0:]/5.0 + NL[23,0:]/7.0))
            P_22_A =  1.0/16.0*f**4*sigma8rat**2*(NL[13,0:]/5.0 + 2.0*NL[16,0:]/7.0 + NL[17,0:]/9.0)

            P_0 = P_00_A + P_00_B + P_00_C + P_01_A + P_01_B + P_02_A + P_02_B + P_11_A + P_12_A + P_22_A
            Dfog = 1.0/(1.0+NL[0,0:]**2*self.params["sigmav"][0]**2/2.0)**2
            Kaiser = 1.0
            if (self.remove_kaiser):
                Kaiser = b1**2 + 2.0/3.0*b1*f + 1.0/5.0*f**2
            pksmoothspline = interpolate.splrep(NL[0,0:], P_0*Dfog/Kaiser, s=0)

        elif (self.datatype == "CorrelationFunction"):

            b1 = self.params["b1sigma8"][0]/self.params["sigma8"][0]
            f = self.params["fsigma8"][0]/self.params["sigma8"][0]
            sigma8rat = (self.params["sigma8"][0]/self.power.sigma8)**2
            NL = self.nonlinearterms.T

            P_00_A =  b1**2*sigma8rat*(NL[1,0:] + 2.0*sigma8rat*(NL[2,0:] + 3.0*NL[0,0:]**2*NL[1,0:]*NL[18,0:]))
            P_01_A =  2.0/3.0*f*b1*sigma8rat*(NL[1,0:] + 2.0*sigma8rat*(NL[3,0:] + b1*NL[6,0:] + 3.0*NL[0,0:]**2*NL[1,0:]*(NL[19,0:] + b1*NL[21,0:])))
            P_02_A =  b1*f**2*sigma8rat**2*(NL[4,0:]/3.0 + NL[10,0:]/5.0 + 2.0*NL[0,0:]**2*NL[1,0:]*(NL[20,0:]/3.0 + NL[23,0:]/5.0))
            P_11_A =  f**2*sigma8rat*(NL[1,0:]/5.0 + sigma8rat*(2.0*NL[7,0:]/5.0 + 4.0*b1*NL[12,0:]/5.0 + b1**2*(NL[15,0:]/3.0 + NL[9,0:]/5.0) + 6.0*NL[0,0:]**2*NL[1,0:]/5.0*(NL[22,0:] + 2.0*b1*NL[21,0:])))
            P_12_A =  f**3*sigma8rat**2*(NL[8,0:]/5.0 + NL[11,0:]/7.0 - b1*(NL[5,0:]/5.0 + NL[14,0:]/7.0) + 2.0*NL[0,0:]**2*NL[1,0:]*(NL[20,0:]/5.0 + NL[23,0:]/7.0))
            P_22_A =  1.0/16.0*f**4*sigma8rat**2*(NL[13,0:]/5.0 + 2.0*NL[16,0:]/7.0 + NL[17,0:]/9.0)

            P_0 = P_00_A + P_01_A + P_02_A + P_11_A + P_12_A + P_22_A
            Dfog = 1.0/(1.0+NL[0,0:]**2*self.params["sigmav"][0]**2/2.0)**2
            Kaiser = 1.0
            if (self.remove_kaiser):
                Kaiser = b1**2 + 2.0/3.0*b1*f + 1.0/5.0*f**2
            pksmoothspline = interpolate.splrep(NL[0,0:], P_0*Dfog/Kaiser, s=0)

        else:
            self.datatype_error()

        return pksmoothspline

    # Compute the many non-linear terms needed for the model. This actually uses a C code as it's faster, so this just wraps that.
    def compute_nonlinearterms(self, k, pk):

        np.savetxt("./files/compute_pt_integrals_input.dat", np.c_[k, pk], fmt="%g %g", header="k    pksmooth", delimiter='  ')
        check_call(["gcc","./src/compute_pt_integrals.c", "-o", "./src/compute_pt_integrals", "-lm", "-lgsl", "-lgslcblas"])
        check_call(["./src/compute_pt_integrals","-infile", "./files/compute_pt_integrals_input.dat", "-outfile", "./files/compute_pt_integrals_output.dat", "-koutmin=0.0001", "-koutmax=1.0", "-nkout=400"])
        nonlinearterms = self.read_nonlinearterms("./files/compute_pt_integrals_output.dat")

        return nonlinearterms

    # Read in the non-linear terms needed for the model from a file.
    def read_nonlinearterms(self, nonlinearfile):

        nonlinearterms = np.loadtxt(nonlinearfile)
        return nonlinearterms

    def datatype_error(self):
        print "Datatype ", self.datatype, " not supported for FullShape class, must be either 'PowerSpectrum', 'CorrelationFunction' or 'BAOextract'"
        exit()

# This class implements the BAO extractor method of Noda 2017. It inherits the properties of the FullShape model, and additionally computes R[P](k,n,Delta) based on the 
# values stored in the model
class BAOExtractor(FullShape):

    def __init__(self, powerspectrum, n=0.0, Delta=0.5, nonlinearterms=None, x=None, alpha=1.0, sigma_nl=10.0, b1sigma8=1.0, b2sigma8=0.0, fsigma8=0.527, sigma8=0.8340, 
                       sigmav=10.0, gamma=0.0, s0=1.0, free_sigma_nl=True, free_sigma8=True, BAO=True, prepare_model_flag=False, remove_kaiser=False, verbose=False):

        FullShape.__init__(self, "BAOExtract", powerspectrum, nonlinearterms=nonlinearterms, x=x, alpha=alpha, sigma_nl=sigma_nl, b1sigma8=b1sigma8, b2sigma8=b2sigma8, fsigma8=fsigma8, sigma8=sigma8, 
                            sigmav=sigmav, gamma=gamma, s0=s0, free_sigma_nl=free_sigma_nl, free_sigma8=free_sigma8, BAO=BAO, prepare_model_flag=prepare_model_flag, remove_kaiser=remove_kaiser, verbose=verbose)

        self.n = n
        self.kwidth = 2.0*math.pi*Delta/powerspectrum.r_s

        return

    def extract_BAO(self, k, pk):

        BAOextract = np.zeros(len(k))
        for i in range(len(k)):
            if (pk[i] > 0):
                index = np.where(np.fabs(k-k[i]) <= self.kwidth)[0]
                numer = np.sum((k[index]-k[i])**(2.0*self.n)*(1.0 - pk[index]/pk[i]))
                denom = np.sum((k[index]-k[i])**(2.0*self.n)*(1.0 - np.cos(self.power.r_s*(k[index]-k[i]))))
                if (denom > 0):
                    BAOextract[i] = numer/denom

        return BAOextract

 
# Implements the LinearPoint fitting for the correlation function. This method fits the correlation function with an odd-order polynomial and then evaluates the linear-point. There 
# are a number of ways to do this, which are controlled by the parameter lpoint which should be between -1.0 and 1.0. If lpoint is zero or negative, the linear point is the inflection 
# point of the polynomial.  Otherwise it is the fraction of the total linear point contribution that comes from the "dip", such that LP = lpoint*s_dip + (1.0-lpoint)*s_peak
class LinearPoint(object):

    def __init__(self, powerspectrum, x=None, polyorder=5, lpoint=-1.0, offset=100.0, polyterms=None, BAO=True, prepare_model_flag=True, 
                    LP_theory=None, polyorder_LP_theory=15, xmin_LP_theory=70.0, xmax_LP_theory=130.0, plot_LP_theory=False, verbose=False):

        self.verbose = verbose
        self.datatype = "CorrelationFunction"
        self.power = powerspectrum
        self.polyorder = polyorder
        self.lpoint = lpoint
        self.offset = offset

        if (polyorder%2 == 0):
            print "Polynomial order must be odd"
            exit()

        if (polyterms is None):
            polyterms = np.zeros(self.polyorder+1)
        else:
            if (len(polyterms) != self.polyorder+1):
                print "Provided polyterms (", polyterms, ") must be a list of length polyorder+1 (", self.polyorder+1, ")"
                exit()

        # The x and y values for the model
        if (x is not None):
            if (hasattr(x,'__len__')):
                self.x = np.array(x)
            else:
                self.x = np.array([x])
        else:
            self.x = np.linspace(0.0, 200.0, 40)
        self.y = np.zeros(len(self.x))
        self.xismooth = np.zeros(len(self.x))

        # We treat the Linear point and alpha as parameters here, but in reality they are not
        # free. They are derived parameters based on the polynomial terms and the value of self.LP_theory
        self.params = {"LP": [100.0, "Linear", 0.0, 200.0],
                       "alpha": [1.0, "Linear", 0.7, 1.3]}
        for i in range(self.polyorder+1):
            param_name = str("a%d" % i)
            self.params[param_name] = [polyterms[i], "Linear", -10.0, 10.0]

        # For this model BAO sets whether we return the xismooth model computed in a similar way to the Polynomial class, or the LinearPoint
        # polynomial fit. xismooth is computed just before fitting if prepare_model_flag is True
        self.BAO = BAO
        self.prepare_model_flag = prepare_model_flag

        if (self.verbose):
            print "         Parameters: ", self.get_all_params()

        # Compute the linear point for the linear theory model
        if (LP_theory is None):
            self.LP_theory = 1.0
            self.LP_theory = self.compute_LP_theory(npoly=polyorder_LP_theory, xmin=xmin_LP_theory, xmax=xmax_LP_theory, do_plot=plot_LP_theory)
        else:
            self.LP_theory = LP_theory

        return

    # Return the names of the all parameters for the model. For this class, it is all free parameters plus the Linear point and alpha, which are derived parameters
    def get_all_params(self):
        return self.get_free_params() + ["LP", "alpha"]
  
    # Return the names of the free parameters for the model. The parameters for this model are simply a0-an where n is the polynomial order
    def get_free_params(self):

        param_name = []
        for i in range(self.polyorder+1):
            param_name.append(str("a%d" % i))

        return param_name

    # Return the names of the free parameters for the model in latex format suitable for plotting
    def get_latex_params(self):

        param_name = []
        for i in range(self.polyorder+1):
            param_name.append(str(r"$a_{%d}$" % i))

        return param_name + [r"$LP$", r"$\alpha$"]

    # This routine evaluates the model for the LinearPoint, which is just an nth order polynomial evaluated at x
    def compute_model(self, x=None):

        if (x is None):
            x = self.x
        else:
            if (hasattr(x,'__len__')):
                x = np.array(x)
            else:
                x = np.array([x])

        if (self.BAO):

            # Set up the polynomial
            x -= self.offset
            free_params = self.get_free_params()
            polyterms = np.empty(len(free_params))
            for counter, i in enumerate(free_params):
                polyterms[counter] = self.params[i][0]
            poly = np.poly1d(polyterms[::-1])

            return poly(x)+interpolate.splev(x, self.xismooth)

        else:

            return interpolate.splev(x, self.xismooth)

    # Routine to take the matter power spectrum, convert it to a narrow binned correlation function and compute the linear point using a very high order polynomial
    def compute_LP_theory(self, npoly=15, xmin=70.0, xmax=130.0, do_plot=0):

        # Compute the correlation function
        x = np.linspace(xmin, xmax, int((xmax-xmin)/0.1))
        ft = SymmetricFourierTransform(ndim=3, N = 3000, h = 0.001)

        # For fitting the linear theory model, repositioning the data so that the linear is near zero seems to work better.
        pkspline = interpolate.splrep(self.power.k, self.power.pk)
        f = lambda k: interpolate.splev(k, pkspline)
        xi = ft.transform(f,x,inverse=True, ret_err=False)

        offset = 100.0
        polyterms = optimize.minimize(self.fit_LP_theory, np.zeros(npoly), args=(x-offset, xi), method="Nelder-Mead", options={'maxiter':500000, 'fatol':1.0e-15})["x"]
        polyfit = lambda x, *free_params: np.poly1d([i for i in free_params][0][::-1])(x)

        LP_theory = self.compute_LP(polyterms, offset=offset)+offset

        if (do_plot):

            # For plotting we want a smooth model. We fit the data in the same way as for the Polynomial class
            pkspline = interpolate.splrep(self.power.k, self.power.pksmooth)
            f = lambda k: interpolate.splev(k, pkspline)
            xismooth = ft.transform(f,x,inverse=True, ret_err=False)

            begin = [1.0, 0.0, 0.0, 0.0]
            smooth_params = optimize.minimize(self.fit_LP_theory_smooth, begin, args=(x, xi, xismooth), method="Nelder-Mead", options={'maxiter':500000, 'fatol':1.0e-15})["x"]
            Apoly = smooth_params[1]/x**2 + smooth_params[2]/x + smooth_params[3]
            xismooth = smooth_params[0]*xismooth+Apoly

            fig = plt.figure(100)
            gs = gridspec.GridSpec(2, 1, height_ratios=[1.0,1.0], hspace=0.0, left=0.13, bottom=0.1, right=0.95, top=0.98)

            big_axes=plt.subplot(gs[0:])
            big_axes.set_axis_bgcolor('none')
            big_axes.spines['top'].set_color('none')
            big_axes.spines['bottom'].set_color('none')
            big_axes.spines['left'].set_color('none')
            big_axes.spines['right'].set_color('none')
            big_axes.set_xlabel(r'$s\,(h^{-1}\,\mathrm{Mpc})$',fontsize=22)
            big_axes.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

            plt_handle = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
            plt_handle[0].errorbar(x,x**2*xi,marker='o',markerfacecolor='w',markeredgecolor='k',color='k',linestyle='None',markeredgewidth=0.8,markersize=3,zorder=0) 
            plt_handle[0].errorbar(x,x**2*polyfit(x-offset,polyterms),color='r',linestyle='-',linewidth=1.3,zorder=5)
            plt_handle[0].errorbar(x,x**2*xismooth,color='k',linestyle='--',linewidth=1.3,zorder=5)
            plt_handle[1].errorbar(x,(xi-xismooth),marker='o',markerfacecolor='w',markeredgecolor='k',color='k',linestyle='None',markeredgewidth=0.8,markersize=3,zorder=0) 
            plt_handle[1].errorbar(x,(polyfit(x-offset,polyterms)-xismooth),color='r',linestyle='-',linewidth=1.3,zorder=5)
            plt_handle[0].axvline(x=LP_theory, linestyle='--', linewidth=1.3)
            plt_handle[1].axvline(x=LP_theory, linestyle='--', linewidth=1.3)
            plt_handle[0].set_xlim(0.95*np.amin(x), 1.05*np.amax(x))
            plt_handle[1].set_xlim(0.95*np.amin(x), 1.05*np.amax(x))
            if (np.amin(x*x*xi) < 0.0):
                plt_handle[0].set_ylim(1.05*np.amin(x*x*xi), 1.05*np.amax(x*x*xi))
            else:
                plt_handle[0].set_ylim(0.95*np.amin(x*x*xi), 1.05*np.amax(x*x*xi))
            plt_handle[1].set_ylim(-2.0e-3, 2.0e-3)
            plt_handle[0].set_ylabel(r'$s^{2}\,\xi(s)\,(h^{-2}Mpc^{2})$',fontsize=22,labelpad=5)
            plt_handle[1].set_ylabel(r'$\xi(s)-\xi^{\mathrm{nw}}(s)$',fontsize=22,labelpad=5)
            for i in plt_handle:
                i.tick_params(width=1.3)
                i.tick_params('both',length=10, which='major')
                i.tick_params('both',length=5, which='minor')
                for axis in ['top','left','bottom','right']:
                    i.spines[axis].set_linewidth(1.3)
                for tick in i.xaxis.get_ticklabels():
                    tick.set_fontsize(14)
                for tick in i.yaxis.get_ticklabels():
                    tick.set_fontsize(14)
                i.set_autoscale_on(False)
            plt.show()

        return LP_theory

    # Routines for fitting the models to the linear theory correlation function using least-squares
    def fit_LP_theory(self, params, x, xi):

        # Set up the polynomial
        poly = np.poly1d(params[::-1])(x)
        return np.sum((xi-poly)**2)

    def fit_LP_theory_smooth(self, params, x, xi, xismooth):

        Apoly = params[1]/x**2 + params[2]/x + params[3]
        poly = params[0]*xismooth + Apoly
        return np.sum((xi-poly)**2)

    # Compute the linear point using polynomial terms and the value of lpoint.
    def compute_LP(self, polyterms, offset=0.0):

        # Set up the polynomial
        poly = np.poly1d(polyterms[::-1])

        # Find the linear point
        if (self.lpoint <= 0.0):

            # Identify the real inflection points. We are only interested in points of rising inflection after subtracting the smooth model
            # (which is based on the fact that the BAO feature should be a bump) There are a maximum of self.polyorder-4 
            # rising inflection points, so we choose between these by choosing the one nearest the 100 Mpc/h
            # (which is valid as we don't expect it to move except due to noise?)
            ip = poly.deriv(2).r
            ipvallow = poly(ip-0.01)
            ipvalhi = poly(ip+0.01)
            index = np.where(np.logical_and(ipvalhi>ipvallow,np.isreal(ip)))[0]
            if (len(index) > 0):
                ip = np.real(ip[index])
                index = np.argmin((np.abs(ip-(100.0-offset))))
                return ip[index]
            else:
                return np.inf

        else:

            # Identify the real stationary points. Identify whether these correspond to peaks or dips using the second derivative
            sp = np.sort(poly.deriv(1).r)
            index = np.where(np.isreal(sp))[0]
            sp = sp[index]
            spval = poly.deriv(2)(sp)
            spdip = np.real(sp[spval>=0.0])
            sppeak = np.real(sp[spval<=0.0])

            # For each dip, find the peak after this and compute the mid points
            if ((len(spdip) > 0) and (len(sppeak)>0)):
                sp = []
                for i in range(len(spdip)):
                    index = np.searchsorted(sppeak, spdip[i])
                    if (index == len(sppeak)):
                        continue
                    sp.append(self.lpoint*spdip[i] + (1.0-self.lpoint)*sppeak[index])
                sp = np.array(sp)
                if (len(sp) > 0):
                    index = np.argmin((np.abs(sp-(100.0-offset))))
                    return sp[index]
                else:
                    return np.inf
            else:
                return np.inf

        return

# Set new values for the priors. All prior information (type and variables) is stored
# in the self.params dictionary, so we just need a name and a list of ["Type", lower, upper,] to update this.
def set_prior(model, name, newprior):

    if (model.verbose):
        print "Changing prior on ", name, ": ", model.params[name][1:], " --> ", newprior

    # Check that the prior type is supported
    if ((newprior[0] != "Linear") and (newprior[0] != "Log") and (newprior[0] != "Gaussian") and (newprior[0] != "LogGaussian")):
        print "New prior type not supported, must be one of: Linear, Log, Gaussian or LogGaussian"
        exit()

    model.params[name][1] = newprior[0]
    model.params[name][2] = newprior[1]
    model.params[name][3] = newprior[2]

    return

# Calculate the values for the prior. We loop over the model.prior dictionary
# and calculate the prior appropriately, depending on the type. Supported types
# are "Linear", "Log", "Gaussian", "LogGaussian"
def compute_prior(model):

    priorsum = 0.0
    for i in model.get_free_params():
        if (model.params[i][1] == "Linear"):
            if (model.params[i][2] <= model.params[i][0] <= model.params[i][3]):
                priorsum += -np.log(model.params[i][3]-model.params[i][2])
            else:
                return -np.inf
        elif (model.params[i][1] == "Log"):
            if (model.params[i][2] <= np.log(model.params[i][0]) <= model.params[i][3]):
                priorsum += -np.log(model.params[i][3]-model.params[i][2])
            else:
                return -np.inf
        elif (model.params[i][1] == "Gaussian"):
            priorsum += -0.5*(model.params[i][0]-model.params[i][2])**2/model.params[i][3]**2
        elif (model.params[i][1] == "LogGaussian"):
            if (model.params[i][0] <= 0.0):
                return -np.inf
            else:
                priorsum += -0.5*(np.log(model.params[i][0])-np.log(model.params[i][2]))**2/model.params[i][3]**2
        else:
            print "Prior type ", model.params[i][1], "for ", i, "not supported, must be one of: Linear, Log, Gaussian or LogGaussian"
            exit()

    return priorsum

# This routine stores the y-values of the model. It is useful to have two separate functions, as we might want to compute a model but not ovewrite the 
# existing model in model.y (i.e., we can call compute_model to get the smooth power spectrum during fitting if we are plotting the BAO wiggles without 
# overwriting the BAO model stored in model.y)
def set_model(model, x=None):

    if (x is None):
        x = model.x
    else:
        if (hasattr(x,'__len__')):
            x = np.array(x)
            model.x = x
        else:
            x = np.array([x])
            model.x = x

    model.y = model.compute_model(x)

    return

