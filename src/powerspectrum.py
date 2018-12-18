# A class associated with matter power spectrum and its smooth/dewiggled counterpart

import math
import warnings
import numpy as np
from scipy import integrate, interpolate, optimize
from scipy.integrate import IntegrationWarning
warnings.filterwarnings('ignore', category=IntegrationWarning, append=True)

# This class uses the Eisenstein and Hu method and takes as input a tabulated matter power spectrum. We can set the value of r_s using the EH98 fitting formula for the sound horizon, 
# or passing in a value (i.e, from CAMB value) The latter is much more recommended! We also fit the input power spectrum with some broadband terms to recover the model.
class EH98Tabulated(object):

    def __init__(self, matterfile, omega_m=0.3121, omega_b=0.0491, hubble=0.6751, ns=0.9653, sigma8=None, r_s=None, verbose=False):

        self.verbose = verbose

        self.omega_m = omega_m
        self.omega_b = omega_b
        self.hubble = hubble
        self.ns = ns
        self.sigma8 = sigma8

        if (r_s is None):
            self.r_s = EH98_rs(self)
            if (self.verbose):
                print "Setting sound horizon using EH98 formulae: r_s = ", self.r_s
        else:
            self.r_s = r_s
            if (self.verbose):
                print "Setting sound horizon using user-defined value: r_s = ", self.r_s

        read_matterpower(self, matterfile)
        self.set_pksmooth()

    # These routine computes the smoothed power spectrum and assign it to self.pksmooth
    def set_pksmooth(self, k=None, pk=None):

        if (k is None):
            if (pk is not None):
                print "When setting pk smooth using kwargs, both k and pk must be supplied. You have only supplied pk"
                exit()
            k = self.k
            pk = self.pk
        else:
            if (pk is None):
                print "When setting pk smooth using kwargs, both k and pk must be supplied. You have only supplied k"
                exit()
            k = k
            pk = pk

        self.pksmooth = self.compute_pksmooth(k, pk)

        return

    # Compute the smooth dewiggled power spectrum using either a polynomial based method (i.e., Hinton 2017), or the EH98 fitting formulae
    def compute_pksmooth(self, k, pk):

        if (self.verbose):
            print "Computing smooth power spectrum using EH98 fitting formulae:  "

        # First compute the normalised Eisenstein and Hu smooth power spectrum
        pk_EH98 = k**self.ns*self.EH98_dewiggled(k)**2
        pk_EH98_spline = interpolate.splrep(k, pk_EH98)
        pk_EH98_norm = math.sqrt(integrate.quad(sigma8_integrand,k[0],k[-1],args=(k[0],k[-1],pk_EH98_spline))[0]/(2.0*math.pi*math.pi))
        pk_EH98 *= (self.sigma8/pk_EH98_norm)**2

        if (self.verbose):
            print "         Fitting input power spectrum to obtain broadband terms"
        nll = lambda *args: self.__EH98_lnlike(*args)
        start = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = optimize.minimize(nll, start, args=(k, pk_EH98, pk), method="Nelder-Mead", tol=1.0e-9, options={'maxiter': 1000000})
        if (self.verbose):
            print "         [B, a1, a2, a3, a4, a5] = ", result['x']

        # Then compute the smooth model
        Apoly = result['x'][1]*k + result['x'][2] + result['x'][3]/k + result['x'][4]/k**2 + result['x'][5]/k**3 

        return result['x'][0]*pk_EH98+Apoly

    # Compute the Eisenstein and Hu dewiggled transfer function
    def EH98_dewiggled(self, k):

        # Fitting parameters
        a1 = 0.328
        a2 = 431.0
        a3 = 0.380
        a4 = 22.30
        g1 = 0.43
        g2 = 4.0
        c1 = 14.2
        c2 = 731.0
        c3 = 62.5
        l1 = 2.0
        l2 = 1.8
        t1 = 2.0
        theta = 2.725 / 2.7     # Normalised CMB temperature

        q0 = k*theta*theta
        alpha = 1.0 - a1*math.log(a2*self.omega_m*self.hubble*self.hubble)*(self.omega_b/self.omega_m) + a3*math.log(a4*self.omega_m*self.hubble*self.hubble)*(self.omega_b/self.omega_m)**2
        gamma_p1 = (1.0 - alpha)/(1.0 + (g1*k*self.r_s*self.hubble)**g2)
        gamma = self.omega_m*self.hubble*(alpha + gamma_p1)
        q = q0/gamma
        c = c1 + c2/(1.0 + c3*q)
        l = np.log(l1*math.exp(1.0) + l2*q)
        t = l/(l + c*q**t1)

        return t

    def __EH98_lnlike(self, params, k, pkEH, pkdata): 

        pk_B, pk_a1, pk_a2, pk_a3, pk_a4, pk_a5 = params

        Apoly = pk_a1*k + pk_a2 + pk_a3/k + pk_a4/k**2 + pk_a5/k**3 
        pkfit = pk_B*pkEH+Apoly

        # Compute the chi_squared
        dk = np.concatenate([[k[0]],k[1:]-k[0:-1]])

        chi_squared = np.sum(((pkdata-pkfit)/pkdata)**2)

        return chi_squared

# This class uses the Eisenstein and Hu method and generates a CAMB matter power spectrum. We take a number of different parameters including dark energy and neutrinos
# but fix everything else currently (i.e., there are more CAMB parameters we could change if we wanted)
class EH98CAMB(object):

    def __init__(self, redshift=0.0, omega_m=0.3121, omega_b=0.0491, hubble=0.6751, omega_k=0.0, tau=0.063, As=2.130e-9, ns=0.9653,
                neutrino_hierarchy='degenerate', num_massive_neutrinos=1, mnu=0.06, nnu=3.046, w=-1.0, sound_speed=1.0, dark_energy_model='fluid', sigma8=None, r_s=None, r_s_type='CAMB', verbose=False):

        import camb

        self.verbose = verbose

        self.redshift = redshift
        self.omega_m = omega_m
        self.omega_b = omega_b
        self.hubble = hubble
        self.ns = ns
        self.sigma8 = sigma8

        if (self.verbose):
            print "Using CAMB to generate matter power spectrum"

        params = camb.set_params(omch2=(omega_m-omega_b)*hubble**2, ombh2=omega_b*hubble**2, H0=100.0*hubble, omk=omega_k, tau=tau, As=As, ns=ns,
                                 neutrino_hierarchy=neutrino_hierarchy, num_massive_neutrinos=num_massive_neutrinos, mnu=mnu, nnu=nnu, 
                                 w=w, sound_speed=sound_speed, dark_energy_model=dark_energy_model)
        params.set_matter_power(redshifts=[redshift],kmax=100,accurate_massive_neutrino_transfers=True)
        results = camb.get_results(params)
        self.k, z, self.pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=100, npoints = 2000)
        self.pk = self.pk[0]

        # Normalise the input power spectrum to the current value of sigma8 if one has been passed
        if (self.sigma8 is None):
            self.sigma8 = results.get_sigma8()[0]
        else:
            pknorm = norm_pk(self.k, self.pk, self.sigma8)
            self.pk *= pknorm

        if (r_s is None):
            if (r_s_type == 'EH98'):
                self.r_s = EH98_rs(self)
                if (self.verbose):
                    print "Setting sound horizon using EH98 formulae: r_s = ", self.r_s
            elif (r_s_type == 'CAMB'):
                self.r_s = camb.get_background(params).get_derived_params()['rdrag']
                if (self.verbose):
                    print "Setting sound horizon using CAMB value: r_s = ", self.r_s
            else:
                print "r_s_type not supported (", r_s_type, "), must be either 'EH98' or 'CAMB', or you must pass in a value for r_s"
                exit()
        else:
            self.r_s = r_s
            if (self.verbose):
                print "Setting sound horizon using user-defined value: r_s = ", self.r_s            

        self.set_pksmooth()

    # These routine computes the smoothed power spectrum and assign it to self.pksmooth
    def set_pksmooth(self, k=None, pk=None):

        if (k is None):
            if (pk is not None):
                print "When setting pk smooth using kwargs, both k and pk must be supplied. You have only supplied pk"
                exit()
            k = self.k
            pk = self.pk
        else:
            if (pk is None):
                print "When setting pk smooth using kwargs, both k and pk must be supplied. You have only supplied k"
                exit()
            k = k
            pk = pk

        self.pksmooth = self.compute_pksmooth(k, pk)

        return

    # Compute the smooth dewiggled power spectrum using either a polynomial based method (i.e., Hinton 2017), or the EH98 fitting formulae
    def compute_pksmooth(self, k, pk):

        if (self.verbose):
            print "Computing smooth power spectrum using EH98 fitting formulae:  "

        # First compute the normalised Eisenstein and Hu smooth power spectrum
        pk_EH98 = k**self.ns*self.EH98_dewiggled(k)**2
        pk_EH98_spline = interpolate.splrep(k, pk_EH98)
        pk_EH98_norm = math.sqrt(integrate.quad(sigma8_integrand,k[0],k[-1],args=(k[0],k[-1],pk_EH98_spline))[0]/(2.0*math.pi*math.pi))
        pk_EH98 *= (self.sigma8/pk_EH98_norm)**2

        if (self.verbose):
            print "         Fitting input power spectrum to obtain broadband terms"
        nll = lambda *args: self.__EH98_lnlike(*args)
        start = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = optimize.minimize(nll, start, args=(k, pk_EH98, pk), method="Nelder-Mead", tol=1.0e-9, options={'maxiter': 1000000})
        if (self.verbose):
            print "         [B, a1, a2, a3, a4, a5] = ", result['x']

        # Then compute the smooth model
        Apoly = result['x'][1]*k + result['x'][2] + result['x'][3]/k + result['x'][4]/k**2 + result['x'][5]/k**3 

        return result['x'][0]*pk_EH98+Apoly

    # Compute the Eisenstein and Hu dewiggled transfer function
    def EH98_dewiggled(self, k):

        # Fitting parameters
        a1 = 0.328
        a2 = 431.0
        a3 = 0.380
        a4 = 22.30
        g1 = 0.43
        g2 = 4.0
        c1 = 14.2
        c2 = 731.0
        c3 = 62.5
        l1 = 2.0
        l2 = 1.8
        t1 = 2.0
        theta = 2.725 / 2.7     # Normalised CMB temperature

        q0 = k*theta*theta
        alpha = 1.0 - a1*math.log(a2*self.omega_m*self.hubble*self.hubble)*(self.omega_b/self.omega_m) + a3*math.log(a4*self.omega_m*self.hubble*self.hubble)*(self.omega_b/self.omega_m)**2
        gamma_p1 = (1.0 - alpha)/(1.0 + (g1*k*self.r_s*self.hubble)**g2)
        gamma = self.omega_m*self.hubble*(alpha + gamma_p1)
        q = q0/gamma
        c = c1 + c2/(1.0 + c3*q)
        l = np.log(l1*math.exp(1.0) + l2*q)
        t = l/(l + c*q**t1)

        return t

    def __EH98_lnlike(self, params, k, pkEH, pkdata): 

        pk_B, pk_a1, pk_a2, pk_a3, pk_a4, pk_a5 = params

        Apoly = pk_a1*k + pk_a2 + pk_a3/k + pk_a4/k**2 + pk_a5/k**3 
        pkfit = pk_B*pkEH+Apoly

        # Compute the chi_squared
        dk = np.concatenate([[k[0]],k[1:]-k[0:-1]])

        chi_squared = np.sum(((pkdata-pkfit)/pkdata)**2)

        return chi_squared

# This class uses the Hinton2017 method and a tabulated input power spectrum. This is great as it is much more cosmology indepndent that EH98, and fits the input power with a smooth polynomial to
# extract the wiggles. Different choices for the degree of polynomial and weights for fitting can be passed using the degree, sigma and weight parameters
class Hinton2017Tabulated(object):

    def __init__(self, matterfile, omega_m=0.3121, omega_b=0.0491, hubble=0.6751, degree=13, sigma=1, weight=0.5, sigma8=None, r_s=None, verbose=False):

        self.verbose = verbose
        self.omega_m = omega_m
        self.omega_b = omega_b
        self.hubble = hubble
        self.sigma8 = sigma8
        self.degree = degree
        self.sigma = sigma
        self.weight = weight

        if (r_s is None):
            self.r_s = EH98_rs()
            if (self.verbose):
                print "Setting sound horizon using EH98 formulae: r_s = ", self.r_s
                print "Note: This is only used for BAOExtractor model"
        else:
            self.r_s = r_s
            if (self.verbose):
                print "Setting sound horizon using user-defined value: r_s = ", self.r_s
                print "Note: This is only used for BAOExtractor model"

        read_matterpower(self, matterfile)
        self.set_pksmooth()

    # These routine computes the smoothed power spectrum and assign it to self.pksmooth
    def set_pksmooth(self, k=None, pk=None):

        if (k is None):
            if (pk is not None):
                print "When setting pk smooth using kwargs, both k and pk must be supplied. You have only supplied pk"
                exit()
            k = self.k
            pk = self.pk
        else:
            if (pk is None):
                print "When setting pk smooth using kwargs, both k and pk must be supplied. You have only supplied k"
                exit()
            k = k
            pk = pk

        self.pksmooth = self.compute_pksmooth(k, pk)

        return

    # Compute the smooth dewiggled power spectrum using either a polynomial based method (i.e., Hinton 2017), or the EH98 fitting formulae
    def compute_pksmooth(self, k, pk):

        if (self.verbose):
            print "Computing smooth power spectrum using Hinton2017 polyfit method: "

        logk = np.log(k)
        logpk = np.log(pk)
        gauss = np.exp(-0.5*((logk - logk[-1])/self.sigma)**2)   
        w = np.ones(len(logpk)) - self.weight * gauss
        z = np.polyfit(logk, logpk, self.degree, w=w)
        p = np.poly1d(z)
        polyval = p(logk)

        return np.exp(polyval)

# This class uses the Hinton 2017 method and generates a CAMB matter power spectrum. We take a number of different parameters including dark energy and neutrinos
# but fix everything else currently (i.e., there are more CAMB parameters we could change if we wanted)
class Hinton2017CAMB(object):

    def __init__(self, redshift=0.0, omega_m=0.3121, omega_b=0.0491, hubble=0.6751, omega_k=0.0, tau=0.063, As=2.130e-9, ns=0.9653,
                neutrino_hierarchy='degenerate', num_massive_neutrinos=1, mnu=0.06, nnu=3.046, w=-1.0, sound_speed=1.0, dark_energy_model='fluid', sigma8=None, r_s=None, r_s_type='CAMB',
                degree=13, sigma=1, weight=0.5, verbose=False):

        import camb

        self.verbose = verbose

        self.redshift = redshift
        self.omega_m = omega_m
        self.omega_b = omega_b
        self.hubble = hubble
        self.ns = ns
        self.sigma8 = sigma8
        self.degree = degree
        self.sigma = sigma
        self.weight = weight

        if (self.verbose):
            print "Using CAMB to generate matter power spectrum"

        params = camb.set_params(omch2=(omega_m-omega_b)*hubble**2, ombh2=omega_b*hubble**2, H0=100.0*hubble, omk=omega_k, tau=tau, As=As, ns=ns,
                                 neutrino_hierarchy=neutrino_hierarchy, num_massive_neutrinos=num_massive_neutrinos, mnu=mnu, nnu=nnu, 
                                 w=w, sound_speed=sound_speed, dark_energy_model=dark_energy_model)
        params.set_matter_power(redshifts=[redshift],kmax=100,accurate_massive_neutrino_transfers=True)
        results = camb.get_results(params)
        self.k, z, self.pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=100, npoints = 2000)
        self.pk = self.pk[0]

        # Normalise the input power spectrum to the current value of sigma8 if one has been passed
        if (self.sigma8 is None):
            self.sigma8 = results.get_sigma8()[0]
        else:
            pknorm = norm_pk(self.k, self.pk, self.sigma8)
            self.pk *= pknorm

        if (r_s is None):
            if (r_s_type == 'EH98'):
                self.r_s = self.EH98_rs()
                if (self.verbose):
                    print "Setting sound horizon using EH98 formulae: r_s = ", self.r_s
                    print "Note: This is only used for BAOExtractor model"
            elif (r_s_type == 'CAMB'):
                self.r_s = camb.get_background(params).get_derived_params()['rdrag']
                if (self.verbose):
                    print "Setting sound horizon using CAMB value: r_s = ", self.r_s
                    print "Note: This is only used for BAOExtractor model"
            else:
                print "r_s_type not supported (", r_s_type, "), must be either 'EH98' or 'CAMB', or you must pass in a value for r_s"
                exit()
        else:
            self.r_s = r_s
            if (self.verbose):
                print "Setting sound horizon using user-defined value: r_s = ", self.r_s     
                print "Note: This is only used for BAOExtractor model"

        self.set_pksmooth()

    # These routine computes the smoothed power spectrum and assign it to self.pksmooth
    def set_pksmooth(self, k=None, pk=None):

        if (k is None):
            if (pk is not None):
                print "When setting pk smooth using kwargs, both k and pk must be supplied. You have only supplied pk"
                exit()
            k = self.k
            pk = self.pk
        else:
            if (pk is None):
                print "When setting pk smooth using kwargs, both k and pk must be supplied. You have only supplied k"
                exit()
            k = k
            pk = pk

        self.pksmooth = self.compute_pksmooth(k, pk)

        return

    # Compute the smooth dewiggled power spectrum using either a polynomial based method (i.e., Hinton 2017), or the EH98 fitting formulae
    def compute_pksmooth(self, k, pk):

        if (self.verbose):
            print "Computing smooth power spectrum using Hinton2017 polyfit method: "

        logk = np.log(k)
        logpk = np.log(pk)
        gauss = np.exp(-0.5*((logk - logk[-1])/self.sigma)**2)   
        w = np.ones(len(logpk)) - self.weight * gauss
        z = np.polyfit(logk, logpk, self.degree, w=w)
        p = np.poly1d(z)
        polyval = p(logk)

        return np.exp(polyval)

# ******************************** #
# Global Functions for all classes #
# ******************************** #

# Read in a linear power spectrum file from CAMB
def read_matterpower(self, matterfile):

    if (self.verbose):
        print "Reading input matter power spectrum file: ", matterfile

    k_file=[]
    pk_file=[]
    infile = open(matterfile,'r')
    for line in infile:
        ln = line.split()
        if (ln[0] == "#"):
            continue
        k_file.append(float(ln[0]))
        pk_file.append(float(ln[1]))
    infile.close()
    self.k = np.array(k_file)
    self.pk = np.array(pk_file)

    # Normalise the input power spectrum to the current value of sigma8 is one has been passed
    if (self.sigma8 is None):
        self.sigma8 = calc_sigma8(self.k, self.pk)
    else:
        pknorm = norm_pk(self.k, self.pk, self.sigma8)
        self.pk *= pknorm

    return

# Normalise the power spectrum to a given sigma8 value
def norm_pk(k, pk, sigma8new):

    sigma8old = calc_sigma8(k,pk)

    if (self.verbose):
        print "Normalising power spectrum to new sigma8 value: ", sigma8old, " --> ", sigma8new

    return (sigma8new/sigma8old)**2

# Compute the value of sigma8 given linear power spectrum
def calc_sigma8(k, pk):

    pkspline = interpolate.splrep(k, pk)
    return math.sqrt(integrate.quad(sigma8_integrand,k[0],k[-1],args=(k[0],k[-1],pkspline), limit=500)[0]/(2.0*math.pi**2))

def sigma8_integrand(k, kmin, kmax, pkspline):
    if ((k < kmin) or (k > kmax)):
        pk = 0.0
    else:
        pk = interpolate.splev(k, pkspline, der=0)
    window = 3.0*((math.sin(8.0*k)/(8.0*k)**3)-(math.cos(8.0*k)/(8.0*k)**2))
    return k*k*window*window*pk

# Compute the Eisenstein and Hu 1998 value for the sound horizon
def EH98_rs(powerspectrum):

    # Fitting parameters
    b1 = 0.313
    b2 = -0.419
    b3 = 0.607
    b4 = 0.674
    b5 = 0.238
    b6 = 0.223
    a1 = 1291.0
    a2 = 0.251
    a3 = 0.659
    a4 = 0.828
    theta = 2.725 / 2.7     # Normalised CMB temperature

    obh2 = powerspectrum.omega_b*powerspectrum.hubble*powerspectrum.hubble
    omh2 = powerspectrum.omega_m*powerspectrum.hubble*powerspectrum.hubble

    z_eq = 2.5e4*omh2/(theta**4)
    k_eq = 7.46e-2*omh2/(theta**2)

    zd1 = b1*omh2**b2*(1.0 + b3*omh2**b4)
    zd2 = b5*omh2**b6
    z_d = a1*(omh2**a2/(1.0+a3*omh2**a4))*(1.0 + zd1*obh2**zd2)

    R_eq = 3.15e4*obh2/(z_eq*theta**4)
    R_d = 3.15e4*obh2/(z_d*theta**4)

    s = 2.0/(3.0*k_eq)*math.sqrt(6.0/R_eq)*math.log((math.sqrt(1.0+R_d) + math.sqrt(R_d+R_eq))/(1.0 + math.sqrt(R_eq)))

    return s
