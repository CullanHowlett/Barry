# Classes associated within reading in the data and associated products. The 3 classes deal with correlation function, power spectrum or BAO extractor data,
# File formats change with the wind, so these may need changing to.

import math
import numpy as np
from scipy.linalg import lapack

class CorrelationFunction(object):

    def __init__(self, x=None, y=None, covmat=None, nmocks=None, verbose=False):

        self.x = x
        self.y = y
        self.cov = covmat
        self.nmocks = nmocks
        self.verbose = verbose

        compute_cov_inv(self)

        return

    def read_data(self, datafile=None, covfile=None, xmin=-1.0e30, xmax=1.0e30):

        if (datafile is not None):
            self.__read_xi(datafile, xmin, xmax)
        if (covfile is not None):
            read_cov(self, covfile)

        return self

    def __read_xi(self, datafile, xmin, xmax):
        self.x, self.y = self.__xi_read(datafile, xmin, xmax)
        return self

    def __xi_read(self, datafile, xmin, xmax):

        if (self.verbose):
            print "Reading correlation function file: ", datafile

        r_file = []
        xi_file = []
        infile = open(datafile,'r')
        for line in infile:
            ln = line.split()
            if (ln[0] == "#"):
                continue
            if ((float(ln[0]) < xmin) or (float(ln[0]) > xmax)):
                continue
            r_file.append(float(ln[0]))
            xi_file.append(float(ln[1]))
        infile.close()
        
        return np.array(r_file), np.array(xi_file)

class PowerSpectrum(object):

    def __init__(self, x=None, y=None, covmat=None, kwin=None, pkwin=None, kwinindex=None, kwinmatin=None, kwinmatout=None, winmat=None, kwinmatoutindex=None, nmocks=None, verbose=False):

        self.x = x
        self.y = y
        self.cov = covmat
        self.kwin = kwin
        self.pkwin = pkwin
        self.kwinindex = kwinindex
        self.winmat = winmat
        self.kwinmatin = kwinmatin
        self.kwinmatout = kwinmatout
        self.kwinmatoutindex = kwinmatoutindex
        self.nmocks = nmocks
        self.verbose = verbose

        self.comparek = [0, 0, 0]

        compute_cov_inv(self)

        return

    def read_data(self, datafile=None, covfile=None, xmin=-1.0e30, xmax=1.0e30, nconcat=1, winfile=None, winmatfile=None):

        if (datafile is not None):
            self.__read_pk(datafile, xmin, xmax, nconcat)
        if (winfile is not None):
            self.__read_pkwin(winfile, xmin, xmax, nconcat)       # xmin and xmax are not actually applied to the window power here, we instead apply the cuts just before computing the likelihood
        if (winmatfile is not None):
            self.__read_winmat(winmatfile, xmin, xmax)
        if (covfile is not None):
            read_cov(self, covfile)

        if ((self.comparek[0] == 0) and (self.x is not None) and (self.kwin is not None)):
            print "Warning: have not compared k-binning between data and window function power"
            print "kdata: ", self.x
            print "kwin: ", self.kwin[self.kwinindex] 
        if ((self.comparek[1] == 0) and (self.x is not None) and (self.kwinmatout is not None)):
            print "Warning: have not compared k-binning between data and window convolution matrix"
            print "kdata: ", self.x
            print "kwinmat: ", self.kwinmatout[self.kwinmatoutindex]
        if ((self.comparek[2] == 0) and (self.kwin is not None) and (self.kwinmatout is not None)):
            print "Warning: have not compared k-binning between window function power and window convolution matrix"
            print "kwin: ", self.kwin
            print "kwinmat: ", self.kwinmatout 

        return self

    def __read_pk(self, datafile, xmin, xmax, nconcat):
        self.x, self.y = self.__pk_read(datafile, xmin, xmax, nconcat)
        if (self.pkwin is not None):
            if (len(self.x) != len(self.kwin[self.kwinindex])):
                print "length of data (", len(self.x), ") not equal to length of window function power (", len(self.kwin[self.kwinindex]), ")"
                exit()       
            for i in range(len(self.x)):
                if (np.fabs(self.x[i]-self.kwin[self.kwinindex[i]]) > 1.0e-6):
                    print "k-values of data (", i, self.x[i], ") not equal to k-values of window function power (", self.kwinindex[i], self.kwin[self.kwinindex[i]], ")"
                    exit()
            self.comparek[0] = 1
        if (self.kwinmatout is not None):
            if (len(self.x) != len(self.kwinmatout[self.kwinmatoutindex])):
                print "length of data (", len(self.x), ") not equal to length of window convolution matrix (", len(self.kwinmatout[self.kwinmatoutindex]), ")"
                exit()  
            for i in range(len(self.x)):
                if (np.fabs(self.x[i]-self.kwinmatout[self.kwinmatoutindex[i]]) > 1.0e-6):
                    print "k-values of data (", i, self.x[i], ") not equal to k-values of window convolution matrix (", self.kwinmatoutindex[i], self.kwinmatout[self.kwinmatoutindex[i]], ")"
                    exit()
            self.comparek[1] = 1    
        return self

    def __read_pkwin(self, winfile, xmin, xmax, nconcat):
        comparepkwinflag = 0
        comparewinmatflag = 0
        self.kwin, self.pkwin = self.__pk_read(winfile, 0.0, 1.0e30, nconcat)
        self.kwinindex = np.where(np.logical_and(self.kwin>=xmin,self.kwin<=xmax))[0]
        if (self.x is not None):
            if (len(self.x) != len(self.kwin[self.kwinindex])):
                print "length of data (", len(self.x), ") not equal to length of window function power (", len(self.kwin[self.kwinindex]), ")"
                exit()   
            for i in range(len(self.kwin[self.kwinindex])):
                if (np.fabs(self.x[i]-self.kwin[self.kwinindex[i]]) > 1.0e-6):
                    print "k-values of data (", i, self.x[i], ") not equal to k-values of window function power (", self.kwinindex[i], self.kwin[self.kwinindex[i]], ")"
                    exit()
            self.comparek[0] = 1
        if (self.kwinmatout is not None):
            if (len(self.kwin) != len(self.kwinmatout)):
                print "length of window function power (", len(self.kwin), ") not equal to length of window convolution matrix (", len(self.kwinmatout), ")"
                exit() 
            for i in range(len(self.kwin)):
                if (np.fabs(self.kwin[i]-self.kwinmatout[i]) > 1.0e-6):
                    print "k-values of window function power (", i, self.kwin[i], ") not equal to k-values of window convolution matrix (", i, self.kwinmatout[i], ")"
                    exit()    
            self.comparek[2] = 1  

        return self

    def __read_winmat(self, winmatfile, xmin, xmax):
        self.kwinmatin, self.kwinmatout, self.winmat = self.__winmat_read(winmatfile)
        self.kwinmatoutindex = np.where(np.logical_and(self.kwinmatout>=xmin,self.kwinmatout<=xmax))[0]
        if (self.x is not None):
            if (len(self.x) != len(self.kwinmatout[self.kwinmatoutindex])):
                print "length of data (", len(self.x), ") not equal to length of window convolution matrix (", len(self.kwinmatout[self.kwinmatoutindex]), ")"
                exit()  
            for i in range(len(self.kwinmatout[self.kwinmatoutindex])):
                if (np.fabs(self.x[i]-self.kwinmatout[self.kwinmatoutindex[i]]) > 1.0e-6):
                    print "k-values of data (", i, self.x[i], ") not equal to k-values of window convolution matrix (", self.kwinmatoutindex[i], self.kwinmatout[self.kwinmatoutindex[i]], ")"
                    exit()
            self.comparek[1] = 1    
        if (self.kwin is not None):
            if (len(self.kwin) != len(self.kwinmatout)):
                print "length of window function power (", len(self.kwin), ") not equal to length of window convolution matrix (", len(self.kwinmatout), ")"
                exit() 
            for i in range(len(self.kwinmatout)):
                if (np.fabs(self.kwin[i]-self.kwinmatout[i]) > 1.0e-6):
                    print "k-values of window function power (", i, self.kwin[i], ") not equal to k-values of window convolution matrix (", i, self.kwinmatout[i], ")"
                    exit()    
            self.comparek[2] = 1  
        return self

    def __pk_read(self, datafile, xmin, xmax, nconcat):

        if (self.verbose):
            print "Reading power spectrum file: ", datafile

        k_file=[]
        pk_file=[]
        counter=0
        ktmp=0.0
        pktmp=0.0
        nktmp=0.0
        infile = open(datafile,'r')
        for line in infile:
            ln = line.split()
            if (ln[0] == '#'):
                continue
            ktmp += float(ln[1])
            pktmp += float(ln[3])*float(ln[2])
            nktmp += float(ln[3])
            counter += 1
            if (counter == nconcat):
                if (ktmp/counter < xmin):
                    ktmp=0.0
                    pktmp=0.0
                    nktmp=0.0
                    counter=0
                    continue
                if (ktmp/counter > xmax):
                    break
                k_file.append(ktmp/counter)
                if (nktmp > 0):
                    pk_file.append(pktmp/nktmp)
                else:
                    pk_file.append(0.0)
                ktmp=0.0
                pktmp=0.0
                nktmp=0.0
                counter=0
        infile.close()

        if (counter > 0):
            if ((ktmp/counter >= xmin) and (ktmp/counter <= xmax)):
                k_file.append(ktmp/counter)
                if (nktmp > 0):
                    pk_file.append(pktmp/nktmp)
                else:
                    pk_file.append(0.0)

        return np.array(k_file), np.array(pk_file)

    def __winmat_read(self, winmatfile):

        if (self.verbose):
            print "Reading window function convolution matrix: ", winmatfile

        infile = open(winmatfile,'r')
        infile.readline()
        nkcol = int(infile.readline())
        kcol = np.zeros(nkcol)
        line = infile.readline()
        ln = line.split()
        for i in range(nkcol):
            kcol[i] = float(ln[i+1])
        nkrow = int(infile.readline())
        krow = np.zeros(nkrow)
        winmat = np.zeros((nkcol+1,nkrow))
        for i in range(nkrow):
            line = infile.readline()
            ln = line.split()
            krow[i] = float(ln[0])
            winmat[0][i] = float(ln[1])
            for j in range(nkcol):
                winmat[j+1][i] = float(ln[j+2])

        return krow, kcol, winmat

# The BAOExtract data class inherits all properties of the power spectrum, with an extra routine that converts measurements of the power spectrum to
# a BAO extract. The idea is that you can read in a power spectrum file, and then convert to a BAO extract using this code
class BAOExtract(PowerSpectrum):

    def __init__(self, x=None, y=None, covmat=None, kwin=None, pkwin=None, kwinindex=None, kwinmatin=None, kwinmatout=None, winmat=None, kwinmatoutindex=None, nmocks=None, verbose=False):

        PowerSpectrum.__init__(self, x=x, y=y, covmat=covmat, kwin=kwin, pkwin=pkwin, kwinindex=kwinindex, kwinmatin=kwinmatin, kwinmatout=kwinmatout, winmat=winmat, kwinmatoutindex=kwinmatoutindex, nmocks=nmocks, verbose=verbose)

        return

    def extract_BAO(self, r_s, n=0, Delta=0.5):

        kwidth = 2.0*math.pi*Delta/r_s

        BAOextract = np.empty(len(self.x))
        for i in range(len(self.x)):
            kindex = np.where(np.fabs(self.x-self.x[i]) <= kwidth)[0]
            numer = np.sum((self.x[kindex]-self.x[i])**(2.0*n)*(1.0 - self.y[kindex]/self.y[i]))
            denom = np.sum((self.x[kindex]-self.x[i])**(2.0*n)*(1.0 - np.cos(r_s*(self.x[kindex]-self.x[i]))))
            BAOextract[i] = numer/denom
        self.y = BAOextract

        return self

def read_cov(data, covfile):

    if (data.verbose):
        print "Calculating covariance matrix: ", covfile

    infile = open(covfile,'r')
    cov = np.loadtxt(infile, unpack=True)
    data.cov = cov

    compute_cov_inv(data)

    return data

def compute_cov_inv(data):

    if (data.cov is None):

        data.cov_det = None
        data.cov_inv = None

    else:

        # Compute the log determinant of the covariance matrix
        cov_copy, pivots, info = lapack.dgetrf(data.cov)
        abs_element = np.fabs(np.diagonal(cov_copy))
        data.cov_det = np.sum(np.log(abs_element))

        # Invert the covariance matrix
        identity = np.eye(len(data.x))
        cov_lu, pivots, cov_inv, info = lapack.dgesv(data.cov, identity)
        data.cov_inv = cov_inv

    return data