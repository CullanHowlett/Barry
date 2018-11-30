# A set of routines Cullan found useful for correlation functions
#   Read in a file and calculate new k-binning and k-limits
#   Compute the average and covariance over a set of mocks
#   Convert p(k) measurements to logarithmic values as these are more Gaussian 

import numpy as np

# A routine to read in some correlation function file, and return r and the spherically 
# averaged correlation function removing all bins outside the fitting range
def read_xi_file(infile, r_min, r_max):

    r_file=[]
    xi_file = []
    infile = open(infile,'r')
    for line in infile:
        ln = line.split()
        if (ln[0] == "#"):
            continue
        if ((float(ln[0]) < r_min) or (float(ln[0]) > r_max)):
            continue
        r_file.append(float(ln[0]))
        xi_file.append(float(ln[1]))
    infile.close()
    r_file = np.array(r_file)
    xi_file = np.array(xi_file)

    return r_file, xi_file

def main():

    nmocks = 250       # The number of mocks (assumes consecutive numbering)
    rfit_min = 30.0     # The minimum k-value to output
    rfit_max = 200.0    # The maximum k-value to output
    binwidth = 2        # The binwidth of the data (we can't compress the correlation function here in the  same way as the power spectrum, 
                        # so we have separate files for different binwidths, with the value here as the last part of the filename)
    mockfile = '/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1'      # The first part of the file name

    # Read in the first power spectrum to work out ow many bins we have
    strin = str('%s_R%d.xi_%d_recon' % (mockfile, 19000, binwidth))
    r, xi = read_xi_file(strin, rfit_min, rfit_max)

    mock_all = np.zeros((nmocks,len(r)))
    for i in range(nmocks):

        strin = str('%s_R%d.xi_%d_recon' % (mockfile, i+19000, binwidth))
        r, mock_all[i] = read_xi_file(strin, rfit_min, rfit_max)

    mockave = np.sum(mock_all,axis=0)/nmocks
    mockcov = np.zeros((len(r),len(r)))
    for i in range(len(r)):
        mockcov[i,0:] = (np.sum(mock_all[0:,i,None]*mock_all,axis=0) - nmocks*mockave[i]*mockave[0:])/(nmocks-1.0)

    strout = str('%s.xi_%d_ave_30-200_recon' % (mockfile, binwidth))
    np.savetxt(strout, np.c_[r,mockave], fmt='%12.6g %12.6g', delimiter='  ', header="r           xi")

    strout = str('%s.xi_%d_cov_30-200_recon' % (mockfile, binwidth))
    np.savetxt(strout, mockcov, fmt='%12.6g', delimiter='  ')

    return 
    
if __name__ == "__main__":
    main()


