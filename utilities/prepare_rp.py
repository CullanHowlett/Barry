# A set of routines Cullan found useful for power spectra
#   Read in a file and calculate new k-binning and k-limits
#   Compute the average and covariance over a set of mocks
#   Convert p(k) measurements to logarithmic values as these are more Gaussian 

import math
import numpy as np

# A routine to read in some power spectrum file and compress the k-bins and apply kbin cuts. The compression is done
# by averaging over n_subsamp bins using the number of modes in each bin as a weight. The k value is the middle value (not weighted)
def read_pk_file(infile, k_min, k_max, n_subsamp):

    k_file=[]
    pk_file=[]
    nk_file=[]
    counter=0
    ktmp=0.0
    pktmp=0.0
    nktmp=0.0
    infile = open(infile,'r')
    for line in infile:
        ln = line.split()
        if (ln[0] == '#'):
            continue
        ktmp += float(ln[1])
        pktmp += float(ln[3])*float(ln[2])
        nktmp += float(ln[3])
        counter += 1
        if (counter == n_subsamp):
            if (ktmp/counter < k_min):
                ktmp=0.0
                pktmp=0.0
                nktmp=0.0
                counter=0
                continue
            if (ktmp/counter > k_max):
                break
            k_file.append(ktmp/counter)
            if (nktmp > 0):
                pk_file.append(pktmp/nktmp)
                nk_file.append(nktmp)
            else:
                pk_file.append(0.0)
                nk_file.append(0)
            ktmp=0.0
            pktmp=0.0
            nktmp=0.0
            counter=0
    infile.close()
    k_file = np.array(k_file)
    pk_file = np.array(pk_file)
    nk_file = np.array(nk_file)

    return k_file, pk_file, nk_file

def main():

    n = 0               # The value of n to use in computing the BAO extract (both Noda2017 and Nishimichi2018 use n=0)
    Delta = 0.5         # The value of Delta (which determines the width of the moving average) to use in computing the BAO extract
    r_s = 147.17        # The value of the sound horizon to use in computing the BAO extract
    nmocks = 1000       # The number of mocks (assumes consecutive numbering)
    kfit_min = 0.02     # The minimum k-value to output
    kfit_max = 0.30      # The maximum k-value to output
    n_subsamp = [1,2,3,5,6,10]       # The number of bins to compress (1 means no compression)
    mockfile = '/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1'      # The first part of the file name

    kwidth = 2.0*math.pi*Delta/r_s
    for sub in n_subsamp:

        # Read in the first power spectrum to work out how many bins we have
        strin = str('%s_R%d.lpow_blake' % (mockfile, 19000))
        k, pk, nk = read_pk_file(strin, 0.0, 1.0, sub)
        kindex = np.where(np.logical_and(k >= kfit_min, k<= kfit_max))[0]

        mock_all = np.zeros((nmocks,len(kindex)))
        nk_mock_all = np.zeros((nmocks,len(kindex)))
        for i in range(nmocks):

            strin = str('%s_R%d.lpow_blake' % (mockfile, i+19000))
            k, pk, nk = read_pk_file(strin, 0.0, 1.0, sub)

            BAOextract = np.zeros(len(k))
            for j in range(len(k)):
                if (pk[j] > 0):
                    index = np.where(np.fabs(k-k[j]) <= kwidth)[0]
                    numer = np.sum((k[index]-k[j])**(2.0*n)*(1.0 - pk[index]/pk[j]))
                    denom = np.sum((k[index]-k[j])**(2.0*n)*(1.0 - np.cos(r_s*(k[index]-k[j]))))
                    if (denom > 0):
                        BAOextract[j] = numer/denom

            mock_all[i] = BAOextract[kindex]
            nk_mock_all[i] = nk[kindex]

        mockave = np.sum(mock_all,axis=0)/nmocks
        nk_mockave = np.sum(nk_mock_all,axis=0)/nmocks
        print nk_mockave
        mockcov = np.zeros((len(kindex),len(kindex)))
        for i in range(len(kindex)):
            mockcov[i,0:] = (np.sum(mock_all[0:,i,None]*mock_all,axis=0) - nmocks*mockave[i]*mockave[0:])/(nmocks-1.0)



        strout = str('%s.rp_1_0p5_%d_0p02-0p30_ave' % (mockfile, sub))
        np.savetxt(strout, np.c_[np.arange(len(k[kindex])),k[kindex],mockave,nk_mockave], fmt='%12d %g %g %12d', delimiter='  ', header="i          k          r_p(k)         nk")

        strout = str('%s.rp_1_0p5_%d_0p02-0p30_cov' % (mockfile, sub))
        np.savetxt(strout, mockcov, fmt='%g', delimiter='  ')

    return 
    
if __name__ == "__main__":
    main()


