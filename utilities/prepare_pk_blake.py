# Routines for reading in a reoutputting Chris Blake's power spectra in a format that Cullan prefers

import numpy as np

def main():

    strin = "/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/pkcatmock_colamockmean_1year_dk0pt002.dat"
    strout = "/Volumes/Work/ICRAR/TAIPAN/Mocks/HOD/Mock_v1/Mock_taipan_year1_v1"

    infile = open(strin,'r')
    for i in range(4):
        infile.readline()
    kmin, kmax, nk, nmocks = map(float, infile.readline().split()) 
    nk = int(nk)
    nmocks = int(nmocks)

    k = np.empty(nk)
    nmodes = np.empty(nk).astype(int)
    for i in range(nk):
        ln = infile.readline().split()
        k[i] = float(ln[0])
        nmodes[i] = float(ln[3])

    for i in range(nk):
        for j in range(nk):
            infile.readline()

    for i in range(nmocks):
        pk = np.empty(nk)
        outfile = str("%s_R%d.lpow_blake" % (strout, 19000+i))
        for j in range(nk):
            ln = infile.readline().split()
            pk[j] = float(ln[2])
        np.savetxt(outfile, np.c_[np.arange(nk)+1,k,pk,nmodes], "%d %g %g %g", header="i      k      pk      nk")

    return 
    
if __name__ == "__main__":
    main()


