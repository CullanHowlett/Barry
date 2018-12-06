import numpy as np
from models import *
from read_data import *
from powerspectrum import *
from fitting_routines import *
from plotting_routines import *
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager
from matplotlib import gridspec

# Read in all the chains for the mocks (pre- and post-recon), and plot the maximum likelihood values for alpha, as well as the errors and compute some summary statistics (the mean alpha, the average error and the sample variance)
if __name__ == "__main__":

    dataflag = 1 
    binwidth = 2
    nmocks = 1000
    firstmock = 19000
    xmin = 0.02
    xmax = 0.30
    power = Hinton2017CAMB(redshift=0.11, mnu=0.0)

    alphamean = np.empty((nmocks,2))
    alphaerrlo = np.empty((nmocks,2))
    alphaerrhi = np.empty((nmocks,2))
    alphasig = np.empty((nmocks,2))
    alphasig = np.empty((nmocks,2))
    for mock in range(nmocks):

        mocknum = mock+firstmock

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
                    covfile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.lpow_%d_0p02-0p30_cov' % binwidth)       # The covariance matrix 
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
                    covfile = str('./files/test_files/mock_average/Mock_taipan_year1_v1.lpow_%d_0p02-0p30_cov_recon' % binwidth)       # The covariance matrix 
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
            fitter = List(data, model, alphavals[1], alphavals[1], 1, liketype="SH2016", do_plot=1, startfromprior=False, optimtype="Powell")
            alpha, chi_squared, likelihood, posterior, chi_squarednoBAO, likelihoodnoBAO, posteriornoBAO = fitter.fit_data()

            alphasig[mock,i] = chi_squarednoBAO-chi_squared
            if (alphasig[mock,i] > 0.0):
                alphasig[mock,i] = np.sqrt(alphasig[mock,i])
            else:
                alphasig[mock,i] = -np.sqrt(np.fabs(alphasig[mock,i]))

            print mock, alphamean[mock,i], alphaerrlo[mock,i], alphaerrhi[mock,i], alphasig[mock,i]


# Plot the maximum likelihood and errors on alpha for all the mocks
fig = plt.figure(10)
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1], left=0.13, bottom=0.13, right=0.98, top=0.98)

nbins2 = 50
minalpha = 0.7
maxalpha = 1.3
index = np.where(np.logical_and(alphasig[0:,0] >= 0.0, alphasig[0:,1] >= 0.0))[0]
index2 = np.where(np.logical_or(alphasig[0:,0] < 0.0, alphasig[0:,1] < 0.0))[0]
alphameanave = np.mean(alphamean[index,0:], axis=0)
alphameanstd = np.std(alphamean[index,0:],axis=0)
alphaerr = (alphaerrlo + alphaerrhi)/2.0
alphaerrave = np.mean(alphaerr[index,0:], axis=0)
alphaerrstd = np.std(alphaerr[index,0:], axis=0)

hist1 = np.histogram(alphamean[index,0],bins=nbins2,range=[minalpha,maxalpha],density=True)
hist2 = np.histogram(alphamean[index,1],bins=nbins2,range=[minalpha,maxalpha],density=True)
hist1_2 = np.histogram(alphamean[index2,0],bins=nbins2,range=[minalpha,maxalpha],density=True)
hist2_2 = np.histogram(alphamean[index2,1],bins=nbins2,range=[minalpha,maxalpha],density=True)

print alphameanave, alphameanstd, alphaerrave

ax1=fig.add_subplot(gs[0])
ax1.plot(hist1[1][1:], hist1[0], color='k', linewidth=1.5, ls='steps', zorder=1)
ax1.bar(hist1[1][0:-1], hist1[0], width=hist1[1][1]-hist1[1][0], color='r', zorder=4, alpha=0.4, fill=False)
ax1.plot(hist1_2[1][1:], hist1_2[0], color='k', linewidth=1.5, ls='steps', zorder=1)
ax1.set_xlim(minalpha,maxalpha)
ax1.set_ylim(np.amin(hist1[0]),np.amax(hist1[0]))
for axis in ['top','left','bottom','right']:
    ax1.spines[axis].set_linewidth(1.3)
for tick in ax1.xaxis.get_ticklabels():
    tick.set_fontsize(14)
for tick in ax1.yaxis.get_ticklabels():
    tick.set_fontsize(14)

ax1=fig.add_subplot(gs[2])
ax1.errorbar([minalpha,maxalpha], [minalpha,maxalpha], marker='None',markerfacecolor='r',markeredgecolor='k',color='r',linestyle='--',markeredgewidth=1.3, zorder=4, linewidth=1.3)
ax1.errorbar(alphamean[index,0], alphamean[index,1], marker='o',markerfacecolor='r',markeredgecolor='r',color='r',linestyle='None',markeredgewidth=1.3, zorder=1, markersize=6, alpha=0.4)
ax1.errorbar(alphamean[index2,0], alphamean[index2,1], marker='o',markerfacecolor='k',markeredgecolor='k',color='k',linestyle='None',markeredgewidth=1.3, zorder=1, markersize=6, alpha=0.4)
ax1.errorbar(alphameanave[0], alphameanave[1], xerr=alphameanstd[0]/np.sqrt(nmocks), yerr=alphameanstd[1]/np.sqrt(nmocks), marker='s',markerfacecolor='b',markeredgecolor='k',color='b',linestyle='None',markeredgewidth=1.3, zorder=5, markersize=10)
ax1.axvline(x=1.0, color='k', linestyle='-', linewidth=1.3)
ax1.axhline(y=1.0, color='k', linestyle='-', linewidth=1.3)
ax1.set_xlabel(r'$\alpha_{P_{k}},\mathrm{Pre-Reconstruction}$',fontsize=22)
ax1.set_ylabel(r'$\alpha_{P_{k}},\mathrm{Post-Reconstruction}$',fontsize=22)
ax1.set_xlim(minalpha, maxalpha)
ax1.set_ylim(minalpha, maxalpha)
for axis in ['top','left','bottom','right']:
    ax1.spines[axis].set_linewidth(1.3)
for tick in ax1.xaxis.get_ticklabels():
    tick.set_fontsize(14)
for tick in ax1.yaxis.get_ticklabels():
    tick.set_fontsize(14)
ax1.text(0.40, 0.15, str(r"$\mathrm{Pre-Reconstruction}:\,%6.5lf\pm%6.5lf$" % (alphameanave[0], alphameanstd[0]/np.sqrt(nmocks))), color='k', fontsize=16, transform=ax1.transAxes)
ax1.text(0.40, 0.07, str(r"$\mathrm{Post-Reconstruction}:\,%6.5lf\pm%6.5lf$" % (alphameanave[1], alphameanstd[1]/np.sqrt(nmocks))), color='k', fontsize=16, transform=ax1.transAxes)

ax1=fig.add_subplot(gs[3])
ax1.plot(hist2[0], hist2[1][0:-1], color='k', linewidth=1.5, ls='steps', zorder=1)
ax1.barh(hist2[1][0:-1], hist2[0], height=hist2[1][1]-hist2[1][0], color='r', zorder=4, alpha=0.4, fill=False)
ax1.plot(hist2_2[0], hist2_2[1][0:-1], color='k', linewidth=1.5, ls='steps', zorder=1)
ax1.set_ylim(minalpha,maxalpha)
ax1.set_xlim(np.amin(hist2[0]),np.amax(hist2[0]))
for axis in ['top','left','bottom','right']:
    ax1.spines[axis].set_linewidth(1.3)
for tick in ax1.xaxis.get_ticklabels():
    tick.set_fontsize(14)
for tick in ax1.yaxis.get_ticklabels():
    tick.set_fontsize(14)

fig.subplots_adjust(hspace=0.0, wspace=0, bottom=0.12, top=0.98, right=0.98, left=0.14)
plt.setp([a.get_xticklabels() for a in fig.axes[0:1]], visible=False)
plt.setp([a.get_yticklabels() for a in fig.axes[2:4]], visible=False)

fig = plt.figure(11)
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1], left=0.13, bottom=0.13, right=0.98, top=0.98)

#minalpha = np.amin([np.amin(stdalpha1),np.amin(stdalpha2)])
minalpha = 0.02
maxalpha = np.amax([np.amax(alphaerr[0]),np.amax(alphaerr[1])])
hist1 = np.histogram(alphaerr[index,0],bins=nbins2,range=[minalpha,maxalpha],density=True)
hist2 = np.histogram(alphaerr[index,1],bins=nbins2,range=[minalpha,maxalpha],density=True)
hist1_2 = np.histogram(alphaerr[index2,0],bins=nbins2,range=[minalpha,maxalpha],density=True)
hist2_2 = np.histogram(alphaerr[index2,1],bins=nbins2,range=[minalpha,maxalpha],density=True)

ax1=fig.add_subplot(gs[0])
ax1.plot(hist1[1][1:], hist1[0], color='k', linewidth=1.5, ls='steps', zorder=1)
ax1.bar(hist1[1][0:-1], hist1[0], width=hist1[1][1]-hist1[1][0], color='r', zorder=4, alpha=0.4, fill=False)
ax1.plot(hist1_2[1][1:], hist1_2[0], color='k', linewidth=1.5, ls='steps', zorder=1)
ax1.set_xlim(minalpha,maxalpha)
ax1.set_ylim(np.amin(hist1[0]),np.amax(hist1[0]))
for axis in ['top','left','bottom','right']:
    ax1.spines[axis].set_linewidth(1.3)
for tick in ax1.xaxis.get_ticklabels():
    tick.set_fontsize(14)
for tick in ax1.yaxis.get_ticklabels():
    tick.set_fontsize(14)

ax1=fig.add_subplot(gs[2])
ax1.errorbar([minalpha,maxalpha], [minalpha,maxalpha], marker='None',markerfacecolor='r',markeredgecolor='k',color='r',linestyle='--',markeredgewidth=1.3, zorder=4, linewidth=1.3)
ax1.errorbar(alphaerr[index,0], alphaerr[index,1], marker='o',markerfacecolor='r',markeredgecolor='r',color='r',linestyle='None',markeredgewidth=1.3, zorder=1, markersize=6, alpha=0.4)
ax1.errorbar(alphaerr[index2,0], alphaerr[index2,1], marker='o',markerfacecolor='k',markeredgecolor='k',color='k',linestyle='None',markeredgewidth=1.3, zorder=1, markersize=6, alpha=0.4)
ax1.errorbar(alphaerrave[0], alphaerrave[1], xerr=alphaerrstd[0]/np.sqrt(nmocks), yerr=alphaerrstd[1]/np.sqrt(nmocks), marker='s',markerfacecolor='b',markeredgecolor='k',color='b',linestyle='None',markeredgewidth=1.3, zorder=5, markersize=10)
ax1.axvline(x=0.02, color='k', linestyle='-', linewidth=1.3)
ax1.axhline(y=0.02, color='k', linestyle='-', linewidth=1.3)
ax1.set_xlabel(r'$\sigma_{\alpha}^{P_{k}},\mathrm{Pre-Reconstruction}$',fontsize=22)
ax1.set_ylabel(r'$\sigma_{\alpha}^{P_{k}},\mathrm{Post-Reconstruction}$',fontsize=22)
ax1.set_xlim(minalpha,maxalpha)
ax1.set_ylim(minalpha,maxalpha)
for axis in ['top','left','bottom','right']:
    ax1.spines[axis].set_linewidth(1.3)
for tick in ax1.xaxis.get_ticklabels():
    tick.set_fontsize(14)
for tick in ax1.yaxis.get_ticklabels():
    tick.set_fontsize(14)
ax1.text(0.60, 0.15, str(r"$\mathrm{Pre-Reconstruction}:\,%6.5lf$" % (alphaerrave[0])), color='k', fontsize=16, transform=ax1.transAxes)
ax1.text(0.60, 0.07, str(r"$\mathrm{Post-Reconstruction}:\,%6.5lf$" % (alphaerrave[1])), color='k', fontsize=16, transform=ax1.transAxes)


ax1=fig.add_subplot(gs[3])
ax1.plot(hist2[0], hist2[1][0:-1], color='k', linewidth=1.5, ls='steps', zorder=1)
ax1.barh(hist2[1][0:-1], hist2[0], height=hist2[1][1]-hist2[1][0], color='r', zorder=4, alpha=0.4, fill=False)
ax1.plot(hist2_2[0], hist2_2[1][0:-1], color='k', linewidth=1.5, ls='steps', zorder=1)
ax1.set_ylim(minalpha,maxalpha)
ax1.set_xlim(np.amin(hist2[0]),np.amax(hist2[0]))
for axis in ['top','left','bottom','right']:
    ax1.spines[axis].set_linewidth(1.3)
for tick in ax1.xaxis.get_ticklabels():
    tick.set_fontsize(14)
for tick in ax1.yaxis.get_ticklabels():
    tick.set_fontsize(14)

fig.subplots_adjust(hspace=0.0, wspace=0, bottom=0.12, top=0.98, right=0.98, left=0.14)
plt.setp([a.get_xticklabels() for a in fig.axes[0:1]], visible=False)
plt.setp([a.get_yticklabels() for a in fig.axes[2:4]], visible=False)

fig = plt.figure(12)
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1], left=0.13, bottom=0.13, right=0.98, top=0.98)

minalpha = np.amin([np.amin(alphasig[0:,0]),np.amin(alphasig[0:,1])])
maxalpha = np.amax([np.amax(alphasig[0:,0]),np.amax(alphasig[0:,1])])
hist1 = np.histogram(alphasig[0:,0],bins=nbins2,range=[minalpha,maxalpha],density=True)
hist2 = np.histogram(alphasig[0:,1],bins=nbins2,range=[minalpha,maxalpha],density=True)
alphasigave = np.mean(alphasig[index3,0:], axis=1)
alphasigstd = np.std(alphasig[index3,0:],axis=1)

ax1=fig.add_subplot(gs[0])
ax1.plot(hist1[1][1:], hist1[0], color='k', linewidth=1.5, ls='steps', zorder=1)
ax1.bar(hist1[1][0:-1], hist1[0], width=hist1[1][1]-hist1[1][0], color='r', zorder=4, alpha=0.4, fill=False)
ax1.set_xlim(minalpha,maxalpha)
ax1.set_ylim(np.amin(hist1[0]),np.amax(hist1[0]))
for axis in ['top','left','bottom','right']:
    ax1.spines[axis].set_linewidth(1.3)
for tick in ax1.xaxis.get_ticklabels():
    tick.set_fontsize(14)
for tick in ax1.yaxis.get_ticklabels():
    tick.set_fontsize(14)

ax1=fig.add_subplot(gs[2])
ax1.errorbar([minalpha,maxalpha], [minalpha,maxalpha], marker='None',markerfacecolor='r',markeredgecolor='k',color='r',linestyle='--',markeredgewidth=1.3, zorder=4, linewidth=1.3)
ax1.errorbar(alphasig[0:,0], alphasig[0:,1], marker='o',markerfacecolor='r',markeredgecolor='k',color='r',linestyle='None',markeredgewidth=1.3, zorder=1, markersize=6, alpha=0.4)
ax1.errorbar(alphasigave[0], alphasigave[1], xerr=alphasigstd[0]/np.sqrt(nmocks), yerr=alphasigstd[1]/np.sqrt(nmocks), marker='s',markerfacecolor='b',markeredgecolor='k',color='b',linestyle='None',markeredgewidth=1.3, zorder=5, markersize=10)
ax1.axvline(x=2.0, color='k', linestyle='-', linewidth=1.3)
ax1.axhline(y=2.0, color='k', linestyle='-', linewidth=1.3)
ax1.set_xlabel(r'$\Delta \chi^{2}_{P_{k}},\mathrm{Pre-Reconstruction}$',fontsize=22)
ax1.set_ylabel(r'$\Delta \chi^{2}_{P_{k}},\mathrm{Post-Reconstruction}$',fontsize=22)
ax1.set_xlim(minalpha, maxalpha)
ax1.set_ylim(minalpha, maxalpha)
for axis in ['top','left','bottom','right']:
    ax1.spines[axis].set_linewidth(1.3)
for tick in ax1.xaxis.get_ticklabels():
    tick.set_fontsize(14)
for tick in ax1.yaxis.get_ticklabels():
    tick.set_fontsize(14)
ax1.text(0.40, 0.15, str(r"$\mathrm{Pre-Reconstruction}:\,%6.5lf\pm%6.5lf$" % (alphasigave[0], alphasigstd[0]/np.sqrt(nmocks))), color='k', fontsize=16, transform=ax1.transAxes)
ax1.text(0.40, 0.07, str(r"$\mathrm{Post-Reconstruction}:\,%6.5lf\pm%6.5lf$" % (alphasigave[1], alphasigstd[1]/np.sqrt(nmocks))), color='k', fontsize=16, transform=ax1.transAxes)

ax1=fig.add_subplot(gs[3])
ax1.plot(hist2[0], hist2[1][0:-1], color='k', linewidth=1.5, ls='steps', zorder=1)
ax1.barh(hist2[1][0:-1], hist2[0], height=hist2[1][1]-hist2[1][0], color='r', zorder=4, alpha=0.4, fill=False)
ax1.set_ylim(minalpha,maxalpha)
ax1.set_xlim(np.amin(hist2[0]),np.amax(hist2[0]))
for axis in ['top','left','bottom','right']:
    ax1.spines[axis].set_linewidth(1.3)
for tick in ax1.xaxis.get_ticklabels():
    tick.set_fontsize(14)
for tick in ax1.yaxis.get_ticklabels():
    tick.set_fontsize(14)

fig.subplots_adjust(hspace=0.0, wspace=0, bottom=0.12, top=0.98, right=0.98, left=0.14)
plt.setp([a.get_xticklabels() for a in fig.axes[0:1]], visible=False)
plt.setp([a.get_yticklabels() for a in fig.axes[2:4]], visible=False)

nbins2 = 80
hist_xi = np.histogram(alphasig[0:,0],bins=nbins2,range=(minalpha,maxalpha))
hist_pk = np.histogram(alphasig[0:,1],bins=nbins2,range=(minalpha,maxalpha))
nsig_xi1 = len(np.where(alphasig[0:,0] >= 1.0)[0])
nsig_pk1 = len(np.where(alphasig[0:,1] >= 1.0)[0])
nsig_xi2 = len(np.where(alphasig[0:,0] >= 2.0)[0])
nsig_pk2 = len(np.where(alphasig[0:,1] >= 2.0)[0])
nsig_1 = len(np.where(np.logical_and(alphasig[0:,0] >= 1.0, alphasig[0:,0] >= 1.0))[0])
nsig_2 = len(np.where(np.logical_and(alphasig[0:,1] >= 2.0, alphasig[0:,1] >= 2.0))[0])

fig = plt.figure(1)
ax2=fig.add_axes([0.13,0.13,0.82,0.82])
ax2.plot(hist_xi[1][1:], np.cumsum(hist_xi[0]), color='r', linewidth=1.5, ls='steps', zorder=5)
ax2.plot(hist_pk[1][1:], np.cumsum(hist_pk[0]), color='b', linewidth=1.5, ls='steps', zorder=5)
ax2.axvline(x=1.0, linewidth=1.2, linestyle='-.', color='k')
ax2.axvline(x=2.0, linewidth=1.2, linestyle='-.', color='k')
ax2.set_yscale('log')
ax2.set_xlim(minalpha, maxalpha)
ax2.set_ylim(0.0, nmocks)
ax2.tick_params('both',length=10, which='major')
ax2.tick_params('both',length=5, which='minor')
ax2.set_xlabel(r'$\sqrt{\Delta \chi^{2}}$',fontsize=18)
ax2.set_ylabel(r'$N_{mocks}$',fontsize=18)
for axis in ['top','left','bottom','right']:
    ax2.spines[axis].set_linewidth(1.5)
for tick in ax2.xaxis.get_ticklabels():
    tick.set_fontsize(15)
for tick in ax2.yaxis.get_ticklabels():
    tick.set_fontsize(15)
ax2.yaxis.set_major_formatter(tk.ScalarFormatter())
gridlines = np.linspace(0.0, 3.0, 4, endpoint=True)
for i in range(len(gridlines)):
  ax2.axhline(y=10.0**gridlines[i], color='0.85', linestyle='--', linewidth=1.0, zorder=0)
  ax2.axhline(y=3.0*10.0**gridlines[i], color='0.85', linestyle='--', linewidth=1.0, zorder=0)
ax2.text(0.35, 0.12, str(r"$%d$" % (nmocks-nsig_xi1)), color='r', fontsize=15, transform=ax2.transAxes)
ax2.text(0.35, 0.08, str(r"$%d$" % (nmocks-nsig_pk1)), color='b', fontsize=15, transform=ax2.transAxes)
ax2.text(0.35, 0.04, str(r"$%d$" % (nmocks-nsig_1)), color='k', fontsize=15, transform=ax2.transAxes)
ax2.text(0.48, 0.12, str(r"$%d$" % (nsig_xi1-nsig_xi2)), color='r', fontsize=15, transform=ax2.transAxes)
ax2.text(0.48, 0.08, str(r"$%d$" % (nsig_pk1-nsig_pk2)), color='b', fontsize=15, transform=ax2.transAxes)
ax2.text(0.48, 0.04, str(r"$%d$" % (nsig_1-nsig_2)), color='k', fontsize=15, transform=ax2.transAxes)
ax2.text(0.60, 0.12, str(r"$%d$" % (nsig_xi2)), color='r', fontsize=15, transform=ax2.transAxes)
ax2.text(0.60, 0.08, str(r"$%d$" % (nsig_pk2)), color='b', fontsize=15, transform=ax2.transAxes)
ax2.text(0.60, 0.04, str(r"$%d$" % (nsig_2)), color='k', fontsize=15, transform=ax2.transAxes)

plt.show()
