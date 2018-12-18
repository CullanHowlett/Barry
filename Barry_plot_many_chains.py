import sys
sys.path.append("./src/")
import numpy as np
from models import *
from read_data import *
from powerspectrum import *
from fitting_routines import *
from plotting_routines import *
from chainconsumer import ChainConsumer
import matplotlib.backends.backend_pdf

# Plot the summary statistics (the mean alpha, the average error and the sample variance) for all mocks
if __name__ == "__main__":

    nmocks = 1000
    infile = open("/Volumes/Work/ICRAR/git/Barry/files/test_files/BAOfits/BAO_Mock_Polynomial_SigmaNLprior_taipan_year1_v1_lpow_0p02-0p30_3_summary.dat", 'r')
    outfile = matplotlib.backends.backend_pdf.PdfPages("/Volumes/Work/ICRAR/git/Barry/files/test_files/BAOfits/BAO_Mock_Polynomial_SigmaNLprior_taipan_year1_v1_lpow_0p02-0p30_3_summary.pdf")

    alphamean = np.empty((nmocks,2))
    alphaerrlo = np.empty((nmocks,2))
    alphaerrhi = np.empty((nmocks,2))
    alphasig = np.empty((nmocks,2))
    counter = 0
    for line in infile:
        ln = line.split()
        if (ln[0] == '#'):
            continue
        alphamean[counter,0] = float(ln[1])
        alphaerrlo[counter,0] = float(ln[2])
        alphaerrhi[counter,0] = float(ln[3])
        alphasig[counter,0] = float(ln[4])
        alphamean[counter,1] = float(ln[5])
        alphaerrlo[counter,1] = float(ln[6])
        alphaerrhi[counter,1] = float(ln[7])
        alphasig[counter,1] = float(ln[8])
        counter += 1
    infile.close()


    # Plot the maximum likelihood values for alpha pre and post-reconstruction, and compute the sample mean and standard error
    fig = plt.figure(10, figsize=(10,7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1], left=0.13, bottom=0.13, right=0.98, top=0.98)

    nbins2 = 75
    minalpha = 0.69
    maxalpha = 1.26
    indexpre = np.where(alphasig[0:,0] >= 2.0)[0]
    indexpost = np.where(alphasig[0:,1] >= 2.0)[0]
    index = np.where(np.logical_and(alphasig[0:,0] >= 2.0, alphasig[0:,1] >= 2.0))[0]
    index2 = np.where(np.logical_and(alphasig[0:,0] < 2.0, alphasig[0:,1] >= 2.0))[0]
    alphameanave = [np.mean(alphamean[indexpre,0]),np.mean(alphamean[indexpost,1])]
    alphameanstd = [np.std(alphamean[indexpre,0]),np.std(alphamean[indexpost,1])]
    alphaerr = (alphaerrlo + alphaerrhi)/2.0
    alphaerrave = [np.mean(alphaerr[indexpre,0]),np.mean(alphaerr[indexpost,1])]
    alphaerrstd = [np.std(alphaerr[indexpre,0]),np.std(alphaerr[indexpost,1])]

    hist1 = np.histogram(alphamean[index,0],bins=nbins2,range=[minalpha,maxalpha])
    hist2 = np.histogram(alphamean[index,1],bins=nbins2,range=[minalpha,maxalpha])
    hist1_2 = np.histogram(alphamean[index2,0],bins=nbins2,range=[minalpha,maxalpha])
    hist2_2 = np.histogram(alphamean[index2,1],bins=nbins2,range=[minalpha,maxalpha])

    print alphameanave, alphameanstd, alphaerrave

    ax1=fig.add_subplot(gs[0])
    ax1.plot(hist1[1][1:], hist1[0], color='r', linewidth=1.5, ls='steps', zorder=2)
    ax1.bar((hist1[1][1:]+hist1[1][0:-1])/2.0, hist1[0], width=hist1[1][1]-hist1[1][0], color='r', zorder=1, alpha=0.2)
    ax1.bar((hist1[1][1:]+hist1[1][0:-1])/2.0, hist1[0], width=hist1[1][1]-hist1[1][0], color='k', zorder=5, alpha=0.4, fill=False)
    ax1.plot(hist1_2[1][1:], hist1_2[0], color='k', linewidth=1.5, ls='steps', zorder=3)
    ax1.set_xlim(minalpha,maxalpha)
    ax1.set_ylim(0.0,np.amax(hist1[0])+2)
    for axis in ['top','left','bottom','right']:
        ax1.spines[axis].set_linewidth(1.3)
    for tick in ax1.xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in ax1.yaxis.get_ticklabels():
        tick.set_fontsize(14)

    ax1=fig.add_subplot(gs[2])
    ax1.errorbar([minalpha,maxalpha], [minalpha,maxalpha], marker='None',markerfacecolor='r',markeredgecolor='k',color='r',linestyle='--',markeredgewidth=1.3, zorder=1, linewidth=1.3)
    ax1.errorbar(alphamean[index,0], alphamean[index,1], marker='o',markerfacecolor='r',markeredgecolor='k',color='r',linestyle='None',markeredgewidth=1.3, zorder=5, markersize=6, alpha=0.4)
    ax1.errorbar(alphamean[index2,0], alphamean[index2,1], marker='x',markerfacecolor='k',markeredgecolor='k',color='k',linestyle='None',markeredgewidth=1.3, zorder=4, markersize=6, alpha=0.4)
    ax1.errorbar(alphameanave[0], alphameanave[1], xerr=alphameanstd[0]/np.sqrt(len(indexpre)), yerr=alphameanstd[1]/np.sqrt(len(indexpost)), marker='s',markerfacecolor='b',markeredgecolor='k',color='b',linestyle='None',markeredgewidth=1.3, zorder=5, markersize=10)
    ax1.axvline(x=1.0, color='k', linestyle='-', linewidth=1.3, zorder=0)
    ax1.axhline(y=1.0, color='k', linestyle='-', linewidth=1.3, zorder=0)
    ax1.set_xlabel(r'$\alpha_{P_{k}},\mathrm{Pre-Reconstruction}$',fontsize=22)
    ax1.set_ylabel(r'$\alpha_{P_{k}},\mathrm{Post-Reconstruction}$',fontsize=22)
    ax1.set_xlim(minalpha, maxalpha)
    ax1.set_ylim(0.78, 1.22)
    for axis in ['top','left','bottom','right']:
        ax1.spines[axis].set_linewidth(1.3)
    for tick in ax1.xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in ax1.yaxis.get_ticklabels():
        tick.set_fontsize(14)
    ax1.text(0.02, 0.90, str(r"$2\sigma\,\mathrm{BAO\,detections}:\,%4d \rightarrow %4d$" % (len(indexpre),len(indexpost))), color='k', fontsize=16, transform=ax1.transAxes, bbox=dict(facecolor='w', edgecolor='none', boxstyle='round'))
    ax1.text(0.25, 0.15, str(r"$\mathrm{Pre-Reconstruction}:\,%5.4lf\pm%5.4lf$" % (alphameanave[0], alphameanstd[0]/np.sqrt(len(indexpre)))), color='k', fontsize=16, transform=ax1.transAxes, bbox=dict(facecolor='w', edgecolor='none', boxstyle='round'))
    ax1.text(0.25, 0.07, str(r"$\mathrm{Post-Reconstruction}:\,%5.4lf\pm%5.4lf$" % (alphameanave[1], alphameanstd[1]/np.sqrt(len(indexpost)))), color='k', fontsize=16, transform=ax1.transAxes, bbox=dict(facecolor='w', edgecolor='none', boxstyle='round'))

    ax1=fig.add_subplot(gs[3])
    ax1.plot(hist2[0], hist2[1][0:-1], color='r', linewidth=1.5, ls='steps', zorder=2)
    ax1.barh((hist2[1][1:]+hist2[1][0:-1])/2.0, hist2[0], height=hist2[1][1]-hist2[1][0], color='r', zorder=1, alpha=0.2)
    ax1.barh((hist2[1][1:]+hist2[1][0:-1])/2.0, hist2[0], height=hist2[1][1]-hist2[1][0], color='k', zorder=5, alpha=0.4, fill=False)
    ax1.plot(hist2_2[0], hist2_2[1][0:-1], color='k', linewidth=1.5, ls='steps', zorder=3)
    ax1.set_ylim(0.78, 1.22)
    ax1.set_xlim(0.0,np.amax(hist2[0])+2)
    for axis in ['top','left','bottom','right']:
        ax1.spines[axis].set_linewidth(1.3)
    for tick in ax1.xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in ax1.yaxis.get_ticklabels():
        tick.set_fontsize(14)

    fig.subplots_adjust(hspace=0.0, wspace=0, bottom=0.12, top=0.98, right=0.98, left=0.14)
    plt.setp([a.get_xticklabels() for a in fig.axes[0:1]], visible=False)
    plt.setp([a.get_yticklabels() for a in fig.axes[2:4]], visible=False)

    outfile.savefig(fig)

    # Plot the error for each mock pre and post-recon and the compute the mean error
    fig = plt.figure(11, figsize=(10,7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1], left=0.13, bottom=0.13, right=0.98, top=0.98)

    minalpha = 0.00
    maxalpha = 0.32
    hist1 = np.histogram(alphaerr[index,0],bins=nbins2,range=[minalpha,maxalpha])
    hist2 = np.histogram(alphaerr[index,1],bins=nbins2,range=[minalpha,maxalpha])
    hist1_2 = np.histogram(alphaerr[index2,0],bins=nbins2,range=[minalpha,maxalpha])
    hist2_2 = np.histogram(alphaerr[index2,1],bins=nbins2,range=[minalpha,maxalpha])

    ax1=fig.add_subplot(gs[0])
    ax1.plot(hist1[1][1:], hist1[0], color='r', linewidth=1.5, ls='steps', zorder=2)
    ax1.bar((hist1[1][1:]+hist1[1][0:-1])/2.0, hist1[0], width=hist1[1][1]-hist1[1][0], color='r', zorder=1, alpha=0.2)
    ax1.bar((hist1[1][1:]+hist1[1][0:-1])/2.0, hist1[0], width=hist1[1][1]-hist1[1][0], color='k', zorder=5, alpha=0.4, fill=False)
    ax1.plot(hist1_2[1][1:], hist1_2[0], color='k', linewidth=1.5, ls='steps', zorder=3)
    ax1.set_xlim(minalpha,0.32)
    ax1.set_ylim(0.0,np.amax(hist1[0])+2)
    for axis in ['top','left','bottom','right']:
        ax1.spines[axis].set_linewidth(1.3)
    for tick in ax1.xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in ax1.yaxis.get_ticklabels():
        tick.set_fontsize(14)

    ax1=fig.add_subplot(gs[2])
    ax1.errorbar([minalpha,maxalpha], [minalpha,maxalpha], marker='None',markerfacecolor='r',markeredgecolor='k',color='r',linestyle='--',markeredgewidth=1.3, zorder=1, linewidth=1.3)
    ax1.errorbar(alphaerr[index,0], alphaerr[index,1], marker='o',markerfacecolor='r',markeredgecolor='k',color='r',linestyle='None',markeredgewidth=1.3, zorder=5, markersize=6, alpha=0.4)
    ax1.errorbar(alphaerr[index2,0], alphaerr[index2,1], marker='x',markerfacecolor='k',markeredgecolor='k',color='k',linestyle='None',markeredgewidth=1.3, zorder=4, markersize=6, alpha=0.4)
    ax1.errorbar(alphaerrave[0], alphaerrave[1], xerr=alphaerrstd[0]/np.sqrt(len(indexpre)), yerr=alphaerrstd[1]/np.sqrt(len(indexpost)), marker='s',markerfacecolor='b',markeredgecolor='k',color='b',linestyle='None',markeredgewidth=1.3, zorder=5, markersize=10)
    ax1.axvline(x=0.02, color='k', linestyle='-', linewidth=1.3, zorder=0)
    ax1.axhline(y=0.02, color='k', linestyle='-', linewidth=1.3, zorder=0)
    ax1.set_xlabel(r'$\sigma_{\alpha}^{P_{k}},\mathrm{Pre-Reconstruction}$',fontsize=22)
    ax1.set_ylabel(r'$\sigma_{\alpha}^{P_{k}},\mathrm{Post-Reconstruction}$',fontsize=22)
    ax1.set_xlim(minalpha,0.32)
    ax1.set_ylim(minalpha,0.15)
    for axis in ['top','left','bottom','right']:
        ax1.spines[axis].set_linewidth(1.3)
    for tick in ax1.xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in ax1.yaxis.get_ticklabels():
        tick.set_fontsize(14)
    ax1.text(0.30, 0.90, str(r"$\mathrm{Better\,errors\,than\,forecast}:\,%4d \rightarrow %4d$" % (len(np.where(alphaerr[0:,0]<0.02)[0]),len(np.where(alphaerr[0:,1]<0.02)[0]))), color='k', fontsize=16, transform=ax1.transAxes, bbox=dict(facecolor='w', edgecolor='none', boxstyle='round'))
    ax1.text(0.40, 0.15, str(r"$\mathrm{Pre-Reconstruction}:\,%5.4lf$" % (alphaerrave[0])), color='k', fontsize=16, transform=ax1.transAxes, bbox=dict(facecolor='w', edgecolor='none', boxstyle='round'))
    ax1.text(0.40, 0.07, str(r"$\mathrm{Post-Reconstruction}:\,%5.4lf$" % (alphaerrave[1])), color='k', fontsize=16, transform=ax1.transAxes, bbox=dict(facecolor='w', edgecolor='none', boxstyle='round'))

    ax1=fig.add_subplot(gs[3])
    ax1.plot(hist2[0], hist2[1][0:-1], color='r', linewidth=1.5, ls='steps', zorder=2)
    ax1.barh((hist2[1][1:]+hist2[1][0:-1])/2.0, hist2[0], height=hist2[1][1]-hist2[1][0], color='r', zorder=1, alpha=0.2)
    ax1.barh((hist2[1][1:]+hist2[1][0:-1])/2.0, hist2[0], height=hist2[1][1]-hist2[1][0], color='k', zorder=5, alpha=0.4, fill=False)
    ax1.plot(hist2_2[0], hist2_2[1][0:-1], color='k', linewidth=1.5, ls='steps', zorder=3)
    ax1.set_ylim(minalpha, 0.15)
    ax1.set_xlim(0.0,np.amax(hist2[0])+2)
    for axis in ['top','left','bottom','right']:
        ax1.spines[axis].set_linewidth(1.3)
    for tick in ax1.xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in ax1.yaxis.get_ticklabels():
        tick.set_fontsize(14)

    fig.subplots_adjust(hspace=0.0, wspace=0, bottom=0.12, top=0.98, right=0.98, left=0.14)
    plt.setp([a.get_xticklabels() for a in fig.axes[0:1]], visible=False)
    plt.setp([a.get_yticklabels() for a in fig.axes[2:4]], visible=False)

    outfile.savefig(fig)

    # Plot the BAO significance pre and post-reconstruction
    fig = plt.figure(12, figsize=(10,7))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3], width_ratios=[3, 1], left=0.13, bottom=0.13, right=0.98, top=0.98)

    minalpha = -2.0
    maxalpha = 7.0
    hist1 = np.histogram(alphasig[0:,0],bins=nbins2,range=[minalpha,maxalpha])
    hist2 = np.histogram(alphasig[0:,1],bins=nbins2,range=[minalpha,maxalpha])
    alphasigave = np.mean(alphasig, axis=0)
    alphasigstd = np.std(alphasig, axis=0)

    ax1=fig.add_subplot(gs[0])
    ax1.plot(hist1[1][1:], hist1[0], color='r', linewidth=1.5, ls='steps', zorder=2)
    ax1.bar((hist1[1][1:]+hist1[1][0:-1])/2.0, hist1[0], width=hist1[1][1]-hist1[1][0], color='r', zorder=1, alpha=0.2)
    ax1.bar((hist1[1][1:]+hist1[1][0:-1])/2.0, hist1[0], width=hist1[1][1]-hist1[1][0], color='k', zorder=5, alpha=0.4, fill=False)
    ax1.set_xlim(-2.0, 6.3)
    ax1.set_ylim(0.0,np.amax(hist1[0])+2)
    for axis in ['top','left','bottom','right']:
        ax1.spines[axis].set_linewidth(1.3)
    for tick in ax1.xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in ax1.yaxis.get_ticklabels():
        tick.set_fontsize(14)

    ax1=fig.add_subplot(gs[2])
    ax1.errorbar([minalpha,maxalpha], [minalpha,maxalpha], marker='None',markerfacecolor='r',markeredgecolor='k',color='r',linestyle='--',markeredgewidth=1.3, zorder=1, linewidth=1.3)
    ax1.errorbar(alphasig[0:,0], alphasig[0:,1], marker='o',markerfacecolor='r',markeredgecolor='k',color='r',linestyle='None',markeredgewidth=1.3, zorder=5, markersize=6, alpha=0.4)
    ax1.errorbar(alphasigave[0], alphasigave[1], xerr=alphasigstd[0]/np.sqrt(nmocks), yerr=alphasigstd[1]/np.sqrt(nmocks), marker='s',markerfacecolor='b',markeredgecolor='k',color='b',linestyle='None',markeredgewidth=1.3, zorder=5, markersize=10)
    ax1.axvline(x=2.0, color='k', linestyle='-', linewidth=1.3, zorder=0)
    ax1.axhline(y=2.0, color='k', linestyle='-', linewidth=1.3, zorder=0)
    ax1.set_xlabel(r'$\Delta \chi^{2}_{P_{k}},\mathrm{Pre-Reconstruction}$',fontsize=22)
    ax1.set_ylabel(r'$\Delta \chi^{2}_{P_{k}},\mathrm{Post-Reconstruction}$',fontsize=22)
    ax1.set_xlim(-2.0, 6.3)
    ax1.set_ylim(-1.0, 7.3)
    for axis in ['top','left','bottom','right']:
        ax1.spines[axis].set_linewidth(1.3)
    for tick in ax1.xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in ax1.yaxis.get_ticklabels():
        tick.set_fontsize(14)
    ax1.text(0.25, 0.15, str(r"$\mathrm{Pre-Reconstruction}:\,%5.4lf\pm%5.4lf$" % (alphasigave[0], alphasigstd[0]/np.sqrt(nmocks))), color='k', fontsize=16, transform=ax1.transAxes)
    ax1.text(0.25, 0.07, str(r"$\mathrm{Post-Reconstruction}:\,%5.4lf\pm%5.4lf$" % (alphasigave[1], alphasigstd[1]/np.sqrt(nmocks))), color='k', fontsize=16, transform=ax1.transAxes)

    ax1=fig.add_subplot(gs[3])
    ax1.plot(hist2[0], hist2[1][0:-1], color='r', linewidth=1.5, ls='steps', zorder=2)
    ax1.barh((hist2[1][1:]+hist2[1][0:-1])/2.0, hist2[0], height=hist2[1][1]-hist2[1][0], color='r', zorder=1, alpha=0.2)
    ax1.barh((hist2[1][1:]+hist2[1][0:-1])/2.0, hist2[0], height=hist2[1][1]-hist2[1][0], color='k', zorder=5, alpha=0.4, fill=False)
    ax1.set_ylim(-1.0, 7.3)
    ax1.set_xlim(0.0,np.amax(hist2[0])+2)
    for axis in ['top','left','bottom','right']:
        ax1.spines[axis].set_linewidth(1.3)
    for tick in ax1.xaxis.get_ticklabels():
        tick.set_fontsize(14)
    for tick in ax1.yaxis.get_ticklabels():
        tick.set_fontsize(14)

    fig.subplots_adjust(hspace=0.0, wspace=0, bottom=0.12, top=0.98, right=0.98, left=0.14)
    plt.setp([a.get_xticklabels() for a in fig.axes[0:1]], visible=False)
    plt.setp([a.get_yticklabels() for a in fig.axes[2:4]], visible=False)

    outfile.savefig(fig)

    nbins2 = 80
    hist_xi = np.histogram(alphasig[0:,0],bins=nbins2,range=(minalpha,maxalpha))
    hist_pk = np.histogram(alphasig[0:,1],bins=nbins2,range=(minalpha,maxalpha))
    nsig_xi1 = len(np.where(alphasig[0:,0] >= 1.0)[0])
    nsig_pk1 = len(np.where(alphasig[0:,1] >= 1.0)[0])
    nsig_xi2 = len(np.where(alphasig[0:,0] >= 2.0)[0])
    nsig_pk2 = len(np.where(alphasig[0:,1] >= 2.0)[0])
    nsig_1 = len(np.where(np.logical_and(alphasig[0:,0] >= 1.0, alphasig[0:,0] >= 1.0))[0])
    nsig_2 = len(np.where(np.logical_and(alphasig[0:,1] >= 2.0, alphasig[0:,1] >= 2.0))[0])

    # Do a cumulative histogram of the BAO signicance
    fig = plt.figure(1, figsize=(10,7))
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
    ax2.text(0.25, 0.12, str(r"$%d$" % (nmocks-nsig_xi1)), color='r', fontsize=15, transform=ax2.transAxes)
    ax2.text(0.25, 0.08, str(r"$%d$" % (nmocks-nsig_pk1)), color='b', fontsize=15, transform=ax2.transAxes)
    ax2.text(0.38, 0.12, str(r"$%d$" % (nsig_xi1-nsig_xi2)), color='r', fontsize=15, transform=ax2.transAxes)
    ax2.text(0.38, 0.08, str(r"$%d$" % (nsig_pk1-nsig_pk2)), color='b', fontsize=15, transform=ax2.transAxes)
    ax2.text(0.50, 0.12, str(r"$%d$" % (nsig_xi2)), color='r', fontsize=15, transform=ax2.transAxes)
    ax2.text(0.50, 0.08, str(r"$%d$" % (nsig_pk2)), color='b', fontsize=15, transform=ax2.transAxes)

    outfile.savefig(fig)
    outfile.close()
