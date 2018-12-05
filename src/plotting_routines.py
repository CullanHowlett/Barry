# A class associated with plotting data and models. This mainly takes a data class object and a model class object as inputs and plots them.

import matplotlib.pyplot as plt
import matplotlib.ticker as tk
import matplotlib.font_manager
from matplotlib import gridspec
import numpy as np
import scipy as sp
from scipy import interpolate
import itertools
from models import *
from read_data import *


matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{ClearSans}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]

class Plotter(object):

    def __init__(self, fignum=1, data=None, model=None, interactive=False):

        self.plt_handle = None
        self.interactive = interactive
        if ((data is not None)):
            self.create_plot(data,model=model,fignum=fignum)

        return

    def create_plot(self, data, model=None, fignum=1, **kwargs):

        if (model is not None):
            if (data.__class__.__name__ != model.datatype):
                print "Datatype (denoting CorrelationFunction, PowerSpectrum, BAOExtract) not consistent between data (", data.__class__.__name__, ") and model (", model.datatype, ")"
                exit()

        color = kwargs.get('color', 'k')
        marker = kwargs.get('marker', 'o')
        markerfacecolor = kwargs.get('markerfacecolor', 'k')
        markeredgecolor = kwargs.get('markeredgecolor', 'k')
        markeredgewidth = kwargs.get('markeredgewidth', 1.3)

        fig = plt.figure(fignum)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.0,1.0], hspace=0.0, left=0.13, bottom=0.1, right=0.95, top=0.98)

        big_axes=plt.subplot(gs[0:])
        big_axes.set_axis_bgcolor('none')
        big_axes.spines['top'].set_color('none')
        big_axes.spines['bottom'].set_color('none')
        big_axes.spines['left'].set_color('none')
        big_axes.spines['right'].set_color('none')
        big_axes.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

        if (data.cov is None):
            cov = np.diag(np.zeros(len(data.x)))
        else:
            cov = data.cov

        self.plt_handle = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
        if (self.interactive):
            self.plt_handle[0].set_xlim(0.95*np.amin(data.x), 1.05*np.amax(data.x))
            self.plt_handle[1].set_xlim(0.95*np.amin(data.x), 1.05*np.amax(data.x))
        if isinstance(data, PowerSpectrum):
            self.plt_handle[1].set_ylim(0.8, 1.2)
            big_axes.set_xlabel(r'$k\,(h\,\mathrm{Mpc}^{-1})$',fontsize=22)
            self.plt_handle[0].set_ylabel(r'$k\,P(k)$',fontsize=22,labelpad=5)
            self.plt_handle[1].set_ylabel(r'$P(k)/P^{\mathrm{nw}}(k)$',fontsize=22,labelpad=5)
            if (self.interactive):
                self.plt_handle[0].set_ylim(0.95*np.amin(data.x*(data.y-np.sqrt(cov[np.diag_indices(len(data.x))]))), 1.05*np.amax(data.x*(data.y+np.sqrt(cov[np.diag_indices(len(data.x))]))))
        elif isinstance(data, CorrelationFunction):
            self.plt_handle[1].set_ylim(-5.0e-3, 5.0e-3)
            big_axes.set_xlabel(r'$s\,(h^{-1}\,\mathrm{Mpc})$',fontsize=22)
            self.plt_handle[0].set_ylabel(r'$s^{2}\,\xi(s)\,(h^{-2}Mpc^{2})$',fontsize=22,labelpad=5)
            self.plt_handle[1].set_ylabel(r'$\xi(s)-\xi^{\mathrm{nw}}(s)$',fontsize=22,labelpad=5)
            if (self.interactive):
                if (np.amin(data.x*data.x*(data.y-np.sqrt(cov[np.diag_indices(len(data.x))]))) < 0.0):
                    self.plt_handle[0].set_ylim(1.05*np.amin(data.x*data.x*(data.y-np.sqrt(cov[np.diag_indices(len(data.x))]))), 1.05*np.amax(data.x*data.x*(data.y+np.sqrt(cov[np.diag_indices(len(data.x))]))))
                else:
                    self.plt_handle[0].set_ylim(0.95*np.amin(data.x*data.x*(data.y-np.sqrt(cov[np.diag_indices(len(data.x))]))), 1.05*np.amax(data.x*data.x*(data.y+np.sqrt(cov[np.diag_indices(len(data.x))]))))
        elif isinstance(data, BAOExtract):
            self.plt_handle[1].set_ylim(-0.15, 0.15)
            big_axes.set_xlabel(r'$k\,(h\,\mathrm{Mpc}^{-1})$',fontsize=22)
            self.plt_handle[0].set_ylabel(r'$R_{p}(k)$',fontsize=22,labelpad=5)
            self.plt_handle[1].set_ylabel(r'$R_{P}(k)/R_{P}^{\mathrm{nw}}(k)$',fontsize=22,labelpad=5)
            if (self.interactive):
                if (np.amin((data.y-np.sqrt(cov[np.diag_indices(len(data.x))]))) < 0.0):
                    self.plt_handle[0].set_ylim(1.05*np.amin((data.y-np.sqrt(cov[np.diag_indices(len(data.x))]))), 1.05*np.amax((data.y+np.sqrt(cov[np.diag_indices(len(data.x))]))))
                else:
                    self.plt_handle[0].set_ylim(0.95*np.amin((data.y-np.sqrt(cov[np.diag_indices(len(data.x))]))), 1.05*np.amax((data.y+np.sqrt(cov[np.diag_indices(len(data.x))]))))
        else:
            print "Datatype ", data.__class__.__name__, " not supported, must be either 'PowerSpectrum', 'CorrelationFunction' or 'BAOExtract'"
            exit()
        for i in self.plt_handle:
            i.tick_params(width=1.3)
            i.tick_params('both',length=10, which='major')
            i.tick_params('both',length=5, which='minor')
            for axis in ['top','left','bottom','right']:
                i.spines[axis].set_linewidth(1.3)
            for tick in i.xaxis.get_ticklabels():
                tick.set_fontsize(14)
            for tick in i.yaxis.get_ticklabels():
                tick.set_fontsize(14)
            if (self.interactive):
                i.set_autoscale_on(False)

        self.add_data_to_plot(data)

        if (model is not None):
            plt_handle = self.add_model_to_plot(data, model)
            if (self.interactive):
                return plt_handle

        return

    def add_data_to_plot(self, data, fignum=1, **kwargs):

        if (self.plt_handle is None):
            if (data ):
                self.create_plot(data,model=model,fignum=fignum)

        color = kwargs.get('color', 'k')
        linestyle = kwargs.get('linestyle', 'None')
        linewidth = kwargs.get('linewidth', 1.3)
        marker = kwargs.get('marker', 'o')
        markerfacecolor = kwargs.get('markerfacecolor', 'k')
        markeredgecolor = kwargs.get('markeredgecolor', 'k')
        markeredgewidth = kwargs.get('markeredgewidth', 1.3)

        if (data.cov is None):
            cov = np.diag(np.zeros(len(data.x)))
        else:
            cov = data.cov

        if isinstance(data, PowerSpectrum):
            self.plt_handle[0].errorbar(data.x,data.x*data.y,yerr=data.x*np.sqrt(cov[np.diag_indices(len(data.x))]),marker=marker,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor,color=color,linestyle=linestyle,markeredgewidth=markeredgewidth, zorder=5) 
        elif isinstance(data, CorrelationFunction):
            self.plt_handle[0].errorbar(data.x,data.x*data.x*data.y,yerr=data.x*data.x*np.sqrt(cov[np.diag_indices(len(data.x))]),marker=marker,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor,color=color,linestyle='None',markeredgewidth=markeredgewidth, zorder=5) 
        elif isinstance(data, BAOExtract):
            self.plt_handle[0].errorbar(data.x,data.y,yerr=np.sqrt(cov[np.diag_indices(len(data.x))]),marker=marker,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor,color=color,linestyle=linestyle,markeredgewidth=markeredgewidth, zorder=5) 
        else:
            print "Datatype ", data.__class__.__name__, " not supported, must be either 'PowerSpectrum', 'CorrelationFunction' or 'BAOExtract'"
            exit()

        return

    def add_model_to_plot(self, data, model, fignum=1, **kwargs):

        if (data.__class__.__name__ != model.datatype):
            print "Datatype (denoting CorrelationFunction, PowerSpectrum, BAOExtract) not consistent between data (", data.__class__.__name__, ") and model (", model.datatype, ")"
            exit()

        if (self.plt_handle is None):
            self.create_plot(data,model=model,fignum=fignum)

        color = kwargs.get('color', 'k')
        linestyle = kwargs.get('linestyle', '-')
        linecolor = kwargs.get('linecolor', 'k')
        linewidth = kwargs.get('linewidth', 1.3)
        marker = kwargs.get('marker', 'o')
        markerfacecolor = kwargs.get('markerfacecolor', 'k')
        markeredgecolor = kwargs.get('markeredgecolor', 'k')
        markeredgewidth = kwargs.get('markeredgewidth', 1.3)

        if (data.cov is None):
            cov = np.diag(np.zeros(len(data.x)))
        else:
            cov = data.cov

        plt_handle = []
        if (model.datatype == "PowerSpectrum"):

            # Evaluate the smooth power spectrum given the current model parameters
            oldBAO = model.BAO
            model.BAO = False
            yvals = model.compute_model(x=data.kwinmatin)
            model.BAO = oldBAO
            p0 = np.sum(data.winmat[0,0:]*yvals)
            pkmod = np.zeros(len(data.winmat[0:])-1)
            for j in range(len(data.winmat[0:])-1):
                pkmod[j] = np.sum(data.winmat[j+1][0:]*yvals) - p0*data.pkwin[j]
            pkmod = pkmod[data.kwinmatoutindex]

            plt_handle.append(self.plt_handle[0].errorbar(data.x,data.x*model.y,color=linecolor,linestyle=linestyle,linewidth=linewidth,zorder=5))
            plt_handle.append(self.plt_handle[0].errorbar(data.x,data.x*pkmod,color=linecolor,linestyle='--',linewidth=linewidth,zorder=5))
            plt_handle.append(self.plt_handle[1].errorbar(data.x,data.y/pkmod,yerr=np.sqrt(cov[np.diag_indices(len(data.x))])/pkmod,marker=marker,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor,color=color,linestyle='None',markeredgewidth=markeredgewidth, zorder=5))
            plt_handle.append(self.plt_handle[1].errorbar(data.x,model.y/pkmod,color=linecolor,linestyle=linestyle,linewidth=linewidth,zorder=5))

        elif (model.datatype == "CorrelationFunction"):

            # Evaluate the smooth correlation function given the current model parameters
            oldBAO = model.BAO
            model.BAO = False
            yvals = model.compute_model(x=data.x)
            model.BAO = oldBAO

            plt_handle.append(self.plt_handle[0].errorbar(model.x,model.x*model.x*model.y,color=linecolor,linestyle=linestyle,linewidth=linewidth,zorder=5))
            plt_handle.append(self.plt_handle[0].errorbar(model.x,model.x*model.x*yvals,color=linecolor,linestyle='--',linewidth=linewidth,zorder=5))
            plt_handle.append(self.plt_handle[1].errorbar(data.x,data.y-yvals,yerr=np.sqrt(cov[np.diag_indices(len(data.x))]),marker=marker,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor,color=color,linestyle='None',markeredgewidth=markeredgewidth, zorder=5))
            plt_handle.append(self.plt_handle[1].errorbar(data.x,model.y-yvals,color=linecolor,linestyle=linestyle,linewidth=linewidth,zorder=5))

        elif (model.datatype == "BAOExtract"):

            # Evaluate the smooth BAOExtract given the current model parameters
            oldBAO = model.BAO
            model.BAO = False
            yvals = model.compute_model(x=data.kwinmatin)
            model.BAO = oldBAO
            p0 = np.sum(data.winmat[0,0:]*yvals)
            pkmod = np.zeros(len(data.winmat[0:])-1)
            for j in range(len(data.winmat[0:])-1):
                pkmod[j] = np.sum(data.winmat[j+1][0:]*yvals) - p0*data.pkwin[j]
            yvals = model.extract_BAO(data.kwinmatout, pkmod)[data.kwinmatoutindex]

            plt_handle.append(self.plt_handle[0].errorbar(data.x,model.y,color=linecolor,linestyle=linestyle,linewidth=linewidth,zorder=5))
            plt_handle.append(self.plt_handle[0].errorbar(data.x,yvals,color=linecolor,linestyle='--',linewidth=linewidth,zorder=5))
            plt_handle.append(self.plt_handle[1].errorbar(data.x,data.y-yvals,yerr=np.sqrt(cov[np.diag_indices(len(data.x))]),marker=marker,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor,color=color,linestyle='None',markeredgewidth=markeredgewidth, zorder=5))
            plt_handle.append(self.plt_handle[1].errorbar(data.x,model.y-yvals,color=linecolor,linestyle=linestyle,linewidth=linewidth,zorder=5))

        else:
            print "Datatype ", model.datatype, " not supported, must be either 'PowerSpectrum', 'CorrelationFunction' or 'BAOExtractor'"
            exit()

        return plt_handle

    def add_model_to_plot_fixedsmooth(self, data, model, smooth_params, fignum=1, **kwargs):

        if (data.__class__.__name__ != model.datatype):
            print "Datatype (denoting CorrelationFunction, PowerSpectrum, BAOExtract) not consistent between data (", data.__class__.__name__, ") and model (", model.datatype, ")"
            exit()

        if (self.plt_handle is None):
            self.create_plot(data,model=model,fignum=fignum)

        color = kwargs.get('color', 'k')
        linestyle = kwargs.get('linestyle', '-')
        linewidth = kwargs.get('linewidth', 1.3)
        marker = kwargs.get('marker', 'o')
        markerfacecolor = kwargs.get('markerfacecolor', 'k')
        markeredgecolor = kwargs.get('markeredgecolor', 'k')
        markeredgewidth = kwargs.get('markeredgewidth', 1.3)

        if (data.cov is None):
            cov = np.diag(np.zeros(len(data.x)))
        else:
            cov = data.cov

        # Evaluate the smooth power spectrum given the passed in parameters
        free_params = model.get_free_params()
        oldparams = np.empty(len(free_params))
        for counter, i in enumerate(free_params):
            oldparams[counter] = model.params[i][0]

        oldBAO = model.BAO
        model.BAO = False
        free_params = model.get_free_params()
        for counter, i in enumerate(free_params):
            model.params[i][0] = smooth_params[counter]

        plt_handle = []
        if (model.datatype == "PowerSpectrum"):

            yvals = model.compute_model(x=data.kwinmatin)
            p0 = np.sum(data.winmat[0,0:]*yvals)
            pkmod = np.zeros(len(data.winmat[0:])-1)
            for j in range(len(data.winmat[0:])-1):
                pkmod[j] = np.sum(data.winmat[j+1][0:]*yvals) - p0*data.pkwin[j]
            pkmod = pkmod[data.kwinmatoutindex]

            plt_handle.append(self.plt_handle[0].errorbar(data.x,data.x*model.y,color=color,linestyle=linestyle,linewidth=linewidth,zorder=5))
            plt_handle.append(self.plt_handle[0].errorbar(data.x,data.x*pkmod,color=color,linestyle='--',linewidth=linewidth,zorder=5))
            plt_handle.append(self.plt_handle[1].errorbar(data.x,data.y/pkmod,yerr=np.sqrt(cov[np.diag_indices(len(data.x))])/pkmod,marker=marker,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor,color=color,linestyle='None',markeredgewidth=markeredgewidth, zorder=5))
            plt_handle.append(self.plt_handle[1].errorbar(data.x,model.y/pkmod,color=color,linestyle=linestyle,linewidth=linewidth,zorder=5))

        elif (model.datatype == "CorrelationFunction"):

            # Evaluate the smooth correlation function given the current model parameters
            yvals = model.compute_model(x=data.x)

            plt_handle.append(self.plt_handle[0].errorbar(model.x,model.x*model.x*model.y,color=color,linestyle=linestyle,linewidth=linewidth,zorder=5))
            plt_handle.append(self.plt_handle[0].errorbar(model.x,model.x*model.x*yvals,color=color,linestyle='--',linewidth=linewidth,zorder=5))
            plt_handle.append(self.plt_handle[1].errorbar(data.x,data.y-yvals,yerr=np.sqrt(cov[np.diag_indices(len(data.x))]),marker=marker,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor,color=color,linestyle='None',markeredgewidth=markeredgewidth, zorder=5))
            plt_handle.append(self.plt_handle[1].errorbar(data.x,model.y-yvals,color=color,linestyle=linestyle,linewidth=linewidth,zorder=5))
        
        elif (model.datatype == "BAOExtract"):

            yvals = model.compute_model(x=data.kwinmatin)
            p0 = np.sum(data.winmat[0,0:]*yvals)
            pkmod = np.zeros(len(data.winmat[0:])-1)
            for j in range(len(data.winmat[0:])-1):
                pkmod[j] = np.sum(data.winmat[j+1][0:]*yvals) - p0*data.pkwin[j]
            yvals = model.extract_BAO(data.kwinmatout, pkmod)[data.kwinmatoutindex]

            plt_handle.append(self.plt_handle[0].errorbar(model.x,model.y,color=color,linestyle=linestyle,linewidth=linewidth,zorder=5))
            plt_handle.append(self.plt_handle[0].errorbar(model.x,yvals,color=color,linestyle='--',linewidth=linewidth,zorder=5))
            plt_handle.append(self.plt_handle[1].errorbar(data.x,data.y-yvals,yerr=np.sqrt(cov[np.diag_indices(len(data.x))]),marker=marker,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor,color=color,linestyle='None',markeredgewidth=markeredgewidth, zorder=5))
            plt_handle.append(self.plt_handle[1].errorbar(data.x,model.y-yvals,color=color,linestyle=linestyle,linewidth=linewidth,zorder=5))

        else:
            print "Datatype ", model.datatype, " not supported, must be either 'PowerSpectrum', 'CorrelationFunction' or 'BAOExtractor'"
            exit()

        model.BAO = oldBAO
        free_params = model.get_free_params()
        for counter, i in enumerate(free_params):
            model.params[i][0] = oldparams[counter]

        return plt_handle

    def display_plot(self, plt_array=None, hold=False):

        if (self.interactive):
            if (hold):
                plt.ioff()
                plt.show()
                plt.ion()
            else:
                plt.pause(0.005)
                if (plt_array is not None):
                    for i in itertools.chain.from_iterable(plt_array):
                        if i is not None:
                            i.remove()
        else:
            plt.show()

        return

