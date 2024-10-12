# -*- coding: utf-8 -*-
"""

@author: Alex Tichter, Tim Tichter
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

from .analysis_utils import eliminate_equalities_on_abscissia
from .analysis_utils import FirstDerivative_LogGrid

class FluxReconstruction:
    def __init__(self, cutoff_time, skip_time_decades, minlambda, maxlambda, time_list, used_times_list, flux_list, t_thresh_list, lambda_list, dt_list, plotpath, showplots=False):  
        self.DPI = 300

        self.SHOWPLOTS = showplots

        self.Cutoff_time = cutoff_time
        self.Skip_time_decades = skip_time_decades
        self.MinLambda = minlambda
        self.MaxLambda = maxlambda

        self.time_list = time_list
        self.used_times_list = used_times_list
        self.flux_list = flux_list
        self.t_thresh_list = t_thresh_list
        self.lambda_list = lambda_list
        self.dt_list = dt_list

        self.plot_path = plotpath

    def run(self):
        self.concatenate_refinements()
        self.plot_lamda_grid_time_switches()
        self.plot_reconstructed_flux()

    def concatenate_refinements(self):
        # Now, assemble the fluxes of the individual simulations. Start with the 
        # simulation with the highest resolution! (so the one which was computed 
        # as the last) (LIFO))
        #==========================================================================
        concatenated_time = np.array(self.used_times_list[-1])
        concatenated_flux = np.array(self.flux_list[-1])
        for i in range(len(self.flux_list)-1):
            Start_Index       = np.abs((np.array(self.used_times_list[len(self.time_list)-(i+2)])-concatenated_time[-1])).argmin()
            concatenated_time = np.concatenate([concatenated_time, np.array(self.used_times_list[-(i+2)][Start_Index::])])
            concatenated_flux = np.concatenate([concatenated_flux, np.array(self.flux_list[-(i+2)][Start_Index::])])
        #--------------------------------------------------------------------------
        # now, create the initial time-part, where any diffusion should be semi
        # infinite and the last part, where diffusion is also semi-infinite by
        # more than the threshold value (e.g. 99.99%).
        #--------------------------------------------------------------------------
        concatenated_flux        = concatenated_flux[concatenated_time > self.Cutoff_time]  
        concatenated_time        = concatenated_time[concatenated_time > self.Cutoff_time]  
        initial_times            = np.logspace(-10,np.log10(self.Cutoff_time)-np.abs(self.Skip_time_decades), 100)
        initial_flux             = (1/(np.pi*initial_times))**0.5
        additional_times         = np.logspace(np.log10(2*concatenated_time[-1] - concatenated_time[-2]), 7, 1000)
        #--------------------------------------------------------------------------
        # Change to logarithmic grids for extrapolation as these are better/smoother
        # to handle and give better results.
        #--------------------------------------------------------------------------
        self.log10_initial_times      = np.log10(initial_times)
        self.log10_initial_flux       = np.log10(initial_flux)
        self.log10_concatenated_time  = np.log10(concatenated_time)
        self.log10_concatenated_flux  = np.log10(concatenated_flux)
        self.log10_additional_times   = np.log10(additional_times)
        self.log10_additional_flux    = self.log10_concatenated_flux[-1] - 0.5*(self.log10_additional_times - self.log10_concatenated_time[-1])
        Concatenated_logtime     = np.concatenate([self.log10_initial_times,  self.log10_concatenated_time])
        Concatenated_logflux     = np.concatenate([self.log10_initial_flux,   self.log10_concatenated_flux])
        Concatenated_logtime     = np.concatenate([Concatenated_logtime, self.log10_additional_times] )
        Concatenated_logflux     = np.concatenate([Concatenated_logflux, self.log10_additional_flux]  )
        #--------------------------------------------------------------------------
        # if some time-entries have been included twice during the concatenation
        # or if there was a movement back in time, eliminate these steps by the following
        # funtion. This needs to be done, since InterpolatedUnivariateSpline needs a 
        # monotonously growing abscissia!
        #--------------------------------------------------------------------------
        Concatenated_logtime, Concatenated_logflux = eliminate_equalities_on_abscissia(x_data = Concatenated_logtime, 
                                                                                    y_data = Concatenated_logflux)
        #--------------------------------------------------------------------------
        # Now, do the interpolation
        #--------------------------------------------------------------------------
        Full_time_interp_logGrid = InterpolatedUnivariateSpline(Concatenated_logtime, Concatenated_logflux, k = 3) 
        Full_time_Grid           = np.logspace(Concatenated_logtime[0], Concatenated_logtime[-1], 25000)
        self.log10_Full_time_Grid     = np.log10(Full_time_Grid)
        self.log10_Full_flux_interpol = Full_time_interp_logGrid(self.log10_Full_time_Grid)
        np.save(self.plot_path + "Y_Decad_log_full_time_grid.npy", self.log10_Full_time_Grid     )
        np.save(self.plot_path + "Y_Decad_log_full_flux_grid.npy", self.log10_Full_flux_interpol )


    def plot_lamda_grid_time_switches(self):
        fig     = plt.figure(figsize = (6,4), dpi = self.DPI, tight_layout=True)
        plot1   = fig.add_subplot(111)
        plot2   = plot1.twinx()
        plot1.set_title("Lambda-grid and time-switches", fontsize = 15)
        plot1.tick_params(direction = 'in', length=4, width=0.5, colors='k', labelsize = 15)
        plot2.tick_params(direction = 'in', length=4, width=0.5, colors='k', labelsize = 15)
        plot1.set_xlabel('$\mathrm{log}_{10}\,(t/\mathrm{s})$', fontsize = 15)
        plot1.set_ylabel('$\mathrm{log}_{10}\,(\lambda)$', fontsize = 15)
        plot2.set_ylabel('$\mathrm{log}_{10}\,(\Delta t)$', fontsize = 15, color = 'grey')
        #--------------------------------------------------------------------------
        # do the actual plotting
        #--------------------------------------------------------------------------
        for i in range(len(self.time_list)):
            plot1.plot(np.log10(np.array(self.time_list[i])), np.log10(np.array(self.lambda_list[i])),  color = 'black',   linewidth = 1.5, linestyle = '-')
            plot1.axvline(np.log10(self.t_thresh_list[i]),  linestyle = '--', linewidth = 0.75,    color = 'black')
            if i == 0:
                plot1.axvline(np.log10(self.used_times_list[i][-1]), color = 'black', linewidth = 1, linestyle = '-.')
            plot1.axvline(np.log10(self.used_times_list[i][-1]), color = 'black', linewidth = 0.5, linestyle = '--')
            plot2.plot(np.log10(np.array(self.time_list[i])), np.log10(np.array(self.dt_list[i])),      color = 'grey',    linewidth = 1, linestyle = '-')
        #--------------------------------------------------------------------------
        # plot specific regions and transitions
        #--------------------------------------------------------------------------
        plot1.axhline(np.log10(0.5),       linestyle = '--', linewidth = 0.75, color = 'black')
        plot1.axhline(np.log10(self.MinLambda), linestyle = '--', linewidth = 0.75, color = 'black')
        plot1.axhline(np.log10(self.MaxLambda), linestyle = '--', linewidth = 0.75, color = 'black')
        plt.savefig(self.plot_path + "01_Lambda_grid_and_time_switches.png", dpi = self.DPI)
        if self.SHOWPLOTS:
                plt.show()

    def plot_reconstructed_flux(self):
        #==========================================================================
        # Plot the flux vs. time and the logarithmic derivative
        #==========================================================================
        fig     = plt.figure(figsize = (6,4), dpi = self.DPI, tight_layout=True)
        plot1   = fig.add_subplot(111)
        plot2   = plot1.twinx()
        plot1.set_title("Reconstructing the flux", fontsize = 15)
        plot1.tick_params(direction = 'in', length=4, width=0.5, colors='k', labelsize = 15)
        plot2.tick_params(direction = 'in', length=4, width=0.5, colors='k', labelsize = 15)
        plot1.set_xlabel('$\mathrm{log}_{10}\,(t/\mathrm{s})$', fontsize = 15)
        plot1.set_ylabel('$\mathrm{log}_{10}\,(f(t)\,/\,\mathrm{mol\,}\mathrm{s}^{-1}\mathrm{cm}^{-2})$', fontsize = 15)
        plot2.set_ylabel('$\mathrm{\Delta\,log}_{10}\,(f_{\mathrm{num}}(t))\,/\,\mathrm{\Delta\,log}_{10}\,(t)$',      fontsize = 15, color = 'grey')
        for i in range(len(self.time_list)):
            if i == 0:
                plot1.plot(np.log10(self.time_list[i]),np.log10((1/(np.pi*self.time_list[i]))**0.5), color = 'red',     linewidth = 0.5,  linestyle = '-')
                plot1.axvline(np.log10(self.used_times_list[i][-1]), color = 'black', linewidth = 1, linestyle = '-.')
            plot1.plot(np.log10(self.used_times_list[i]),np.log10((1/(np.pi*self.used_times_list[i]))**0.5), color = 'red',     linewidth = 0.5,  linestyle = '-')
            plot1.plot(np.log10(self.used_times_list[i]),np.log10(self.flux_list[i]),            color = 'black',   linewidth = 0.75, linestyle = '--')
            plot1.axvline(np.log10(self.used_times_list[i][-1]), color = 'black', linewidth = 0.5, linestyle = '--')
            LogDerivative = FirstDerivative_LogGrid(xdata = self.used_times_list[i], ydata = self.flux_list[i])
            plot2.plot(LogDerivative[0], LogDerivative[1], color = 'grey',   linewidth = 0.5  )
            plot2.axhline(-0.5,     color = 'black', linestyle = ':',          linewidth = 0.7)
        
        plot1.plot(self.log10_initial_times,     self.log10_initial_flux,       color = 'magenta',    linewidth = 1.0, linestyle = '--')
        plot1.plot(self.log10_concatenated_time, self.log10_concatenated_flux,  color = 'black',      linewidth = 1.0, linestyle = '-' ) 
        plot1.plot(self.log10_additional_times,  self.log10_additional_flux,    color = 'magenta',    linewidth = 1.0, linestyle = '--')
        plot1.plot(self.log10_Full_time_Grid,    self.log10_Full_flux_interpol, color = 'black',      linewidth = 1.2, linestyle = ':' )
        plot1.set_xlim(-10.2, 2)
        plt.savefig(self.plot_path + "02_Log_f_vs_log_time_and_Derivatives.png", dpi = self.DPI)
        if self.SHOWPLOTS:
            plt.show()