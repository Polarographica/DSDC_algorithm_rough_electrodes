# -*- coding: utf-8 -*-
"""

@author: Alex Tichter, Tim Tichter
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt

font = {'family': 'Times New Roman', 'color':  'black','weight': 'normal','size': 15,}
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.sans-serif'] = ['Times new Roman']

from .Gaver_Stehfest_NLT_and_NILT import *

class DeconvolutionStep:
    def __init__(self, plotpath, showplots=False):
        self.DPI = 300

        self.SHOWPLOTS = showplots

        self.plot_path = plotpath

        self.data_times = None
        self.data_flux = None

        self.M_t_times = None
        self.M_t_values = None

        self.s_values = None
        self.f_s_values = None

    def run(self): 
        self.load_data()
        self.calculate_mass_transfer_function()
        self.plot_mass_transfer_function()

    def load_data(self):
        """
        Load the data from the given path
        """
        self.data_times = np.load(self.plot_path + 'Y_Decad_log_full_time_grid.npy')
        self.data_flux = np.load(self.plot_path + 'Y_Decad_log_full_flux_grid.npy')

    def calculate_mass_transfer_function(self):
        initial_times            = np.logspace(-21, -10.01, 200)
        initial_flux             = (1/(np.pi*initial_times))**0.5
        additional_times         = np.logspace(7.01, 21, 500)

        log10_initial_times      = np.log10(initial_times)
        log10_initial_flux       = np.log10(initial_flux)
        log10_additional_times   = np.log10(additional_times)
        log10_additional_flux    = self.data_flux[-1] - 0.5*(log10_additional_times - self.data_times[-1])
        
        Concatenated_logtime     = np.concatenate([log10_initial_times,  self.data_times])
        Concatenated_logflux     = np.concatenate([log10_initial_flux,   self.data_flux])
        Concatenated_logtime     = np.concatenate([Concatenated_logtime, log10_additional_times] )
        Concatenated_logflux     = np.concatenate([Concatenated_logflux, log10_additional_flux]  )
        
        Full_time_interp_logGrid = InterpolatedUnivariateSpline(Concatenated_logtime, Concatenated_logflux, k = 3) 
        Full_time_Grid           = np.logspace(-20, 20, 160000)
        log10_Full_time_Grid     = np.log10(Full_time_Grid)
        log10_Full_flux_interpol = Full_time_interp_logGrid(log10_Full_time_Grid)
        ###########################################################################
        # Now, perform numerical Laplace transformation of the normalized flux
        ###########################################################################
        print("Now computing Laplacetransformed flux f(s)")
        fs_num = num_lap_trans(time      = Full_time_Grid, 
                            timefunc  = 10**log10_Full_flux_interpol, 
                            s_min     = 1e-10, 
                            s_max     = 1e10, 
                            s_num     = 80000)
        print("Laplacetransformation of the flux was calculated. Now computing M(t)")
        self.s_values      = fs_num[0]
        self.f_s_values    = fs_num[1]

        M_s_values    = 1/((self.s_values**2)*self.f_s_values)
        M_s_interpol  = InterpolatedUnivariateSpline(self.s_values, M_s_values, k = 3)
        self.M_t_times = np.logspace(-6,6, 48000)
        self.M_t_values = gaver_stehfest_inversion(timepoints = self.M_t_times, LaplaceFunction = M_s_interpol)
        
        np.save(self.plot_path + "M_t_times.npy", self.M_t_times)
        np.save(self.plot_path + "M_t_values.npy", self.M_t_values)
        
        print("M(t) was computed")

    def plot_mass_transfer_function(self):
        fig     = plt.figure(figsize = (6,4), dpi = self.DPI, tight_layout=True)
        plot1   = fig.add_subplot(111)
        plot1.set_title("NLT of the normalized flux", fontsize = 15)
        plot1.plot(np.log10(self.s_values), np.log10(1/self.s_values**0.5), color = 'red',  label = 'analytical')
        plot1.plot(np.log10(self.s_values), np.log10(self.f_s_values), color = 'black', linestyle = ':',    label = 'NLT')   
        plot1.set_xlabel('log$_{10}\,(t\,/\,\mathrm{s})$', fontsize = 15)
        plot1.set_ylabel('$\mathrm{log}_{10}(\overline{f}(s))$', fontsize = 15)
        plot1.tick_params(direction = 'in', length=4, width=0.5, colors='k', labelsize = 15)
        plot1.legend(frameon = False, fontsize = 15)
        plot2   = plot1.twinx()
        plot2.plot(np.log10(self.s_values), 100*(self.f_s_values/(1/self.s_values**0.5)-1), color = 'grey')
        plot2.set_ylabel('$M_{\mathrm{num}}(t)/M_{\mathrm{ana}}(t)$ in %', fontsize = 15, color = 'grey')
        plot2.set_ylim(-0.25,0.25)
        plot2.tick_params(direction = 'in', length=4, width=0.5, colors='k', labelsize = 15)
        plot2.axhline(0, color = 'black', linewidth = 0.7, linestyle = ':')
        
        plt.savefig(self.plot_path + "NLT_for_f_s_planar_and_comparison.png", dpi = self.DPI)
        
        if self.SHOWPLOTS:
            plt.show()

        # M_T_planar
        fig     = plt.figure(figsize = (6,4), dpi = self.DPI, tight_layout=True)
        plot1   = fig.add_subplot(111)
        plot1.set_title("Computation of the mass-transfer-function", fontsize = 15)
        plot1.plot(np.log10(self.M_t_times), 2*(self.M_t_times/np.pi)**0.5, color = 'red',  label = 'analytical')
        plot1.plot(np.log10(self.M_t_times), self.M_t_values, color = 'black', linestyle = ':', label = 'NILT')   
        plot1.set_xlabel('log$_{10}\,(t\,/\,\mathrm{s})$', fontsize = 15)
        plot1.set_ylabel('$M(t)$', fontsize = 15)
        plot1.tick_params(direction = 'in', length=4, width=0.5, colors='k', labelsize = 15)
        plot1.legend(frameon = False, fontsize = 15)
        plot2   = plot1.twinx()
        plot2.plot(np.log10(self.M_t_times), 100*(self.M_t_values/(2*(self.M_t_times/np.pi)**0.5)-1), color = 'grey')
        plot2.set_ylabel('$M_{\mathrm{num}}(t)/M_{\mathrm{ana}}(t)$ in %', fontsize = 15, color = 'grey')
        plot2.set_ylim(-0.25,0.25)
        plot2.tick_params(direction = 'in', length=4, width=0.5, colors='k', labelsize = 15)
        plot2.axhline(0, color = 'black', linewidth = 0.7, linestyle = ':')
        plt.savefig(self.plot_path + "NILT_for_M_t_planar_and_comparison.png", dpi = self.DPI)
        
        if self.SHOWPLOTS:
            plt.show()

