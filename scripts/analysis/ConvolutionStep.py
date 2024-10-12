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

class ConvolutionStep:
    def __init__(self, plotpath, cv_dict=None, showplots=False):
        self.initialize_cv_params(cv_dict)

        self.DPI = 300

        self.SHOWPLOTS = showplots
        self.plot_path = plotpath

        self.M_t_times = None
        self.M_t_values = None

        self.mass_transfer_function = None

        self.CV = None
        self.CV_planar = None

    
    def run(self):
        self.load_M_t()
        self.generate_mass_transfer_function()

        self.CV = self.cv_calculator(area = 1, mass_transfer_function = self.mass_transfer_function)
        self.CV_planar = self.cv_calculator(area = 1, mass_transfer_function = self.planar_mass_transfer_function)
        
        self.plot_cv_and_planar()

    def cv_calculator(self, area, mass_transfer_function):
        #==============================================================================
        # Define a function for computing the normalized voltammetric profiles
        #==============================================================================
        c = 1
        dXi = 0.01

        A = area
        
        sigma = self.n*self.F*self.Diff_co*self.Scanrate/(self.R*self.T)
        E_span = 2*self.E_up - self.E_in - self.E_fin
        
        Xi_in = self.F*self.E_in/(self.R*self.T)
        Xi_up = self.F*self.E_up/(self.R*self.T)
        Xi_fin = self.F*self.E_fin/(self.R*self.T)
        Xi_forw = np.arange(Xi_in, Xi_up, dXi)
        Xi_backw = np.arange(Xi_up, Xi_fin, -dXi)
        Xi = np.concatenate([Xi_forw, Xi_backw])
        Xi_Num = len(Xi)
        MaxTime = E_span/self.Scanrate
        time = np.linspace(0, MaxTime, Xi_Num)
        kinTerm = self.Diff_co**0.5/(self.kzero*np.exp(self.alpha*Xi))
        print("minimum time = ", time[1])
        print("maximum time = ", time[-1])
        Current = np.zeros(Xi_Num)
        #--------------------------------------------------------------------------
        # interpolate convolution function
        #--------------------------------------------------------------------------
        ConvFunc = mass_transfer_function(time)
        DeltaConvFunc = ConvFunc[1::]-ConvFunc[:-1]
        for i in range(Xi_Num-1):
            if i == 0:
                Current[i] = self.n*self.F*A*c*self.Diff_co**0.5/(kinTerm[i] + ConvFunc[1]*(1 + np.exp(-Xi[i]) ) )
            if i > 0:
                ConvListSum = np.sum(Current[:i+1:]*DeltaConvFunc[i::-1])
                Current[i] = (self.n*self.F*A*c*self.Diff_co**0.5 - (1 + np.exp(-Xi[i]) )*ConvListSum  )/(kinTerm[i] + ConvFunc[1]*(1 + np.exp(-Xi[i]) ) )
        print(np.max(Current[:-1:]/(self.n*self.F*A*c*sigma**0.5)))
        return Xi[:-1]*self.R*self.T/self.F , Current[:-1]/(self.n*self.F*A*c*sigma**0.5)

    def initialize_cv_params(self,cv_dict):
        if cv_dict is None:
            self.E_in     = -0.4
            self.E_up     = 0.4
            self.E_fin    = self.E_in
            self.n        = 1
            self.R        = 8.314
            self.T        = 298
            self.F        = 96485
            self.Diff_co  = 1e-6
            self.alpha    = 0.5
            self.Scanrate = 0.01
            self.Lambda   = 15
            self.kzero    = self.Lambda*(self.Diff_co*self.n*self.F*self.Scanrate/(self.R*self.T))**0.5
        else:
            self.E_in     = cv_dict['E_in']
            self.E_up     = cv_dict['E_up']
            self.E_fin    = cv_dict['E_fin']
            self.n        = cv_dict['n']
            self.R        = cv_dict['R']
            self.T        = cv_dict['T']
            self.F        = cv_dict['F']
            self.Diff_co  = cv_dict['Diff_co']
            self.alpha    = cv_dict['alpha']
            self.Scanrate = cv_dict['Scanrate']
            self.Lambda   = cv_dict['Lambda']
            self.kzero    = self.Lambda*(self.Diff_co*self.n*self.F*self.Scanrate/(self.R*self.T))**0.5
        print("kzero = ", self.kzero)

    def load_M_t(self):
        self.M_t_times = np.load(self.plot_path + 'M_t_times.npy')
        self.M_t_values = np.load(self.plot_path + 'M_t_values.npy')

    def planar_mass_transfer_function(self,t):
        return 2*(t/np.pi)**0.5

    def generate_mass_transfer_function(self):
        self.mass_transfer_function = InterpolatedUnivariateSpline(self.M_t_times, self.M_t_values, k = 3)

    def plot_cv_and_planar(self):
        fig     = plt.figure(figsize = (13.5,4), dpi = self.DPI, tight_layout=True)
        plot1   = fig.add_subplot(131)
        plot1.plot(self.CV_planar[0], self.CV_planar[1],  color = 'red',   linestyle = '-',   label = "Planar")
        plot1.plot(self.CV[0], self.CV[1], color = 'black', linestyle = '-')
        plot1.set_title("CVs pyramids", fontsize = 15)
        plot1.set_xlabel(r'$(E(t)-E^{0})/\mathrm{V}$', fontsize = 15)
        plot1.set_ylabel(r'$\chi(t)$', fontsize = 15)
        plot1.tick_params(direction = 'in', length=4, width=0.5, colors='k', labelsize = 15)
        plot1.legend(frameon = False, fontsize = 13)

        plt.savefig(self.plot_path + "CV_and_planar.png", dpi = self.DPI)
        if self.SHOWPLOTS:
            plt.show()


