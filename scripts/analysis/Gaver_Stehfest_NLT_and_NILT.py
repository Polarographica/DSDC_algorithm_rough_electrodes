# -*- coding: utf-8 -*-
"""

@author: Tim Tichter, Alex Tichter
"""

#==============================================================================
# In this script, the functions for a numerical discrete logarithmic 
# Laplace transformation and for the numerical inverse Laplace transformation
# according to the Gaver-Stehfest algorithm are defined. Please note that 
# the Gaver Stehfest algorithm does only work for functions which have a non-
# oscillatory behaviour in the time-domain (i.e. which have poles or branch-cuts
# located on the negative real axis only). Fortunately, this is the case for 
# all our mass-transfer functions which we want to compute so this can be done
# for a much faster and more elegant deconvolution procedure.
#==============================================================================
import     numpy                as     np
from       scipy.interpolate    import InterpolatedUnivariateSpline
from       math                 import floor

from tqdm import tqdm

#==============================================================================
# The following function performs a numerical Laplace transformation on an 
# exponentially expanding time-grid
#==============================================================================
def num_lap_trans(time, timefunc, s_min, s_max, s_num):
    d_t   = time[1::]-time[0:-1:]
    s     = np.logspace(np.log10(s_min), np.log10(s_max), s_num)
    Fs    = np.zeros(len(s))
    for i in tqdm(range(len(s))):
        Kernel     = timefunc[1:-2]*np.exp(-s[i]*time[1:-2])
        Summands   = Kernel*d_t[1:-1:]
        Fs[i]      = np.sum(Summands) + 0.5*d_t[0]*timefunc[0]*np.exp(-s[i]*time[0]
                            ) + 0.5*d_t[-1]*timefunc[-1]*np.exp(-s[i]*time[-1])  
    Interpol = InterpolatedUnivariateSpline(s, Fs)
    return s, Fs, Interpol

#============================================================================== 
# The following two functions are needed for computing the Gaver-coefficients.#
# The first is a factorial function, the second computs binominal coefficients.     
#==============================================================================
def fact(x):
    result = 1
    for i in range(x):
        result = result*(i+1)
    return float(result)

def binomk(a,b):
    return fact(a)/(fact(b)*fact(a-b))

#============================================================================== 
# The following computes the coefficients for the Gaver-Stehfest method     
#==============================================================================
def gaver_stehfest_inversion(timepoints, LaplaceFunction):
    #-------------------------------------------
    # the following few lines will compute the
    # Gaver-coefficients.
    #-------------------------------------------
    n    = 7
    ak_n = np.zeros(2*n)
    for kk in range(len(ak_n)):
        k = kk+1
        j = int(floor((k+1)/2.0))
        up = min((k), n)
        arr = np.arange(j,up+1,1)
        summands = []
        for i in range(up-j+1):
            summands.append(((arr[i]**(n+1))/fact(n))*binomk(n,arr[i])*binomk(2*arr[i],arr[i])*binomk(arr[i],k-arr[i]))
        ak_n[kk] = ((-1)**(n+k))*np.sum(summands)
    #print("Gaver_coeffs =", n, "\t", ak_n)
    #-------------------------------------------
    # At next, the actual Laplace-inversion will
    # be computed from the input data.
    #-------------------------------------------
    f = np.zeros(len(timepoints))
    for i in tqdm(range(len(timepoints))):
        vf = np.log(2)/timepoints[i]
        summands = []
        for ii in range(2*n):
            summands.append(ak_n[ii]*LaplaceFunction((ii+1)*np.log(2)/timepoints[i])   )
        summe = np.sum(summands)
        f[i] = float(vf*summe)
    return f


###############################################################################
###############################################################################
# The stuff below is testing. Select the type of the mass transfer function.
# m--> the classical mass transfer function and M--> its antiderivative!
###############################################################################
###############################################################################
Type = "m"
#============================================================================== 
# The following functions are the time-domain and the Laplace-domain
# pairs of the semi-infinite mass-transfer functions
#==============================================================================  
def time_domain_function(time):
    return 1/(np.pi*time)**0.5

def laplace_domain_function(s):
    return 1/(s)**0.5

#============================================================================== 
# The following functions are the time-domain and the Laplace-domain
# pairs of the integrals of the semi-infinite mass-transfer functions
#==============================================================================  
if Type == "M":
    def time_domain_function(time):
        return 2*(time/np.pi)**0.5
    
    def laplace_domain_function(s):
        return 1/(s)**1.5

###############################################################################
# Testing the fucntions
###############################################################################
'''
#==============================================================================
# define the time-grid
#==============================================================================
t_min = 1e-6
t_max = 50000
t_num = 5000
t1    = np.logspace(np.log10(t_min), np.log10(t_max), t_num)
ft1   = time_domain_function(time = t1)

#==============================================================================
# Here, the numerical Laplace transform of thee time-doman data is performed
#==============================================================================
fs_num = num_lap_trans(time       = t1, 
                       timefunc   = ft1, 
                       s_min      = 0.0001, 
                       s_max      = 10000, 
                       s_num      = 10000)
LapFunc = fs_num[2]
#==============================================================================
# Here, the time-grid for the inverse Laplace transformation of the
# numerically-transformed time-domain solution is generated. Subsequently, the
# numerical inverse Laplace transformation is done
#==============================================================================
t2       = np.logspace(-2,3,1000)
timefunc = gaver_stehfest_inversion(timepoints = t2, LaplaceFunction = LapFunc)
#==============================================================================
# plot the results
#==============================================================================
#------------------------------------------------------------------------------
# modify plot appearance
#------------------------------------------------------------------------------
font = {'family': 'Times New Roman', 'color':  'black','weight': 'normal','size': 15,}
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.sans-serif'] = ['Times new Roman']

plt.figure(figsize = (8,3), dpi = 100)
plt.subplot(121)
#------------------------------------------------------------------------------
# numerical Laplace transformation
#------------------------------------------------------------------------------
plt.plot(np.log10(fs_num[0]), laplace_domain_function(s = fs_num[0]), color = 'red', label = 'analytical')
plt.plot(np.log10(fs_num[0]), fs_num[1], color = 'black', linestyle = ':', label = 'NLT')
plt.xlabel('log$_{10}\,(s\,/\,\mathrm{s}^{-1})$', fontsize = 15)
plt.ylabel('$\overline{m}(s)$', fontsize = 15)
if Type == "M":
    plt.ylabel('$\overline{M}(s)$', fontsize = 15)
plt.tick_params(direction = 'in', length=4, width=0.5, colors='k', labelsize = 15)
plt.legend(frameon = False, fontsize = 15)
#------------------------------------------------------------------------------
# numerical inverse Laplace transformation
#------------------------------------------------------------------------------
plt.subplot(122)
plt.plot(np.log10(t2), time_domain_function(time = t2), color = 'red', label = 'analytical')
plt.plot(np.log10(t2), timefunc, color = 'black', linestyle = ':', label = 'NILT(NLT)')
plt.xlabel('log$_{10}\,(t\,/\,\mathrm{s})$', fontsize = 15)
plt.ylabel('$m(t)$', fontsize = 15)
if Type == "M":
    plt.ylabel('$M(t)$', fontsize = 15)
plt.tick_params(direction = 'in', length=4, width=0.5, colors='k', labelsize = 15)
plt.legend(frameon = False, fontsize = 15)

plt.tight_layout()
plt.show()
'''










