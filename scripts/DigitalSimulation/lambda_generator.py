# -*- coding: utf-8 -*-
"""


@author: Tim Tichter
"""

import   numpy                  as     np
# import   matplotlib.pyplot      as     plt
from     scipy.special          import erfinv
#==============================================================================
# Declare fonts for Plot appearance
#==============================================================================
# font = {'family': 'Times New Roman', 'color':  'black','weight': 'normal','size': 15,}
# plt.rcParams['mathtext.fontset'] = 'dejavuserif'
# plt.rcParams['font.sans-serif'] = ['Times new Roman']
#==============================================================================
# The following function is used to create a sigmoidal grid of Lambda-values
# it is Lambda = D*dt/(Voxelfactor*dx)^2
# Typically, Lambda <= 0.5 for stable simulations. However, numerical 
# experimentation (and literature) have shown that for Cottrellian expeiments
# large Lambda-values can be used (up to Lamda = 500). To make it more general,
# we call the largest Lambda Lambda_max
# Generally, Lambda values are used to compute the dt values for a given set of 
# dx, Voxelfactor and D. This allows to modify the grid-spacing during the 
# simulations while preserving the numerical accuracy (by keeping Lambda
# below a certain threshold).
# The following function was derived.... 
#       - It keeps Lambda = Lambda_0 at the 
#         initial timestep. 
#       - It keeps Lambda < 0.5 below a certain time, t_s 
#         which can be specified. Later on, this time is set to the point in 
#         time at which a semi-infinite diffusion profile at a distance of 
#         three stack heights will be 99% of the bulk value. 
#       - The Lambda_max will be set to 10 later on
#==============================================================================
def Lambda_Generator(Lambda_0, Lambda_max, t_ini, t_s, t_recent):
    Lambda_max = Lambda_max - Lambda_0
    beta       =  1/(t_s - t_ini)*np.log((Lambda_0*(1 - Lambda_0) 
                    - 0.5*Lambda_max)/(Lambda_0*(0.5 - Lambda_max)))
    if beta*(t_recent - t_ini) <= 100:    
        f_t_beta =  Lambda_0*(np.exp(beta*(t_recent - t_ini)) - 1)
    if beta*(t_recent - t_ini) > 100:  
        f_t_beta = Lambda_0*(2.6881171418161356e+43 - 1)
    Lambda  = Lambda_0 + (Lambda_max*f_t_beta)/( Lambda_max +  f_t_beta  )
    return Lambda
###############################################################################
###############################################################################
# Plug the lambda-generator into a function
###############################################################################
###############################################################################
def Time_Grid_Parametrizer(D, dx, zlen, Vox, MinLambda, MaxLambda, Max_time):
    #==============================================================================
    # At next, the time-grid of a REVOXELING stack as well as the Lambda-Grid 
    # will be initialized and defined
    #==============================================================================
    t          = []
    dt         = []
    Lambdas    = []
    Last_t     = 0
    i          = 0
    t_thresh   = ((zlen*dx)**2/(4*D*(erfinv(0.1)**2)))
    #------------------------------------------------------
    # according to the definition of Lambda,
    # the smallest (first) time increment
    # is defined here
    #------------------------------------------------------
    t_initial = (MinLambda/D)*(dx/Vox)**2
    #------------------------------------------------------
    # now, start the lambda-generating loop
    #------------------------------------------------------
    while Last_t <= Max_time:
        if i == 0:
            dt.append((MinLambda/D)*(dx/Vox)**2)
            t.append((MinLambda/D)*(dx/Vox)**2)
            Lambdas.append(MinLambda)
            Last_t = t[-1]
        if i > 0:
            ll       = Lambda_Generator(Lambda_0   = MinLambda, Lambda_max = MaxLambda, 
                                        t_ini      = t_initial, t_s        = t_thresh, 
                                        t_recent   = Last_t)
            Lambdas.append(ll)
            dt.append( (ll/D)*(dx/Vox)**2)
            Last_t   = t[-1] + dt[-1]
            t.append(Last_t)
        i += 1
    return np.array(t), np.array(dt), np.array(Lambdas), t_thresh
