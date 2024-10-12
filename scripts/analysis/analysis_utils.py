# -*- coding: utf-8 -*-
"""

@author: Tim Tichter, Alex Tichter
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

#=================================================================================================================
# The following function is used to eliminate a potential double-definition of some points which may occur 
# during concatenation. This needs to be done, in order to provide a strongly-monotonically ascending abscissia
# for InterpolatedUnivariateSpline.
#=================================================================================================================
def eliminate_equalities_on_abscissia(x_data, y_data):
    x_out = []
    y_out = []
    for i in range(len(x_data)):
        if i == 0:
            x_out.append(x_data[i])
            y_out.append(y_data[i])
        if i > 0:
            if x_data[i] >x_out[-1]:
                x_out.append(x_data[i])
                y_out.append(y_data[i])
    x_out = np.array(x_out)
    y_out = np.array(y_out)
    return x_out, y_out

#=================================================================================================================
# The following function is used to compute a first derivative on a double-logarithmic grid of two non-logarithmic
# inputs. The accuracy of the derivative is O h^2 and asymmetrical three-point finite differences are used at the 
# border-points.
#=================================================================================================================
def FirstDerivative_LogGrid(xdata, ydata):
    x_data  = np.log10(xdata)
    y_data  = np.log10(ydata)
    interp  = InterpolatedUnivariateSpline(x_data, y_data, k = 2)
    x_new   = np.logspace(x_data[0], x_data[-1], 1000)
    y_new   = interp(np.log10(x_new))
    x_data  = np.log10(x_new)
    y_data  = y_new
    delta_x = x_data[1] - x_data[0]
    if len(x_data) == len(y_data):
        Derimat = np.zeros((len(x_data),len(x_data))) 
        Derimat[0,0]  , Derimat[0,1]  , Derimat[0,2]    = -3, 4, -1
        Derimat[-1,-1], Derimat[-1,-2], Derimat[-1,-3]  =  3,-4, 1
        for i in range(len(x_data)):
            for j in range(len(y_data)):
                if i > 0 and i < (len(x_data)-1) :
                    if j == i-1:
                        Derimat[i,j] = -1
                    if j == i+1:
                        Derimat[i,j] = 1
                    if j == i:
                        Derimat[i,j] = 0
        Derimat = Derimat/(2*delta_x )
    else:
        print("Error! x and y must have the same length!")
    return x_data, np.dot(Derimat, y_data)