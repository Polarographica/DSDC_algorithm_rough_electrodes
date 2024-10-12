# -*- coding: utf-8 -*-
"""

@author: Tim Tichter, Alex Tichter
"""
import os

import numpy as np
from skimage import io
from scipy.special import erfinv
from scipy.interpolate import InterpolatedUnivariateSpline

# SECTION COEFFICIENTS
def initialize_coeffs(voxel_factor, dx, shape_electrode, electrolyte_z_grid):
    coef_universal_dict = {}
    coef_vect_r_dict = {}

    dz_initial = dx/voxel_factor
    z_grid_electrode = np.arange(0, shape_electrode[2]*dz_initial, dz_initial)
    z_grid_total = np.concatenate((z_grid_electrode, electrolyte_z_grid))
    print("len of the z-grid = \t", len(z_grid_total))
    #----------------------------------------------------------------------
    # compute the finite difference coefficients of the
    # (arbitrarily spaced) z-grid
    #----------------------------------------------------------------------
    coeffs_z_alpha, coeffs_z_beta, coeffs_z_gamma  = grid_coeff_generator(z_grid_total)

    #----------------------------------------------------------------------
    # re-shape the alpha, beta and gamma vectors for making them compatible with
    # the grid-stack multiplication
    #----------------------------------------------------------------------
    vect_r_alpha   = coeffs_z_alpha.reshape(1,1,len(coeffs_z_alpha))
    vect_r_beta   = coeffs_z_beta.reshape(1,1,len(coeffs_z_beta))
    vect_r_gamma   = coeffs_z_gamma.reshape(1,1,len(coeffs_z_gamma))
    #----------------------------------------------------------------------
    # since the z-grid has the same expansion at each x,y position (all 
    # z-columns look same) it is sufficient to compute the diagonals of 
    # the diffusion matrix once and to adjust them according to the 
    # t-stepping and the internal boundaries later on. This is done by
    # the Matbuilder_Z function later on in the Douglas-Gunn loop
    #----------------------------------------------------------------------
    universal_LD        = -coeffs_z_alpha[1::]
    universal_MD        = -coeffs_z_beta[::]
    universal_MD[0]    += -coeffs_z_alpha[0]
    universal_MD[-1]   += -coeffs_z_gamma[-1]
    universal_UD        = -coeffs_z_gamma[:-1:]

    coef_vect_r_dict['alpha'] = vect_r_alpha
    coef_vect_r_dict['beta'] = vect_r_beta
    coef_vect_r_dict['gamma'] = vect_r_gamma

    coef_universal_dict['LD'] = universal_LD
    coef_universal_dict['MD'] = universal_MD
    coef_universal_dict['UD'] = universal_UD

    return coef_universal_dict, coef_vect_r_dict

#=================================================================================================================
# Since we aim for an arbitrarily spaced grid in the z-dimension, we need to compute the finite difference
# coefficients for this particular grid for each and every point. However, since any column in z (so standing on 
# the pair x_i, y_j) has the same width in z, it is better to not re-compute the coefficients in every time-loop
# but to rather compute them once and store them in a look-up table (array(s)) for later use. This is done here.
# Note, that along the z-dimension we would need a tridiagonal matrix for the computation. However, instead of
# storing a dense matrix, it is better to just store the diagonals. This is done here. The counkting in z is k 
# x_i, y_j, z_k. So the main diagonal is called k_line. The off-diagonals are the k_minus_one_line and the
# k_plus_one_line.
#=================================================================================================================
def grid_coeff_generator(z_grid):
    alpha_vect   = np.zeros(len(z_grid))
    beta_vect    = np.zeros(len(z_grid))
    gamma_vect   = np.zeros(len(z_grid))
    
    for i in range(len(z_grid)):
        if i == 0:
            dz2_n     = z_grid[i+1]-z_grid[i]
            dz1_n     = dz2_n
        elif i == len(z_grid)-1:
            dz1_n     = z_grid[i]-z_grid[i-1]
            dz2_n     = dz1_n   
        else:
            dz1_n     = z_grid[i]-z_grid[i-1]
            dz2_n     = z_grid[i+1]-z_grid[i]
        
        matLine   = fd_coeffs_secDerCentral_arbGrid_z(dz1 = dz1_n, dz2 = dz2_n)
        
        alpha_vect[i] = matLine[0]
        beta_vect[i]  = matLine[1] 
        gamma_vect[i] = matLine[2]

    return alpha_vect, beta_vect, gamma_vect

#=================================================================================================================
# The following function is used to generate the finite difference coefficients for a second derivative on an
# arbitrarily spaced grid. This is done by the procedure found in the book by D. Britz and J. Strutwolf.
# "Digital Simulation in Electrochemistry", fourth edition, Springer, Chapter 3.8 p. 51-54.
#=================================================================================================================

def fd_coeffs_secDerCentral_arbGrid_z(dz1, dz2):
    A      = np.zeros((2,2))
    A[0,0] = -dz1
    A[0,1] =  (dz1**2)/2
    A[1,0] =  dz2
    A[1,1] =  (dz2**2)/2

    A_inv  = np.linalg.inv(A)
    
    sd_col = A_inv[1,::]
    
    c1     = sd_col[0]
    c2     = -(sd_col[0]+sd_col[1])
    c3     = sd_col[1]

    return c1, c2, c3


# SECTION STACKS
# SECTION UPVOXELER
def up_voxeler_function(arr, voxel_factor, dtype = np.float32):
    arr_up_pixeld   = np.zeros((voxel_factor*arr.shape[0], voxel_factor*arr.shape[1], voxel_factor*arr.shape[2]),dtype)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            for k in range(arr.shape[2]):
                sub_array = arr[i,j,k]*np.ones((voxel_factor, voxel_factor, voxel_factor),dtype)
                arr_up_pixeld[voxel_factor*i:voxel_factor*(i+1),voxel_factor*j:voxel_factor*(j+1), voxel_factor*k:voxel_factor*(k+1)]  = sub_array
    return arr_up_pixeld

def initalize_stack_from_file(path, dtype, voxel_factor, dx, D, time_max, z_grid_expander):
    stack_electrode = read_electrode_structure(path,dtype) 
    
    shape_electrode = stack_electrode.shape
        
    electrolyte_z_grid = initialize_z_grid(voxel_factor, dx, D, time_max, z_grid_expander) 
    
    electrolyte_z_grid += dx*(shape_electrode[2]) 
    
    print("Voxel factor is:\t", voxel_factor)
    if voxel_factor > 1:
        print("Voxel factor is larger than 1")
        stack_electrode = up_voxeler_function(stack_electrode, voxel_factor, dtype)
        shape_electrode = stack_electrode.shape
    

    stack_electrolyte = np.ones((stack_electrode.shape[0],stack_electrode.shape[1],len(electrolyte_z_grid)),dtype)
    print(stack_electrolyte.dtype)
    shape_electrolyte = stack_electrolyte.shape
    
    stack_start = np.concatenate((stack_electrode, stack_electrolyte), axis=2)

    stack_main = stack_start.copy()

    return stack_start, stack_main, electrolyte_z_grid, shape_electrode, shape_electrolyte



def initialize_z_grid(voxel_factor, dx, D, time_max, z_grid_expander):
        #--------------------------------------------------------------------------
        # find thee point in space at which the concentration profile will be
        # 99% of the bulk-value at the maximum time of the experiment.
        # Until this point, diffusion is assumed to be semi-infinite
        # The concentration profile of a planar semi-infinite diffusion
        # domain is given by c(x,t) = c_0*erf(x/(2*\sqrt{D*t})) 
        # Setting c(x,t)/c_0 = 0.99 one arrives at:
        # x = 2*erfinv(0.99)*\sqrt{Dt}
        #--------------------------------------------------------------------------
        end_x         = 2*erfinv(0.99)*(D*time_max)**0.5
        print("x-max of semi-infinite diffusion in cm = %.5f" %end_x)
        #--------------------------------------------------------------------------
        # Now, initialite the electrolyte grid
        #--------------------------------------------------------------------------
        electrolyte_grid = [0]
        u                = 0
        while electrolyte_grid[-1] <= end_x:
            dx_recent = (dx/voxel_factor)*np.exp(z_grid_expander*u)
            electrolyte_grid.append(electrolyte_grid[-1] + dx_recent)
            u += 1
        electrolyte_grid = np.array(electrolyte_grid)
        return electrolyte_grid

def read_electrode_structure(path, dtype):
        
        electrode_file_names = []

        # check if path exists
        if not os.path.exists(path):
            print('Path to the electrode structure does not exist')
            exit()
        
        # read the electrode structure
        for img in os.listdir(path):
            if img.endswith('.tif'):
                electrode_file_names.append(img)

        # check if there are any files
        if len(electrode_file_names) == 0:
            print('No files in the electrode structure folder')
            exit()
        
        first_image = io.imread(path + '/' + electrode_file_names[0])

        dimx, dimy = first_image.shape[:2]
        dimz = len(electrode_file_names)

        stack_electrode = np.zeros((dimx, dimy, dimz),dtype)
        
        for i in range(dimz):
            stack_electrode[:,:,i] = io.imread(path + '/' + electrode_file_names[i])[::,::,0]

        #--------------------------------------------------------------------------
        # subsequently to loading the images with pixel-color 255 or 0, define
        # what it electrode material and what is electrolyte. For this purpose, 
        # the contrast needs to be inverted.
        #--------------------------------------------------------------------------
        stack_electrode = stack_electrode[::,::,::] / 255.0
        stack_electrode = (stack_electrode - 1)**2
        print("Shape of the loaded electrode stack:\t", stack_electrode.shape)
        
        return stack_electrode

# SECTION SIMILARITY UTILS UTILS
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

#=================================================================================================================
# The following function is used to check the similarity of two computation results in a given interval.
#=================================================================================================================
def Check_Similarity(x_data, y_data, checkpoint_x, checkpoint_y):
    x_data   = np.log10(x_data)
    y_data   = np.log10(y_data)
    interp   = InterpolatedUnivariateSpline(x_data, y_data, k = 2)
    y_interp = interp(np.log10(checkpoint_x))
    y_true   = 10**y_interp
    # # print all the variables
    # print("x_data = ", x_data)
    # print("y_data = ", y_data)
    # print("checkpoint_x = ", checkpoint_x)
    # print("checkpoint_y = ", checkpoint_y)
    # print("y_interp = ", y_interp)
    # print("y_true = ", y_true)
    # print(np.abs((y_true/checkpoint_y)))
    # print(np.abs((y_true/checkpoint_y - 1)))


    return np.abs((y_true/checkpoint_y - 1)*100)

