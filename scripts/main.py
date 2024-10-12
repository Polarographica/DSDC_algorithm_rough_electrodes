# -*- coding: utf-8 -*-
"""
DOUGLAS-GUNN EXPANDING IN Z

@author: Tim Tichter, Alex Tichter
"""

import argparse
import os

from DigitalSimulation import DouglasGunnAdaptive

# Argument Parser, to set the parameters of Douglas Gunn expanding in Z
if __name__ == '__main__':
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
    
    parser = argparse.ArgumentParser(description='Douglas-Gunn expanding in Z')

    # Arguments for the new Douglas Gunn algorithm
    parser.add_argument('--electrodestructure', type=str, help='Path to the electrode strcture', default='Pyramid_Data_20_slope1')
    parser.add_argument('--runnameaddition', type=str, help='Name of the run file', default='default_run')

    parser.add_argument('--n_power_of_two', type=int, help='Number of power of two', default=1)
    
    parser.add_argument('--delta_x', type=float, help='delta x', default=1e-5)
    parser.add_argument('--diffusion_coefficient', type=float, help='diffusion coefficient', default=1e-6)
    parser.add_argument('--time_max', type=float, help='Max time', default=100)
    parser.add_argument('--z_grid_expander', type=float, help='Z grid expander', default=0.05)
    
    parser.add_argument('--lambda_min', type=float, help='Min lambda', default=0.1)
    parser.add_argument('--lambda_max', type=float, help='Max lambda', default=10)

    parser.add_argument('--stop_ratio_slope', type=float, help='Stop ratio slope', default=99.99)
    parser.add_argument('--stop_ratio_similarity', type=float, help='Stop ratio similarity', default=99.99)
    parser.add_argument('--slope_compare_min_time', type=int, help='Treshold of the last value in last_times for checking the similarity', default=0.1)
    
    parser.add_argument('--np_dtype', type=str, help='Numpy data type for stacks choose one [float32,float64]', default='float64')
    parser.add_argument('--jit_parallel', type=bool, help='Use jit parallel', default=False)
    parser.add_argument('--jit_nopython', type=bool, help='Use jit nopython', default=True)

    parser.add_argument('--continuerun', type=bool, help='Continue the run', default=False)

    parser.add_argument('--debug', type=bool, help='Debug mode', default=False)

    args = parser.parse_args()

    # print the args
    for arg in vars(args):
        print(arg, getattr(args, arg))

    dga = DouglasGunnAdaptive(base_path, args)
    dga.refinement_loop()
    
