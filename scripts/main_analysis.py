# -*- coding: utf-8 -*-
"""

@author: Alex Tichter
"""
import argparse
import os

from analysis import Analysis

if __name__ == '__main__':
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    parser = argparse.ArgumentParser(description='Douglas-Gunn Analysis expanding in Z')

    parser.add_argument('--electrodestructure', type=str, help='Path to the electrode strcture', default='Pyramid_Data_20_slope1')
    parser.add_argument('--runnameaddition', type=str, help='Name of the run file', default='default_run')
    parser.add_argument('--showplots', type=bool, help='Show the plots', default=False)

    args = parser.parse_args()

    # print the args
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # start the analysis
    analysis = Analysis(base_path, args)
    