# -*- coding: utf-8 -*-
"""


@author: Alex Tichter
"""
import os
import json

import numpy as np

from .FluxReconstruction import FluxReconstruction
from .ConvolutionStep import ConvolutionStep
from .DeconvolutionStep import DeconvolutionStep

class Analysis:
    def __init__(self, base_path, args):
        self.args = args
        
        self.SHOWPLOTS = self.args.showplots
        
        self.D = 1e-6
        self.Cutoff_time = 1e-6
        self.Skip_time_decades = 1
        self.MinLambda = 0.1
        self.MaxLambda = 10
        self.Display_Sec_Ordinate = False

        # stack parameters
        self.electrode_structure = args.electrodestructure #name of the folder with the electrode structure

        # initializing paths and filenames
        self.run_name_addition = args.runnameaddition #name of the run file
        self.base_bath = base_path
        self.data_path = base_path + '/data'
        self.electrode_path = self.data_path + '/electrode_structures/' + self.electrode_structure

        self.run_name = f"DGA_{self.electrode_structure}_{self.run_name_addition}"
        
        self.output_path = self.data_path + '/output/runs/' + self.run_name
        self.config_path = self.output_path
        self.stack_path = self.output_path + '/stacks'
        self.array_path = self.output_path + '/arrays/'
        self.plot_path = self.output_path + '/plots/'

        # create the output directory if it does not exist
        # other paths should exist due to the DouglasGunn run, otherwise it cant be analyzed
        if not os.path.exists(self.plot_path):
            os.makedirs(self.plot_path)
        
        # Internal variables
        self.time_list = []
        self.used_times_list = []
        self.flux_list = []
        self.t_thresh_list = []
        self.lambda_list = []
        self.dt_list = []

        self.config = None

        self.load_config()
        self.load_data()
        self.run()

    def run(self):
        print('Running analysis')
        print('Electrode structure:', self.electrode_structure)

        flux_reconstruction = FluxReconstruction(self.Cutoff_time, self.Skip_time_decades, self.MinLambda, self.MaxLambda, self.time_list, self.used_times_list, 
                                                 self.flux_list, self.t_thresh_list, self.lambda_list, self.dt_list, self.plot_path, self.SHOWPLOTS)
        flux_reconstruction.run()
        print('Flux reconstruction done')
        deconvolution_step = DeconvolutionStep(self.plot_path, showplots=self.SHOWPLOTS)
        deconvolution_step.run()
        print('Deconvolution done')
        convolution_step = ConvolutionStep(self.plot_path, showplots=self.SHOWPLOTS)
        convolution_step.run()

        print('Analysis done')

    def load_data(self):
        for i in range(self.n_power_of_two):
            path =  self.array_path + 'refinement_' + str(2**i) + '/'
            print(path)

            self.time_list.append(np.load(path + 't_list.npy'))
            self.used_times_list.append(np.load(path + 'used_times.npy'))
            self.flux_list.append(np.load(path + 'flux_list.npy'))
            self.lambda_list.append(np.load(path + 'lambda_list.npy'))
            self.dt_list.append(np.load(path + 'dt_list.npy'))

            self.t_thresh_list.append(self.config['refinement_stats']['refinement' + str(2**i)]['t_thresh'])


    def load_config(self):
        self.config = json.load(open(self.config_path + '/config.json'))
        self.n_power_of_two = self.config["refinement"]["last_iter"] + 1

