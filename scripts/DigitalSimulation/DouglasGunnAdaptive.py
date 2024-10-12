# -*- coding: utf-8 -*-
"""

@author: Tim Tichter, Alex Tichter
"""
# Python Standard Library
import os

# External Libraries
import numpy as np
import json
from tqdm import tqdm

# Internal dependencies
from .lambda_generator import Time_Grid_Parametrizer
from .DouglasGunnLoopFactory import DouglasGunnLoopFactory
from .ThreeDimDerivFactory import ThreeDimDerivFactory

from .douglasGunnAdaptive_utils import initalize_stack_from_file, initialize_coeffs, FirstDerivative_LogGrid, Check_Similarity

class DouglasGunnAdaptive:
    def __init__(self,base_path,args):
        # internal parameters to controle the flow of the program
        self.DEBUG = bool(args.debug)
        print("Debug mode is:\t", self.DEBUG)

        self.JIT_PARALLEL = args.jit_parallel 
        self.JIT_NOPYTHON = args.jit_nopython 

        self.CONTINUERUN = args.continuerun

        if args.np_dtype == 'float32':
            self.DTYPE = np.float32
        elif args.np_dtype == 'float64':
            self.DTYPE = np.float64
        else:
            print("Unknown data type for numpy arrays, please choose one of [float32, float64]")
            exit()

        # setting the parameters of the class
        # refinement parameters
        self.power_of_two = args.n_power_of_two 
        self.last_iter = None

        # general parameters
        self.dx = args.delta_x # delta x
        self.D = args.diffusion_coefficient # diff_coeff
        self.time_max = args.time_max #maximum time
        self.z_grid_expander = args.z_grid_expander #z grid expander

        # time grid parameters
        self.lambda_min = args.lambda_min #minimum lambda
        self.lambda_max = args.lambda_max #maximum lambda

        # similarity parameters
        self.stop_sloperatio = args.stop_ratio_slope #stop at slope ratio
        self.stop_similarity = args.stop_ratio_similarity #stop at similarity
        self.slope_compare_min_time = args.slope_compare_min_time
        
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
        self.array_path = self.output_path + '/arrays'

        # make directory with run name if it does not exist in output path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.exists(self.stack_path):
            os.makedirs(self.stack_path)
        if not os.path.exists(self.array_path):
            os.makedirs(self.array_path)

        if self.CONTINUERUN and self.check_existing_run():
            # check if config exists
            if os.path.exists(self.config_path + '/config.json'):
                with open(self.config_path + '/config.json', 'r') as file:
                    self.config = json.load(file)
            elif self.CONTINUERUN:
                print("No config file found, but continue run flag set")
                exit()
            else:
                # initialize config
                self.config = {}
                self.set_config_values()
                self.save_config(self.config, "config.json")
                

        else:
            # initialize config
            self.config = {}
            self.set_config_values()
            self.save_config(self.config, "config.json")

        # initialize the factories and methods for the main loops
        self.douglas_gunn_loop_factory = DouglasGunnLoopFactory(self.JIT_NOPYTHON,self.JIT_PARALLEL,self.DTYPE)
        self.douglas_gunn_loop_run_function = self.douglas_gunn_loop_factory.return_run_function()

        self.three_dim_deriv_factory = ThreeDimDerivFactory(self.JIT_NOPYTHON,self.JIT_PARALLEL,self.DTYPE)
        self.three_dim_deriv_run_function = self.three_dim_deriv_factory.return_run_function()

        # initalize the stacks to make the explicit
        self.stack_start = None
        self.stack_main = None

    def check_existing_run(self):
        if os.path.exists(self.config_path + '/config.json'):
            return True
        else:
            return False

    def load_refinements(self):
        # load used_times and flux list

        # os walk arrays path and check for n folders
        print(self.array_path + '/*')
        folders = os.listdir(self.array_path + '/')
        print(folders)
        folder_numbers = []
        for folder in folders:
            print(folder)
            if folder.startswith("refinement_"):
                folder_numbers.append(int(folder.split("_")[1]))


        if len(folder_numbers) > 0:
            folder_numbers.sort()
            for folder in folder_numbers:
                self.used_times_list.append(np.load(self.array_path + '/refinement_' + str(folder) + '/used_times.npy'))
                self.flux_list.append(np.load(self.array_path + '/refinement_' + str(folder) + '/flux_list.npy'))

        self.last_iter = self.config["refinement"]["last_iter"]

    def refinement_loop(self):
        self.used_times_list = []
        self.flux_list = []
        if self.CONTINUERUN:
            self.load_refinements()
            print(self.last_iter)
            i = self.last_iter
            # Could happen that no config file was found and stuff gets overwritten, then you want to have a clean start
            if i is not None:
                i += 1 # Start the next iteration
            else:
                i = 0
        else:

            self.config["refinement_stats"] = {}
            i = 0

        while i < self.power_of_two:
            refinement = 2 ** i
            
            self.config["refinement_stats"]["refinement" + str(refinement)] = {}

            if self.DEBUG:
                print("------------------\nrecent refinement\n------------------\n")
                print(refinement)
                print("\n------------------\n")

            self.stack_start, self.stack_main, electrolyte_z_grid, shape_electrode, shape_electrolyte = initalize_stack_from_file(self.electrode_path, self.DTYPE, refinement, self.dx, self.D, self.time_max, self.z_grid_expander)
            
            self.save_stack(self.stack_start, "stack_start.npy")

            self.config["refinement_stats"]["refinement" + str(refinement)]["shape_stack_start"] = self.stack_start.shape
            self.config["refinement_stats"]["refinement" + str(refinement)]["shape_stack_main"] = self.stack_main.shape
            self.config["refinement_stats"]["refinement" + str(refinement)]["shape_electrode"] = shape_electrode
            self.config["refinement_stats"]["refinement" + str(refinement)]["shape_electrolyte"] = shape_electrolyte
            
            coef_universal_dict, coef_vect_r_dict = initialize_coeffs(refinement, self.dx, shape_electrode, electrolyte_z_grid)

            # generate time grid
            if i > 0:
                self.time_max = self.used_times_list[i - 1][-1]
                print("Time max is:\t", self.time_max)
            t, dt, lambdas, t_thresh = Time_Grid_Parametrizer(self.D, self.dx, int(shape_electrode[2]/refinement), refinement, self.lambda_min, self.lambda_max, self.time_max)

            if self.DEBUG:
                print("timesteps = \t",len(t))

            # call the run with early stopping
            flux, used_times, act_sites = self.run(refinement, i, t, dt, lambdas, coef_universal_dict, coef_vect_r_dict)

            flux = np.array(flux)/self.D**0.5
            self.used_times_list.append(used_times)
            self.flux_list.append(flux)

            # save the internal refinement variables
            self.save_array(used_times, 'used_times.npy', refinement)
            self.save_array(flux, 'flux_list.npy', refinement)
            self.save_array(lambdas, 'lambda_list.npy', refinement)
            self.save_array(dt, 'dt_list.npy', refinement)
            self.save_array(t, 't_list.npy', refinement)
            
            self.config["refinement_stats"]["refinement" + str(refinement)]["t_thresh"] = t_thresh
            self.config["refinement_stats"]["refinement" + str(refinement)]["act_sites"] = act_sites

            self.last_iter = i 
            i += 1
            
            self.set_config_values()
            self.save_config(self.config, "config.json")
        
    def run(self,refinement,iter_refine,t,dt,Lambdas,coef_universal_dict, coef_vect_r_dict):
        flux = []
        used_times = []
        interrupt = False

        act_sites = self.three_dim_deriv_run_function(self.stack_start,self.stack_start)
        
        if self.DEBUG:
            print("Number of active sites is:\t", act_sites)
            print("shape of stack_main is:\t", self.stack_main.shape)
            print("dtype of stack_main is:\t", self.stack_main.dtype)

        self.stack_main = np.pad(self.stack_main,((1,1),(1,1),(1,1)),mode='edge')

        if self.DEBUG:
            print("shape of stack_main is:\t", self.stack_main.shape)
            print("dtype of stack_main is:\t", self.stack_main.dtype)

        for i in tqdm(range(len(t))):
            if not interrupt:
                flux.append(self.D*self.three_dim_deriv_run_function(self.stack_main[1:-1:,1:-1:,1:-1:], self.stack_start)/(act_sites*(self.dx/refinement)))
                used_times.append(t[i])
                self.stack_main = self.douglas_gunn_loop_run_function(self.stack_main, self.stack_start, Lambdas[i], self.D, dt[i], coef_universal_dict["LD"], coef_universal_dict["MD"], coef_universal_dict["UD"], coef_vect_r_dict["alpha"], coef_vect_r_dict["beta"], coef_vect_r_dict["gamma"])
            else:
                break

            if i%1000 == 0 and i > 0 and interrupt == False:
                if iter_refine == 0:
                    if used_times[-1] > self.slope_compare_min_time:
                        LogDer_recent = FirstDerivative_LogGrid(xdata = np.array(used_times[-10::]), ydata = np.array(flux[-10::]))
                        if np.min(np.abs(LogDer_recent[1]+0.5)/0.5) <= (100-self.stop_sloperatio)/100:
                            interrupt = True
                if iter_refine > 0: 
                    SimThresh = Check_Similarity(x_data       = self.used_times_list[iter_refine-1], 
                                                 y_data       = self.flux_list[iter_refine-1], 
                                                 checkpoint_x = used_times[-1], 
                                                 checkpoint_y = flux[-1]/self.D**0.5) # normalise the flux before the comparison
                    if SimThresh <= (100 - self.stop_similarity):
                        interrupt = True
        return flux, used_times, act_sites

    # Section for controlling the serialization of the data
    def save_array(self, array, name, refinement=None):
        try:
            if refinement is None:
                path = self.array_path + '/' 
            else:
                path = self.array_path + '/' + "refinement_" + str(refinement) + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            np.save(path + name, np.array(array))

            # save as text file
            np.savetxt(path + name + '.txt', array, delimiter='\t')
        except Exception as e:
            print("Error in saving array" + name + " in refinement " + str(refinement))
            print(e)

    def save_stack(self, stack, name):
        np.save(self.stack_path + '/' + name, stack)
    
    def save_config(self, config, name):
        with open(self.config_path + '/' + name, 'w') as file:
            json.dump(config, file)
    
    def set_config_values(self):
        self.config["refinement"] = {}
        self.config["refinement"]["power_of_two"] = self.power_of_two
        self.config["refinement"]["last_iter"] = self.last_iter
        
        self.config["general"] = {}
        self.config["general"]["dx"] = self.dx
        self.config["general"]["D"] = self.D
        self.config["general"]["time_max"] = self.time_max
        self.config["general"]["z_grid_expander"] = self.z_grid_expander
        
        self.config["timegrid"] = {}
        self.config["timegrid"]["lambda_min"] = self.lambda_min
        self.config["timegrid"]["lambda_max"] = self.lambda_max

        self.config["similarity"] = {}
        self.config["similarity"]["stop_sloperatio"] = self.stop_sloperatio
        self.config["similarity"]["stop_similarity"] = self.stop_similarity

        self.config["stack"] = {}
        self.config["stack"]["electrode_structure"] = self.electrode_structure

        self.config["paths"] = {}
        self.config["paths"]["run_name"] = self.run_name
        self.config["paths"]["run_name_addition"] = self.run_name_addition
        self.config["paths"]["base_bath"] = self.base_bath
        self.config["paths"]["data_path"] = self.data_path
        self.config["paths"]["electrode_path"] = self.electrode_path
        self.config["paths"]["output_path"] = self.output_path
        self.config["paths"]["config_path"] = self.config_path
        self.config["paths"]["stack_path"] = self.stack_path
        self.config["paths"]["array_path"] = self.array_path