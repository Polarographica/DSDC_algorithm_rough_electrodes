import os

from scripts.analysis import ConvolutionStep

# make sure to have run the main_analysis.py script before running this script to ensure that the mass-transfer function 
# has been generated (and the M_t_times.py and M_t_values.py files are present in the data/output/runs/ folder)

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# replace them if needed
electrode_structure = '001_Planar_stack_8x8' 
run_name_addition = 'example_planar_99.99'

run_name = f"DGA_{electrode_structure}_{run_name_addition}"

plotpath = base_path + '/data/output/runs/' + run_name + '/plots/'

# below you can edit the parameters of the CV
cv_dict = cv_dict = { 'E_in': -0.4, 
                      'E_up': 0.4, 
                      'E_fin': -0.4, 
                      'n': 1, 
                      'R': 8.314, 
                      'T': 298, 
                      'F': 96485, 
                      'Diff_co': 1e-6, 
                      'alpha': 0.5, 
                      'Scanrate': 0.01, 
                      'Lambda': 15}

cs = ConvolutionStep(plotpath=plotpath,cv_dict=cv_dict)
cs.run()