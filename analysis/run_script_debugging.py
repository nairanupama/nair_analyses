from IPython import embed as shell
import os, sys, datetime, pickle
import subprocess, logging, time
import pp

import scipy as sp
import numpy as np
import matplotlib.pylab as pl
import pandas as pd

from SmoothPursuitSession import SmoothPursuitSession as SPS
#from group_level import HexagonalSaccadeAdaptationGroupLevelAnalyses

<<<<<<< HEAD
# specific for your experiment:
data_folder = 'aeneas_home/nair_analyses/data_folder'
raw_data_folder = 'aeneas_home/nair_analyses/raw_data_folder'
=======

# specific for your experiment:
data_folder = 'home/nair_analyses'
raw_data_folder = 'home/nair_analyses/an_5_2016-02-04_10.13.25.edf.edf'
>>>>>>> ac54774a1256c2040ad0fa75d260fa2c6f536193
conditions = {'key':1}
exp_name = 'smooth_pursuit'

# subject 'run_array':
sjs_all = [

	# initials, list of edf_filesS
<<<<<<< HEAD
	['an', 'an_5_2016-02-04_10.13.25.edf']
=======
	['an', ['an_5_2016-02-04_10.13.25.edf.edf']]
>>>>>>> ac54774a1256c2040ad0fa75d260fa2c6f536193
	# ['tb', ['tb_1_2014-02-10_15.01.57.edf','tb_1_2014-02-10_15.01.57.edf']],

]

def run_subject(sj, data_folder, do_preps = False, exp_name = exp_name):
	"""
	This function executes the single-subject analyses.
	"""
<<<<<<< HEAD
	raw_data = [os.path.join(raw_data_folder, sj[0], sj[conditions[c]]) for c in conditions.keys()]	
	#initials = sj[0]
	
	aliases  = [c for c in conditions.keys()]

	shell()
	hsas = SPS(subject = sj[0], experiment_name = exp_name, project_directory = data_folder,
		conditions = conditions)

	if do_preps: 
		hsas.import_raw_data(raw_data, aliases)
		hsas.import_all_data(aliases)


=======
	raw_data = [os.path.join(raw_data_folder, sj[0], sj[conditions[c]]) for c in conditions.keys()]
	aliases  = [c for c in conditions.keys()]
	hsas = SPS(subject = sj[0], experiment_name = exp_name, project_directory = data_folder, conditions = conditions)
	if do_preps: 
		hsas.import_raw_data(raw_data, aliases)
		hsas.import_all_data(aliases)
>>>>>>> ac54774a1256c2040ad0fa75d260fa2c6f536193
	#if compute_saccades:
	#	for alias in aliases:
	#		hsas.detect_all_saccades(alias = alias)
	#		hsas.velocity_for_saccades(alias = alias)
	#		hsas.gaze_for_saccades(alias = alias)
	#if individual_plots:
	#	hsas.amplitudes_all_adaptation_blocks(measures = ['expanded_amplitude','peak_velocity'])
		
	return True

	
#def group_level_analyses(sj, data_folder, exp_name = 'SA_pirate', conditions = {'react_same':1},create_group_lvl_data = False,evaluate_trial_selection=False,fit_and_plot=False):
#	"""
#	This function executes the group-level analyses.
#	"""
#	aliases = [c for c in conditions.keys()]
#	hsas_gl = HexagonalSaccadeAdaptationGroupLevelAnalyses(subjects = list(np.array(sj)[:,0]), data_folder = data_folder, conditions=aliases,exp_name=exp_name)
	
#	if create_group_lvl_data:
#		hsas_gl.all_data_to_hdf(aliases)
#	if evaluate_trial_selection:
#		hsas_gl.create_trial_selection_evaluation_figs(aliases)
#	if fit_and_plot:
#		hsas_gl.fit_functions_to_individual_blocks(aliases=aliases,plot_individual_blocks=False,plot_averages=False,plot_condition_average=False,baseline_correction=False)

def analyze_subjects(sjs_all, do_preps):
	"""
	This function has the ability to call run_subjects in parallel.
	"""
	if len(sjs_all) > 1: 
		job_server = pp.Server(ppservers=())
	
		start_time = time.time()
<<<<<<< HEAD
		jobs = [(sj, job_server.submit(run_subject,(sj, data_folder, do_preps, individual_plots),(), ("SPS"))) for sj in sjs_all]
=======
		jobs = [(sj, job_server.submit(run_subject,(sj, data_folder, do_preps),(), ("SPS"))) for sj in sjs_all]
>>>>>>> ac54774a1256c2040ad0fa75d260fa2c6f536193
		results = []
		for s, job in jobs:
			job()
	
		print "Time elapsed: ", time.time() - start_time, "s"
		job_server.print_stats()
	else:
<<<<<<< HEAD
		run_subject(sj=sjs_all[0], data_folder = data_folder, do_preps = do_preps)


=======
		run_subject(sjs_all[0], data_folder = data_folder, do_preps = do_preps)
>>>>>>> ac54774a1256c2040ad0fa75d260fa2c6f536193


def main():
	"""
	This function determines whether the analyze_subjects or group_level_analyses should be run, and with which options
	"""
	analyze_subjects(sjs_all, do_preps = True)
	#group_level_analyses(sjs_all, data_folder,create_group_lvl_data = False,evaluate_trial_selection=False,fit_and_plot=True)
<<<<<<< HEAD
	print "hello"
=======
	#print "hello"
>>>>>>> ac54774a1256c2040ad0fa75d260fa2c6f536193

if __name__ == '__main__':
	main()

