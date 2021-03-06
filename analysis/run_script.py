from IPython import embed as shell
import os, sys, datetime, pickle
import subprocess, logging, time
import pp


import scipy as sp
import numpy as np
import matplotlib.pylab as pl
import pandas as pd


shell()

from SmoothPursuitSession import SmoothPursuitSession as SPS
from group_level import HexagonalSaccadeAdaptationGroupLevelAnalyses



# specific for your experiment:
data_folder = '/home/vanes/SA_pirate/data/'
raw_data_folder = '/home/raw_data/SA_pirate/data/'
conditions = {'react_same':1,'react_alt':2}
exp_name = 'SA_pirate'

# subject 'run_array':
sjs_all = [

	# initials, list of edf_files
	['de', ['de_1_2014-02-04_10.13.25.edf','de_1_2014-02-04_10.13.25.edf']],
	['tb', ['tb_1_2014-02-10_15.01.57.edf','tb_1_2014-02-10_15.01.57.edf']],

]

def run_subject(sj, data_folder, do_preps = False, compute_saccades = False, individual_plots = False, exp_name = exp_name, conditions = conditions):
	"""
	This function executes the single-subject analyses.
	"""
	raw_data = [os.path.join(raw_data_folder, sj[conditions[c]]) for c in conditions.keys()]
	aliases  = [c for c in conditions.keys()]
	hsas = Tools.Sessions.HexagonalSaccadeAdaptationSession.HexagonalSaccadeAdaptationSession(subject = sj[0], experiment_name = exp_name, project_directory = data_folder, conditions = conditions)
	if do_preps: 
		hsas.import_raw_data(raw_data, aliases)
		hsas.import_all_data(aliases)
	if compute_saccades:
		for alias in aliases:
			hsas.detect_all_saccades(alias = alias)
			hsas.velocity_for_saccades(alias = alias)
			hsas.gaze_for_saccades(alias = alias)
	if individual_plots:
		hsas.amplitudes_all_adaptation_blocks(measures = ['expanded_amplitude','peak_velocity'])
		
	return True
	
def group_level_analyses(sj, data_folder, exp_name = 'SA_pirate', conditions = {'react_same':1},create_group_lvl_data = False,evaluate_trial_selection=False,fit_and_plot=False):
	"""
	This function executes the group-level analyses.
	"""
	aliases = [c for c in conditions.keys()]
	hsas_gl = HexagonalSaccadeAdaptationGroupLevelAnalyses(subjects = list(np.array(sj)[:,0]), data_folder = data_folder, conditions=aliases,exp_name=exp_name)
	
	if create_group_lvl_data:
		hsas_gl.all_data_to_hdf(aliases)
	if evaluate_trial_selection:
		hsas_gl.create_trial_selection_evaluation_figs(aliases)
	if fit_and_plot:
		hsas_gl.fit_functions_to_individual_blocks(aliases=aliases,plot_individual_blocks=False,plot_averages=False,plot_condition_average=False,baseline_correction=False)

def analyze_subjects(sjs_all, do_preps, compute_saccades, individual_plots):
	"""
	This function has the ability to call run_subjects in parallel.
	"""
	if len(sjs_all) > 1: 
		job_server = pp.Server(ppservers=())
	
		start_time = time.time()
		jobs = [(sj, job_server.submit(run_subject,(sj, data_folder, do_preps, compute_saccades, individual_plots,), (), ("Tools","Tools.Sessions","Tools.Sessions.HexagonalSaccadeAdaptationSession"))) for sj in sjs_all]
		results = []
		for s, job in jobs:
			job()
	
		print "Time elapsed: ", time.time() - start_time, "s"
		job_server.print_stats()
	else:
		run_subject(sjs_all[0], data_folder = data_folder, do_preps = do_preps, compute_saccades = compute_saccades, individual_plots = individual_plots)


def main():
	"""
	This function determines whether the analyze_subjects or group_level_analyses should be run, and with which options
	"""
	analyze_subjects(sjs_all, do_preps = False, compute_saccades = False, individual_plots = False)
	group_level_analyses(sjs_all, data_folder,create_group_lvl_data = False,evaluate_trial_selection=False,fit_and_plot=True)

if __name__ == '__main__':
	main()

# 