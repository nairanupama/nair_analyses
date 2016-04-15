#!/usr/bin/env python

# encoding: utf-8

"""

EyeLinkSession.py



Created by Tomas Knapen on 2011-04-27.

Copyright (c) 2011 __MyCompanyName__. All rights reserved.

"""



import os, math



import numpy as np

import scipy as sp

import scipy.stats as stats

import matplotlib.pylab as pl

import pandas as pd

import numpy.linalg as LA

import bottleneck as bn

from scipy.optimize import curve_fit

from scipy import stats, polyval, polyfit

from lmfit import minimize, Parameters, Parameter, report_fit



from joblib import Parallel, delayed

import itertools

from itertools import chain

import seaborn as sns





import logging, logging.handlers, logging.config

# sys.path.append( os.environ['ANALYSIS_HOME'] )

from Tools.log import *

from Tools.Operators import EDFOperator, HDFEyeOperator, EyeSignalOperator

from Tools.Operators.EyeSignalOperator import detect_saccade_from_data

from Tools.Operators.CommandLineOperator import ExecCommandLine

from Tools.other_scripts.plotting_tools import *

from Tools.other_scripts.circularTools import *

#from Tools.Sessions.HexagonalSaccadeAdaptationSession import  HexagonalSaccadeAdaptationSession



from IPython import embed as shell
	

def nan_helper(y):

	return np.isnan(y), lambda z: z.nonzero()[0]



def fit_lin(data):

	"""

	Fits a linear function to input using polyfit and returns:

	[0]: intercept

	[1]: slope

	[2]: resulting function with length of input	

	[3]: RSS 

	[4]: AIC 

	"""

	n = len(data)

	k = 2

	(ar,br)=polyfit(np.arange(n),data,1)

	xr = polyval([ar,br],np.arange(n))

	RSS = np.sum((data - xr)**2)

	if n/k > 40:

		AIC = n*np.log(RSS/n) + 2*k

	else:

		AIC = n*np.log(RSS/n) + 2*k + ((2*k)**2 + 2*k) / (n-k-1)

	

	return ar, br, xr, RSS, AIC



def fit_exp(data):

	"""

	Fits an exponential function to input using lmfit and returns:

	[0]: intercept

	[1]: gain ('stretch')

	[2]: slope ('shape')

	[3]: resulting function with length of input

	[4]: RSS

	[5]: AIC 

	"""

	n = len(data)

	k = 3

	params=Parameters()	

	params.add('slope',value=0.1)

	params.add('gain',value=1.0)

	params.add('intercept',value=1.0)

	x = np.arange(n)

	def fit_func(params, x, data):

		fit = params['gain'].value * np.exp(x * - params['slope'].value) - params['intercept'].value

		return fit - data

	result = minimize(fit_func, params, args=(x,data), method = 'leastsq')	

	final = data + result.residual

	RSS = np.sum((data-final)**2)

	if n/k > 40:

		AIC = n*np.log(RSS/n) + 2*k

	else:

		AIC = n*np.log(RSS/n) + 2*k + ((2*k)**2 + 2*k) / (n-k-1)

	

	return result.params['intercept'], result.params['gain'],result.params['slope'], final, RSS, AIC



def fit_pow(data):

	"""

	Fits a power function to input using polyfit and returns:

	[0]: intercept

	[1]: slope

	[2]: resulting function with length of input	

	[3]: RSS

	[4]: AIC

	"""

	n = len(data)

	k = 2

	(ar,br)=polyfit(np.log10(np.arange(1,n+1)),np.log10(data),1)

	xr=polyval([ar,br],np.log10(np.arange(1,n+1)))

	RSS = np.sum((data - 10**xr)**2)

	if n/k > 40:

		AIC = n*np.log(RSS/n) + 2*k

	else:

		AIC = n*np.log(RSS/n) + 2*k + ((2*k)**2 + 2*k) / (n-k-1)

	

	return 10**ar,10**br,10**xr, RSS, AIC



class HexagonalSaccadeAdaptationGroupLevelAnalyses(object): #dont need this one

	"""

	Instances of this class can be used to execute group level analyses for Hexagonal Saccade Adaptation experiments

	"""

	def __init__(self, subjects, data_folder, conditions, exp_name,loggingLevel = logging.DEBUG ):

		self.experiment_name = exp_name

		self.subjects = subjects

		self.data_dir = data_folder

		self.plot_dir = os.path.join(self.data_dir, self.experiment_name , 'group_level','figs')

		self.conditions = conditions

		self.initials = [s.initials for s in self.subjects]

		self.nr_trials_per_block = 150

		self.nr_blocks = 8

		try:

			os.mkdir(os.path.join( data_folder, self.experiment_name, 'group_level'))

			os.mkdir(os.path.join( data_folder, self.experiment_name, 'group_level','data'))

			os.mkdir(os.path.join( data_folder, self.experiment_name, 'group_level','figs' ))

			os.mkdir(os.path.join( data_folder, self.experiment_name, 'group_level','log' ))

		except OSError:

			pass

		self.group_lvl_data_directory = os.path.join( data_folder, self.experiment_name, 'group_level','data' )

		self.group_lvl_plot_directory = os.path.join( data_folder, self.experiment_name, 'group_level','figs' )

		self.hdf5_filename = os.path.join(self.group_lvl_data_directory, 'all_data.hdf5')

		

		self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)

		self.vel_type_dictionary = np.dtype([('EL_timestamp', np.float64), ('event_type', np.float64), ('up_down', '|S25'), ('scancode', np.float64), ('key', np.float64), ('modifier', np.float64), ('exp_timestamp', np.float64)])

		

		# add logging for this session

		# sessions create their own logging file handler

		self.log_filename = os.path.join(data_folder, self.experiment_name, 'group_level','log', 'sessionLogFile.log')

		self.loggingLevel = loggingLevel

		self.logger = logging.getLogger( self.__class__.__name__ )

		self.logger.setLevel(self.loggingLevel)

		addLoggingHandler( logging.handlers.TimedRotatingFileHandler( self.log_filename , when = 'H', delay = 2, backupCount = 10), loggingLevel = self.loggingLevel )

		loggingLevelSetup()

		for handler in logging_handlers:

			self.logger.addHandler(handler)

		self.logger.info('starting analysis in ' + self.data_dir)

		



	def all_data_to_hdf(self, aliases,analyze_eye = 'L'):

		""" Takes all data from individual hdf5 files, applies hsas.trial_selection to it

		and saves velocities, expanded amplitudes and which_trials to 

		data_dir/group_level/data/all_data.hdf5.

		Resulting dimensions are participants * trials 

		"""

		

		# create seperate logger for trial rejection

		self.trial_rejection_log_filename = os.path.join(self.data_dir, self.experiment_name, 'group_level','log', 'trial_rejection.log')

		if os.path.isfile(self.trial_rejection_log_filename): os.remove(self.trial_rejection_log_filename)

		self.trial_rejection_logger = logging.getLogger( self.__class__.__name__ )

		self.trial_rejection_logger.setLevel(self.loggingLevel)

		addLoggingHandler( logging.handlers.TimedRotatingFileHandler( self.trial_rejection_log_filename , when = 'H', delay = 2, backupCount = 10), loggingLevel = self.loggingLevel )

		loggingLevelSetup()

		for handler in logging_handlers:

			self.trial_rejection_logger.addHandler(handler)

		

		# delete the hdf5 file and .log file for trial selection if they already exists 

		if  os.path.isfile(self.hdf5_filename): os.remove(self.hdf5_filename)

		# and create a new one

		self.ho.open_hdf_file()

		

		for this_condition in aliases:

			saccades_tables = []

			which_trials_ok = []

			for s  in self.subjects:

				self.logger.info('Reading data and detecting bad trials from %s ' %s.initials)

				hsas = HexagonalSaccadeAdaptationSession(subject = s, experiment_name =self.experiment_name, project_directory = self.data_dir, conditions = this_condition)

				with pd.get_store(os.path.join(self.data_dir, self.experiment_name, s.initials, 'processed', s.initials + '.hdf5')) as h5_file:

					saccade_table = h5_file['%s/saccades_per_trial'%this_condition]

				saccade_table = saccade_table[saccade_table.eye == analyze_eye].set_index('trial')

				saccades_tables.append(saccade_table)

				which_trials_ok.append(hsas.trial_selection(alias = this_condition,initials = s.initials,trial_rejection_logger = self.trial_rejection_logger)[0])



			amps = np.array([[saccades_tables[j][saccades_tables[j].index == i]['expanded_amplitude'].mean() for i in range(hsas.nr_blocks*hsas.nr_trials_per_block)] for j in range(len(saccades_tables))]).reshape((len(saccades_tables),hsas.nr_blocks,150))

			vels = np.array([[saccades_tables[j][saccades_tables[j].index == i]['expanded_amplitude'].mean() for i in range(hsas.nr_blocks*hsas.nr_trials_per_block)] for j in range(len(saccades_tables))]).reshape((len(saccades_tables),hsas.nr_blocks,150))

			which_amps = np.array(which_trials_ok).reshape((len(saccades_tables),hsas.nr_blocks,hsas.nr_trials_per_block))

			amps[-which_amps] = np.nan

			vels[-which_amps] = np.nan

				

			self.logger.info('Adding velocity and expanded_amplitude data from %s to all_data ' %this_condition)

			# fill hdf5 with velocities and expanded amplitudes 

			for block in np.arange(hsas.nr_blocks):

				vel_df = pd.DataFrame(vels[:,block,:])

				self.ho.data_frame_to_hdf(alias = this_condition + '/velocities', name = 'block_%s'%block, data_frame = vel_df)

				amps_df = pd.DataFrame(amps[:,block,:])

				self.ho.data_frame_to_hdf(alias = this_condition + '/expanded_amplitudes', name = 'block_%s'%block, data_frame = amps_df)

				

	def create_trial_selection_evaluation_figs(self,aliases):

		""" Reads in rejected trials from the rejected_trials.log file and

		generates two bar plots, one arranged by participant,

		and one by condition. 

		"""

		# shell()

		measures = ['total','blinks','amplitude','velocity','gaze']

		which_trials = np.zeros((len(aliases),len(self.initials),len(measures)))		

		for ci, this_condition in enumerate(aliases):

			fileName = os.path.join(self.data_dir, self.experiment_name, 'group_level', 'log','trial_rejection.log')

			file = open(fileName,'r')

			line = file.readline()

			while line != '':

				if np.all(['Trials rejected' in line, this_condition in line]):

					for i, m in enumerate(measures):

						which_trials[ci,self.initials.index(line.split()[line.split().index('subject')+1]),i] = line.split()[line.split().index(m)+1]

				line = file.readline()

			file.close()

		

		f = pl.figure(figsize = (24,12))

		colors = ['#75e69d', '#75a5e6',  '#e68a75','#e6df75']

		s1 = f.add_subplot(1,1,1)

		if len(aliases)==4: bar_positions = [ -0.375,-0.125,0.125,0.375]

		else: bar_positions = [0]

		bar_width = 1/len(aliases)

		for i,s in enumerate(self.initials):

			for c, condition in enumerate(aliases):

				if i == 0:

					s1.bar(bar_positions[c]+i*2,which_trials[c,i,1]/(self.nr_trials_per_block*(self.nr_blocks))*100,width=bar_width,color=colors[c],label=condition)

				else:

					s1.bar(bar_positions[c]+i*2,which_trials[c,i,1]/(self.nr_trials_per_block*(self.nr_blocks))*100,width=bar_width,color=colors[c])

		s1.set_xlabel('subject')

		s1.set_ylabel('%  rejected trials')

		pl.xticks(np.arange(0,len(self.initials)*2,2),self.initials)

		leg = pl.legend( loc='upper right', fancybox = True, shadow = False )

		for t in leg.get_texts():

			t.set_fontsize('large')

		leg.get_frame().set_alpha(0.7)

		s1.axis([-2,len(self.initials)*2+2,0,20]);

		

		if not os.path.isdir(os.path.join(self.plot_dir,'trial_rejections')): os.mkdir(os.path.join(self.plot_dir,'trial_rejections'))

		f.savefig(os.path.join(self.plot_dir,'trial_rejections', 'trial_rejections_participants.pdf'))

		

		f = pl.figure(figsize = (24,12))

		colors = ['#75e69d', '#75a5e6',  '#e68a75','#e6df75']

		s1 = f.add_subplot(1,1,1)

		bar_positions = np.arange(0,0.25*len(self.initials),0.25)-0.25*len(self.initials)/2

		bar_width = 0.25

		

		for c, condition in enumerate(aliases):

			for i,s in enumerate(self.initials):

				s1.bar(bar_positions[i]+c*6,which_trials[c,i,1]/(self.nr_trials_per_block*(self.nr_blocks))*100,width=bar_width,color=colors[c])

		s1.set_xlabel('condition')

		s1.set_ylabel('%  rejected trials')

		pl.xticks(np.arange(0,24,len(aliases)),aliases)

		s1.axis([-4,22,0,30])



		f.savefig(os.path.join(self.plot_dir,'trial_rejections', 'trial_rejections_conditions.pdf'))

		

	def fit_functions_to_individual_blocks(self,aliases,plot_individual_blocks,plot_averages,plot_condition_average,baseline_correction):

		""""

		This function takes the first block of every participant, 

		fits linear, exponential and powerlaw functions to it

		and compares these models based on RSS and AIC.

		"""

		if not os.path.isdir(os.path.join(self.plot_dir,'block_fits')): os.mkdir(os.path.join(self.plot_dir,'block_fits'))

		if not os.path.isdir(os.path.join(self.plot_dir,'condition_averages')): os.mkdir(os.path.join(self.plot_dir,'condition_averages'))

			

		lin_res_cond_avg_per_block = []

		exp_res_cond_avg_per_block = [] 

		pow_res_cond_avg_per_block = [] 

		exp_fit_cond_avg_per_block = [] 

		pow_fit_cond_avg_per_block = [] 

		all_means = []

		all_ses = []

		all_data = []

		for blocki in np.arange(8):

			lin_res_one_block_all_conditions = []

			exp_res_one_block_all_conditions = []

			pow_res_one_block_all_conditions = []

			exp_fit_one_block_all_conditions = []

			pow_fit_one_block_all_conditions = []

			all_conditions_one_block_means = []

			all_conditions_one_block_ses = []

			all_conditions_one_block_data = []

			for i, this_condition in enumerate(aliases):

				with pd.get_store(os.path.join(self.data_dir, self.experiment_name,'group_level','data','all_data.hdf5')) as h5_file:

					block_data = np.array((h5_file['/%s/expanded_amplitudes/block_%d'%(this_condition,blocki)]))

				

				lin_rss, exp_rss, pow_rss = [],[],[]

				lin_aic, exp_aic, pow_aic = [],[],[]

				lin_fit, exp_fit, pow_fit = [],[],[]

				lin_res, exp_res, pow_res = [],[],[]

				moving_avg = []

				

				# convert from amplitude to gain:

				block_data = block_data/10

				

				all_subjects = []

				for i, data in enumerate(block_data):

		

					# first interpolate missing data (rejected trials)

					nans, x= nan_helper(data)

					data[nans]= np.interp(x(nans), x(~nans), data[~nans]) 

					

					# then subtract baseline for all but first blocks

					if baseline_correction:

						if blocki != 0:

							data = data - baseline + 1



					moving_avg.append([ np.mean(data[i:i+9]) for i in np.arange(0,150,10)])

					# then fit functions and calculate RSS and AIC per function

					all_subjects.append(data)	

					lin_results = fit_lin(data)

					lin_fit.append(lin_results[2])

					lin_rss.append(lin_results[3])

					lin_aic.append(lin_results[4])

					lin_res.append((lin_results[2]-data)**2)

					exp_results = fit_exp(data)

					exp_fit.append(exp_results[3])

					exp_rss.append(exp_results[4])

					exp_aic.append(exp_results[5])

					exp_res.append((exp_results[3]-data)**2)

					pow_results = fit_pow(data)

					pow_fit.append(pow_results[2])

					pow_rss.append(pow_results[3])

					pow_aic.append(pow_results[4])

					pow_res.append((pow_results[2]-data)**2)

					

				if blocki == 0:

					baseline = np.mean(exp_fit,axis=0)

				block_mean = np.nanmean(all_subjects, axis=0)

				block_se = np.nanstd(all_subjects,axis=0)/np.sqrt(len(self.initials))

				all_conditions_one_block_means.append(block_mean)

				all_conditions_one_block_ses.append(block_se)

				all_conditions_one_block_data.append(all_subjects)

				

				lin_res_one_block_all_conditions.append(lin_res)

				exp_res_one_block_all_conditions.append(exp_res)

				pow_res_one_block_all_conditions.append(pow_res)

				exp_fit_one_block_all_conditions.append(exp_fit)

				pow_fit_one_block_all_conditions.append(pow_fit)

				

				lin_color = '#61a4d7'

				exp_color = '#77d761'

				pow_color = '#d76169'

				

				if plot_individual_blocks:

				

					f = pl.figure(figsize = (18,6))

				

					s = f.add_subplot(1,3,1)	

					s.plot(np.arange(len(block_mean)),block_mean, 'ok', markersize=0.2,mew = 0.15, alpha = 0.575, mec = 'w', ms = 2)

					sns.tsplot(np.dstack((lin_fit,exp_fit,pow_fit)),legend=True,condition=['linear','exponential','powerlaw'],color=[lin_color,exp_color,pow_color],ci=95)

					sns.tsplot(np.mean(np.array(moving_avg),axis=0), time=np.arange(0,150,10)+5, legend=True,condition='moving average',color='magenta',ci=95)

					s.axhline(1.0, linewidth = 0.25)

					s.axis([-10,160,0.6,1.2])

					s.set_xlabel('trial')

					s.set_ylabel('gain')

					s.set_title(this_condition)

					pl.xticks(np.arange(0,160,50))



					s2 = f.add_subplot(1,3,2)	

					sns.boxplot([lin_rss, exp_rss, pow_rss],color=[lin_color,exp_color,pow_color])

					s2.set_ylabel('RSS')

					s2.set_title(this_condition)

					pl.ylim(ymin=0)

					pl.xticks([1,2,3],['linear','exponential','powerlaw'])

			

					s3 = f.add_subplot(1,3,3)

					sns.boxplot([np.array(lin_aic)-np.array(exp_aic),np.array(lin_aic)-np.array(pow_aic),np.array(exp_aic)-np.array(lin_aic)],color=[lin_color,exp_color,pow_color])

					s3.set_ylabel('AIC difference')

					s3.set_title(this_condition)

					pl.xticks([1,2,3],['linear - exponential','linear - powerlaw','exponential - powerlaw'])

			

					f.savefig(os.path.join(self.plot_dir, 'block_fits', 'block_%d_%s.pdf'%(blocki,this_condition)))

					pl.close()

				

				

				

			all_means.append(all_conditions_one_block_means)

			all_ses.append(all_conditions_one_block_ses)

			all_data.append(all_conditions_one_block_data)

			lin_res_cond_avg_per_block.append(lin_res_one_block_all_conditions)

			exp_res_cond_avg_per_block.append(exp_res_one_block_all_conditions)

			pow_res_cond_avg_per_block.append(pow_res_one_block_all_conditions)

			exp_fit_cond_avg_per_block.append(exp_fit_one_block_all_conditions)

			pow_fit_cond_avg_per_block.append(pow_fit_one_block_all_conditions)

		all_data = np.swapaxes(np.swapaxes(np.array(all_data),0,1)[:,1:7,:,:],1,2)

		if baseline_correction:

			np.save(os.path.join(self.data_dir, self.experiment_name , 'group_level','data', 'baseline_corrected_data.npy'),all_data)

		else:

			np.save(os.path.join(self.data_dir, self.experiment_name , 'group_level','data', 'uncorrected_data.npy'),all_data)

		pow_fits = np.swapaxes(np.swapaxes(np.array(pow_fit_cond_avg_per_block),0,1)[:,1:7,:,:],1,2)

		np.save(os.path.join(self.data_dir, self.experiment_name , 'group_level','data', 'pow_fits.npy'),pow_fits)

			

		if plot_averages:

			# compare across conditions

			lin_residuals_block_0 = np.array(lin_res_cond_avg_per_block)[0,:,:,:].mean(axis=0)

			exp_residuals_block_0 = np.array(exp_res_cond_avg_per_block)[0,:,:,:].mean(axis=0)

			pow_residuals_block_0 = np.array(pow_res_cond_avg_per_block)[0,:,:,:].mean(axis=0)

		

			lin_residuals_down = np.array(lin_res_cond_avg_per_block)[np.array([1,3,5]),:,:,:].mean(axis=0).mean(axis=0)

			exp_residuals_down = np.array(exp_res_cond_avg_per_block)[np.array([1,3,5]),:,:,:].mean(axis=0).mean(axis=0)

			pow_residuals_down = np.array(pow_res_cond_avg_per_block)[np.array([1,3,5]),:,:,:].mean(axis=0).mean(axis=0)

		

			lin_residuals_up = np.array(lin_res_cond_avg_per_block)[np.array([2,4,6]),:,:,:].mean(axis=0).mean(axis=0)

			exp_residuals_up = np.array(exp_res_cond_avg_per_block)[np.array([2,4,6]),:,:,:].mean(axis=0).mean(axis=0)

			pow_residuals_up = np.array(pow_res_cond_avg_per_block)[np.array([2,4,6]),:,:,:].mean(axis=0).mean(axis=0)

		

			f = pl.figure(figsize = (24,10))

			s = f.add_subplot(2,3,1)

			sns.tsplot(np.dstack((lin_residuals_block_0-exp_residuals_block_0,lin_residuals_block_0-pow_residuals_block_0,exp_residuals_block_0-pow_residuals_block_0)),legend=True,condition=['linear-exponential','linear-powerlaw','exponential-powerlaw'],color=[lin_color,exp_color,pow_color],ci=95)

			s.set_ylabel('Residuals difference (saccade gain)')

			s.set_xlabel('trials')

			s.set_title('Block 0')

			s.axhline(0.0, linewidth = 0.25)

			s = f.add_subplot(2,3,2)

			sns.tsplot(np.dstack((lin_residuals_down-exp_residuals_down,lin_residuals_down-pow_residuals_down,exp_residuals_down-pow_residuals_down)),legend=True,condition=['linear-exponential','linear-powerlaw','exponential-powerlaw'],color=[lin_color,exp_color,pow_color],ci=95)

			s.set_ylabel('Residuals difference (saccade gain)')

			s.set_xlabel('trials')

			s.set_title('Gain Down Blocks')

			s.axhline(0.0, linewidth = 0.25)

			s = f.add_subplot(2,3,3)

			sns.tsplot(np.dstack((lin_residuals_up-exp_residuals_up,lin_residuals_up-pow_residuals_up,exp_residuals_up-pow_residuals_up)),legend=True,condition=['linear-exponential','linear-powerlaw','exponential-powerlaw'],color=[lin_color,exp_color,pow_color],ci=95)

			s.set_ylabel('Residuals difference (saccade gain)')

			s.set_xlabel('trials')

			s.set_title('Gain Up Blocks')

			s.axhline(0.0, linewidth = 0.25)

			s = f.add_subplot(2,3,4)

			sns.tsplot(np.dstack((lin_residuals_block_0[:,0:10]-exp_residuals_block_0[:,0:10],lin_residuals_block_0[:,0:10]-pow_residuals_block_0[:,0:10],exp_residuals_block_0[:,0:10]-pow_residuals_block_0[:,0:10])),legend=True,condition=['linear-exponential','linear-powerlaw','exponential-powerlaw'],color=[lin_color,exp_color,pow_color],ci=95)

			s.set_ylabel('Residuals difference (saccade gain)')

			s.set_xlabel('trials')

			s.set_title('Block 0 - zoomed in')

			s.axhline(0.0, linewidth = 0.25)

			s = f.add_subplot(2,3,5)

			sns.tsplot(np.dstack((lin_residuals_down[:,0:10]-exp_residuals_down[:,0:10],lin_residuals_down[:,0:10]-pow_residuals_down[:,0:10],exp_residuals_down[:,0:10]-pow_residuals_down[:,0:10])),legend=True,condition=['linear-exponential','linear-powerlaw','exponential-powerlaw'],color=[lin_color,exp_color,pow_color],ci=95)

			s.set_ylabel('Residuals difference (saccade gain)')

			s.set_xlabel('trials')

			s.set_title('Gain Down Blocks - zoomed in')

			s.axhline(0.0, linewidth = 0.25)

			s = f.add_subplot(2,3,6)

			s.set_ylabel('Residuals difference (saccade gain)')

			s.set_xlabel('trials')

			s.set_title('Gain Up Blocks - zoomed in')

			sns.tsplot(np.dstack((lin_residuals_up[:,0:10]-exp_residuals_up[:,0:10],lin_residuals_up[:,0:10]-pow_residuals_up[:,0:10],exp_residuals_up[:,0:10]-pow_residuals_up[:,0:10])),legend=True,condition=['linear-exponential','linear-powerlaw','exponential-powerlaw'],color=[lin_color,exp_color,pow_color],ci=95)

			s.axhline(0.0, linewidth = 0.25)

			f.savefig(os.path.join(self.plot_dir, 'block_fits', 'fit_comparisons_adaptation_blocks.pdf'))

			pl.close()

		

		if plot_condition_average:

			colors = np.array(['r','g'])

			my_palette = [(119/255.0,215/255.0,97/255.0), (215/255.0,97/255.0,105/255.0)]

			exp_fits = np.swapaxes(np.array(exp_fit_cond_avg_per_block),0,1)[:,1:7,:,:]

			pow_fits = np.swapaxes(np.array(pow_fit_cond_avg_per_block),0,1)[:,1:7,:,:]

			ses_data = np.swapaxes(np.array(all_ses),0,1)[:,1:7,:]

			mean_data = np.swapaxes(np.array(all_means),0,1)[:,1:7,:]

			# shell()

			for c, data in enumerate(mean_data):

				f = pl.figure(figsize=(12,4))

				s = f.add_subplot(111)

				

				for b, block in enumerate(data):

					s.plot(block.shape[0] * b + np.arange(block.shape[0]), block, colors[b%2] + 'o', mew = 1.75, alpha = 0.625, mec = 'w', ms = 6  )

					pl.fill_between(block.shape[0] * i + np.arange(block.shape[0]), block - ses_data[c,b,:], block + ses_data[c,b,:], alpha = np.linspace(0.025,0.075,len(data))[2], color = colors[b%2]  )

					if b%2: sns.tsplot(exp_fits[c,b,:,:],time = block.shape[0] * b + np.arange(block.shape[0]), color = colors[b%2])

					else: sns.tsplot(pow_fits[c,b,:,:],time = block.shape[0] * b + np.arange(block.shape[0]),color = colors[b%2])

					# for p, pp in enumerate(all_data):

					# 	s.plot(block.shape[0] * b + np.arange(block.shape[0]), all_data[c,b,p,:], 'o', mec = colors[b%2], mew = 1, alpha = 0.06225,  ms = 3)

				

				simpleaxis(s)

				spine_shift(s)

				s.axis([-20,920,0.75,1.25])

				s.set_xticks(np.arange(0,1050,150))

				s.grid(axis = 'x', linestyle = '--', linewidth = 0.25)

				s.axhline(1.0, linewidth = 0.25)

				s.set_xlabel('trials within blocks')

				s.set_ylabel('saccade gain')

				s.set_title(aliases[c] + '\nCondition averages')

				

				f.savefig(os.path.join(self.plot_dir, 'condition_averages', aliases[c] + 'condition_average.pdf'))

				pl.close()

			

				start_points_exp = exp_fits[c,np.array([0,2,4]),:,np.array([0])]

				end_points_exp = exp_fits[c,np.array([0,2,4]),:,np.array([149])]

				start_points_pow = pow_fits[c,np.array([1,3,5]),:,np.array([0])]

				end_points_pow = pow_fits[c,np.array([1,3,5]),:,np.array([149])]



				## seaborn factorplot fashion doesn't work perfectly, as factorplot cannot plot in subplots... 

				blocks = np.tile(np.array( [np.tile(0,len(self.initials)),np.tile(1,len(self.initials)),np.tile(2,len(self.initials))]).flatten(),2).flatten()

				subjects = np.tile(np.arange(0,len(self.initials)),6).T

				condition = np.array([np.tile('gain-up',len(self.initials)*3), np.tile('gain-down',len(self.initials)*3)]).flatten().T

				end_points =  np.array([exp_fits[c,np.array([1,3,5]),:,np.array([0])].flatten(),pow_fits[c,np.array([0,2,4]),:,np.array([0])].flatten()]).flatten().T

				start_points =  np.array([exp_fits[c,np.array([0,2,4]),:,np.array([0])].flatten(),pow_fits[c,np.array([1,3,5]),:,np.array([0])].flatten()]).flatten().T

				starts = pd.DataFrame({'starts':start_points,'subject':subjects,'condition':condition,'blocks':blocks},dtype=float)

				ends = pd.DataFrame({'ends':end_points,'subject':subjects,'condition':condition,'blocks':blocks},dtype=float)

				sns.factorplot('blocks','starts','condition',starts,dodge=.05,palette = my_palette,ci=95)

				s = pl.gca()

				f = pl.gcf()

				s.set_title('Startspoints ' + aliases[c])

				f.savefig(os.path.join(self.plot_dir, 'condition_averages', aliases[c] + '_startpoints.pdf'))

				pl.close()

				sns.factorplot('blocks','ends','condition',ends,dodge=.05,palette = my_palette,ci=95)

				s = pl.gca()

				f = pl.gcf()

				s.set_title('Endpoints ' + aliases[c])

				f.savefig(os.path.join(self.plot_dir, 'condition_averages', aliases[c] + '_endpoints.pdf'))

				pl.close()

				

