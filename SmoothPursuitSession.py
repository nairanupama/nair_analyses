#!/usr/bin/env python

# encoding: utf-8

"""

EyeLinkSession.py



Created by Tomas Knapen on 2011-04-27.

Copyright (c) 2011 __MyCompanyName__. All rights reserved.

"""

from IPython import embed as shell

import os

import math



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





import logging

import logging.handlers

import logging.config



# sys.path.append( os.environ['ANALYSIS_HOME'] )

from ..log import *

from ..Operators import EDFOperator, HDFEyeOperator, EyeSignalOperator

from ..Operators.EyeSignalOperator import detect_saccade_from_data

from ..Operators.CommandLineOperator import ExecCommandLine

from ..other_scripts.plotting_tools import *

from ..other_scripts.circularTools import *





class SmoothPursuitSession(object):

    """SmoothPursuitSession"""



    def __init__(self, subject, experiment_name, project_directory, conditions, loggingLevel=logging.DEBUG):

        self.subject = subject

        self.experiment_name = experiment_name

        self.conditions = conditions

        try:

            os.mkdir(os.path.join(project_directory, experiment_name))

            os.mkdir(

                os.path.join(project_directory, experiment_name, self.subject.initials))

        except OSError:

            pass

        self.project_directory = project_directory

        self.base_directory = os.path.join(

            self.project_directory, self.experiment_name, self.subject.initials)



        self.create_folder_hierarchy()

        self.hdf5_filename = os.path.join(

            self.base_directory, 'processed', self.subject.initials + '.hdf5')

        self.ho = HDFEyeOperator.HDFEyeOperator(self.hdf5_filename)



        self.nr_trials_per_block = 150

        self.nr_blocks = 8

        self.saccade_amplitude = 10

        self.block_trial_indices = np.array([np.arange(

            0, self.nr_trials_per_block) + (i * self.nr_trials_per_block) for i in range(self.nr_blocks)])



        self.all_block_colors = ['k', 'r', 'g', 'r', 'g', 'r', 'g', 'k']



        self.velocity_profile_duration = self.signal_profile_duration = 100

        # add logging for this session

        # sessions create their own logging file handler

        self.loggingLevel = loggingLevel

        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.setLevel(self.loggingLevel)

        addLoggingHandler(logging.handlers.TimedRotatingFileHandler(os.path.join(

            self.base_directory, 'log', 'sessionLogFile.log'), when='H', delay=2, backupCount=10), loggingLevel=self.loggingLevel)

        loggingLevelSetup()

        for handler in logging_handlers:

            self.logger.addHandler(handler)

        self.logger.info('starting analysis in ' + self.base_directory)



    def create_folder_hierarchy(self):

        """createFolderHierarchy does... guess what."""

        this_dir = self.project_directory

        for d in [self.experiment_name, self.subject.initials]:

            try:

                this_dir = os.path.join(this_dir, d)

                os.mkdir(this_dir)

            except OSError:

                pass

        for p in ['raw', 'processed', 'figs', 'log']:

            try:

                os.mkdir(os.path.join(self.base_directory, p))

            except OSError:

                pass



    def import_raw_data(self, edf_files, aliases):

        """import_raw_data loops across edf_files and their respective aliases and copies and renames them into the raw directory."""

        for edf_file, alias in zip(edf_files, aliases):

            self.logger.info('importing file ' + edf_file + ' as ' + alias)

            ExecCommandLine(

                'cp "' + edf_file + '" "' + os.path.join(self.base_directory, 'raw', alias + '.edf"'))



    def import_all_data(self, aliases):

        """import_all_data loops across the aliases of the sessions and converts the respective edf files, adds them to the self.ho's hdf5 file. """

        for alias in aliases:

            self.ho.add_edf_file(

                os.path.join(self.base_directory, 'raw', alias + '.edf'))

            self.ho.edf_message_data_to_hdf(alias=alias)

            self.ho.edf_gaze_data_to_hdf(alias=alias)



    def detect_all_saccades(self, alias, n_jobs=-1, threshold=5.0, use_eye=[]):

        """docstring for detect_all_saccades"""

        self.logger.info('starting saccade detection of ' + alias)

        all_saccades = []

        for bi, tb in enumerate(self.block_trial_indices):

            this_block_eye = self.ho.eye_during_trial(

                trial_nr=tb[0], alias=alias)

            this_block_sr = self.ho.sample_rate_during_trial(

                trial_nr=tb[0], alias=alias)

            this_block_res = []

            for tbe in this_block_eye:

                xy_data = [self.ho.signal_from_trial_phases(trial_nr=tr, trial_phases=[

                                                            1, 3], alias=alias, signal='gaze', requested_eye=tbe, time_extensions=[0, 200]) for tr in tb]

                vel_data = [self.ho.signal_from_trial_phases(trial_nr=tr, trial_phases=[

                                                             1, 3], alias=alias, signal='vel', requested_eye=tbe, time_extensions=[0, 200]) for tr in tb]

                res = Parallel(n_jobs=n_jobs, verbose=9)(delayed(detect_saccade_from_data)(

                    xy, vel, l=threshold, sample_rate=this_block_sr) for xy, vel in zip(xy_data, vel_data))

                # res = [detect_saccade_from_data(xy, vel, l = threshold, sample_rate = this_block_sr ) for xy, vel in zip(xy_data, vel_data)]

                for (tr, r) in zip(tb, res):

                    r[0].update({'trial': tr, 'eye': tbe, 'block': bi})

                this_block_res.append([r[0] for r in res])

            all_saccades.append(this_block_res)

        all_saccades = list(chain.from_iterable(all_saccades))

        all_saccades_pd = pd.DataFrame(list(chain.from_iterable(all_saccades)))

        self.ho.data_frame_to_hdf(

            alias=alias, name='saccades_per_trial', data_frame=all_saccades_pd)

