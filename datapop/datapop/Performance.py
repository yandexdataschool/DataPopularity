from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

######################
###Version 0.2     ###
######################
###Using REP v0.5  ###
######################

import numpy as np
import pandas as pd
import re
from .DataBase import DataBase

class Performance(object):
    """
    Metrics of the algorithm's performance
    Parameters:
    -----------
    :param pandas.DataFrame data: train data.

    :param int nb_of_weeks: number of weeks in data sets access history.

    :param pandas.DataFrame popularity_report: report from the DataPopularityEstimator

    :param pandas.DataFrame prediction_report: report from the DataIntensityPrediction

    :param pandas.DataFrame report: report from the DataPlacementOptimizer
    """

    def __init__(self, data=None, nb_of_weeks=104, popularity_report=None, prediction_report=None, report=None):
        self.data_origin = data
        self._fix_columns()

        selection = (self.data_origin.sort(columns='Name')['Nb_Replicas']>=1.).values #In DataPlacementOptimizer TODO
        self.data_origin = self.data_origin.sort(columns='Name')[selection]

        self.nb_of_weeks = nb_of_weeks
        self.popularity_report = popularity_report.sort(columns='Name')[selection]
        self.prediction_report = prediction_report.sort(columns='Name')[selection]
        self.report = report.sort(columns='Name')


    def _rename(self, x):
        """
        :param str or int x: column's name
        :return: str renamed column's name
        """
        return re.sub('\W', '_', str(x))

    def _fix_columns(self):
        """
        Rename data columns
        :return: list[str] renamed column names
        """
        self.data_origin.columns = map(self._rename, self.data_origin.columns)
        return self.data_origin.columns

    def get_original_downloading_time(self, t_disk=0.1, t_tape=30, const_tape=240):
        """
        Get downloading time of all datasets by all users before the algorithm implementation
        :param t_disk: downloading time of 1 Gb of data from disk
        :param t_tape: time needed to restore 1 Gb of data from tape
        :param const_tape: constant time needed to restore 1 Gb of data from tape
        :return: downloading time of all datasets by all users before the algorithm implementation
        """
        Rp = self.data_origin.sort(columns='Name')['Nb_Replicas'].values.astype(np.float)
        d = ((self.data_origin.sort(columns='Name')['Storage'].values=='Disk')*1.)
        S = self.data_origin.sort(columns='Name')['LFNSize'].values.astype(np.float)
        I = (self.data_origin.sort(columns='Name')['26'].values.astype(np.float))#/26.
        a = (0.05 + 1/(Rp + (Rp==0)*1.))*(Rp!=0)*1.
        m = (d==0)*(I!=0)*1.

        t = (I*S*t_disk*a*d).sum() + ((const_tape + S*t_tape)*m).sum() + (I*S*t_disk*m).sum()
        return t

    def get_downloading_time_ratio(self, t_disk=0.1, t_tape=30, const_tape=240):
        """
        Get ratio of the downloading time of all datasets by all users before the algorithm implementation and the downloading time of all datasets by all users after the algorithm implementation
        :param t_disk: downloading time of 1 Gb of data from disk
        :param t_tape: time needed to restore 1 Gb of data from tape
        :param const_tape: constant time needed to restore 1 Gb of data from tape
        :return: downloading time ratio
        """
        Rp = self.report.sort(columns='Name')['NbReplicas'].values.astype(np.float)
        d = self.report.sort(columns='Name')['OnDisk'].values.astype(np.float)
        S = self.data_origin.sort(columns='Name')['LFNSize'].values.astype(np.float)
        I = (self.data_origin.sort(columns='Name')['26'].values.astype(np.float))#/26.
        a = (0.05 + 1/(Rp + (Rp==0)*1.))*(Rp!=0)*1.
        m = (d==0)*(I!=0)*1.

        t = (I*S*t_disk*a*d).sum() + ((const_tape + S*t_tape)*m).sum() + (I*S*t_disk*m).sum()
        t_original = self.get_original_downloading_time(t_disk, t_tape, const_tape)
        return t/t_original

    def get_saving_space(self):
        """
        Get % of saving disk space after the algorithm implementation
        :return: percent of saving space
        """
        Rp = self.report.sort(columns='Name')['NbReplicas'].values.astype(np.float)
        d = self.report.sort(columns='Name')['OnDisk'].values.astype(np.float)
        S = self.data_origin.sort(columns='Name')['LFNSize'].values.astype(np.float)
        I = (self.data_origin.sort(columns='Name')['26'].values.astype(np.float))#/26.
        a = (0.05 + 1/(Rp + (Rp==0)*1.))*(Rp!=0)*1.
        m = (d==0)*(I!=0)*1.

        space = (S*Rp*d).sum() + (S*m).sum()
        space_original = self.data_origin.sort(columns='Name')['DiskSize'].values.astype(np.float32).sum()
        return (1-space/space_original)*100

    def get_nb_of_mistakes(self):
        """
        Get number of the algorithm's mistakes
        :return: number of mistakes
        """
        d = self.report.sort(columns='Name')['OnDisk'].values.astype(np.float)
        I = (self.data_origin.sort(columns='Name')['26'].values.astype(np.float))#/26.
        m = ((d==0)*(I!=0)*1.).sum()
        return m

    def get_performance_report(self, t_disk=0.1, t_tape=30, const_tape=240):
        """
        Get full performance report
        :param t_disk: downloading time of 1 Gb of data from disk
        :param t_tape: time needed to restore 1 Gb of data from tape
        :param const_tape: constant time needed to restore 1 Gb of data from tape
        :return: performance report
        """
        performance_report = pd.DataFrame()
        performance_report['Downloading_time_ratio (train)'] = [self.get_downloading_time_ratio(t_disk, t_tape, const_tape)]
        performance_report['Saving_space_(%) (train)'] = [self.get_saving_space()]
        performance_report['Nb_of_mistakes (train)'] = [self.get_nb_of_mistakes()]
        return performance_report
