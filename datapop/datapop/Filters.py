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

class Filters(DataBase):
    """
    Data sets storage placement optimizer.
    Parameters:
    ------------
    :param pandas.DataFrame popularity_report: popularity report from DataPopularityEstimator.

    :param pandas.DataFrame prediction_report: predicted intensities from DataIntensityPredictor.

    :param pandas.DataFrame opti_report: report from DataPlacementOptimizer.

    :param pandas.DataFrame data: train data.
    """

    def __init__(self, popularity_report, prediction_report, opti_report, data):
        super(Filters, self).__init__(source_path=None, data=data, nb_of_weeks=5) ##Not effective
        self.popularity_report = popularity_report.sort(columns='Name')
        self.prediction_report = prediction_report.sort(columns='Name')
        self.opti_report = opti_report.sort(columns='Name')
        self.data1 = self.data.copy()
        self.data = self.data1[self.data1['Name'].isin(self.popularity_report['Name'])].sort(columns='Name')

    def SaveNTB(self, N, mode='base'): #Need for optimization
        """
        This is the filter 'what to do to save N TB of disk space.'
        :param int N: number of TB is wanted to be saved.
        :param mode: For 'base' - datasets with high popularity value will be removed from disk.
        For 'save' - datasets with high popularity value will have just one replica on disk.
        :return: pandas.DataFrame, float
        """
        popularity_report = self.popularity_report.irow(np.argsort(self.popularity_report['Popularity'].values))
        prediction_report = self.prediction_report.irow(np.argsort(self.popularity_report['Popularity'].values))
        opti_report = self.opti_report.irow(np.argsort(self.popularity_report['Popularity'].values))
        data = self.data.irow(np.argsort(self.popularity_report['Popularity'].values))

        diff_replicas = data['Nb_Replicas'].values - opti_report['NbReplicas'].values

        save_report = pd.DataFrame()
        save_report['Name'] = data['Name'].values
        ################
        replicas = data['Nb_Replicas'].values
        if mode=='base':
            save_report['To_tape'] = (data['Nb_Replicas']*(opti_report['OnDisk'].values==0)*data['LFNSize']).values
            max_diff = int(diff_replicas.max()//1 + 1)
            steps = int(diff_replicas.max()//1 - diff_replicas.min()//1 + 1)
            for i in range(1, steps):
                if i==1:
                    label = '-%i_replica' % i
                else:
                    label = '-%i_replicas' % i
                selection = (diff_replicas >= max_diff-i)*(opti_report['OnDisk'].values==1)
                save_report[label] = (selection*(replicas >= 2.)*data['LFNSize']).values
                replicas = replicas - 1.*selection

        if mode=='save':
            save_report['Leave_one_replica'] = ((data['Nb_Replicas']-1)*(opti_report['OnDisk'].values==0)*data['LFNSize']).values
            max_diff = int(diff_replicas.max()//1 + 1)
            steps = int(diff_replicas.max()//1 - diff_replicas.min()//1 + 1)
            for i in range(1, steps):
                if i==1:
                    label = '-%i_replica' % i
                else:
                    label = '-%i_replicas' % i
                selection = (diff_replicas >= max_diff-i)*(opti_report['OnDisk'].values==1)
                save_report[label] = (selection*(replicas >= 2.)*data['LFNSize']).values
                replicas = replicas - 1.*selection
        ################
        cols = save_report.columns[1:]
        rows = range(0, data.shape[0])
        rows.reverse()
        todo = np.array(data.shape[0]*[0.])
        save_space = 0
        for col in cols:
            if col == 'To_tape':
                for row in rows:
                    val = save_report[[col]].irow([row]).values[0][0]
                    if save_space <= N:
                        if val != 0:
                            save_space += val
                            todo[row] = -1.*data[['Nb_Replicas']].irow([row]).values[0][0]
                    else:
                        break
                if save_space > N:
                    break

            elif col == 'Leave_one_replica':
                for row in rows:
                    val = save_report[[col]].irow([row]).values[0][0]
                    if save_space <= N:
                        if val != 0:
                            save_space += val
                            todo[row] = 1. - 1.*data[['Nb_Replicas']].irow([row]).values[0][0]
                    else:
                        break
                if save_space > N:
                    break

            else:
                todo_val = 0
                for row in rows:
                    val = save_report[[col]].irow([row]).values[0][0]
                    if save_space <= N:
                        if val != 0:
                            save_space += val
                            todo[row] -= todo_val + 1
                    else:
                        break
                if save_space > N:
                    break


        save_report['LFNSize'] = data['LFNSize'].values
        if mode=='base':
            save_report['Opti_NbReplicas'] = (opti_report['NbReplicas'].values)*(opti_report['OnDisk'].values==1)
        elif mode=='save':
            save_report['Opti_NbReplicas'] = (opti_report['NbReplicas'].values)*(opti_report['OnDisk'].values==1) +\
                                            1.*(opti_report['OnDisk'].values==0)
        save_report['Current_NbReplicas'] = data['Nb_Replicas'].values
        save_report['Popularity'] = popularity_report['Popularity'].values
        save_report['ToDo'] = todo
        save_report = save_report.sort(columns='Name')

        return save_report[['Name', 'LFNSize', 'Opti_NbReplicas', 'Current_NbReplicas', 'ToDo']], save_space
        #return save_repot, save_space
