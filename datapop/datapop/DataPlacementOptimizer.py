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

import matplotlib.pyplot as plt
from .DataBase import DataBase

class DataPlacementOptimizer(DataBase):
    """
    Data sets storage placement optimizer.
    Parameters:
    ------------
    :param pandas.DataFrame popularity_report: popularity report from DataPopularityEstimator.

    :param pandas.DataFrame prediction_report: predicted intensities from DataIntensityPredictor.

    :param pandas.DataFrame data: train data.

    :param: str source_path: a data file path.

    :param: int nb_of_weeks: number of weeks in data access history.
    """

    def __init__(self, popularity_report, prediction_report, data, source_path=None, nb_of_weeks=104):
        super(DataPlacementOptimizer, self).__init__(source_path=source_path, data=data, nb_of_weeks=nb_of_weeks)

        selection = (data.sort(columns='Name')['Nb_Replicas']>=1.).values #In Performance. TODO
        data_sel = data.sort(columns='Name')[selection]

        self.popularity_report = popularity_report.sort(columns='Name')[selection]
        self.prediction_report = prediction_report.sort(columns='Name')[selection]
        self.data = data_sel[data_sel['Name'].isin(self.popularity_report['Name'])].sort(columns='Name')
        self.total_report = self._init_total_report()


    def _init_total_report(self):
        """
        Initialize total_report.

        :return: pandas.DataFrame total_report. Add to total_report 'Popularity', 'Intensity', 'LFNSize' and 'Name' columns.
        """
        total_report = self.popularity_report
        total_report['Intensity'] = self.prediction_report['Intensity'].values
        total_report['LFNSize'] = self.data['LFNSize'].values
        return total_report

    def _set_disk_tape_marker(self, total_report, pop_cut=0.5):
        """
        Modify total_report.

        :param pandas.DataFrame total_report: total_report

        :param float(0,1] pop_cut: popularity cut value.

        :return: pandas.DataFrame total_report.
        """
        if pop_cut<=0 or pop_cut>1:
            assert 0, 'Pop_cut value should be in range (0,1].'
        pop = total_report['Popularity'].values
        total_report['OnDisk'] = (pop<=pop_cut)*1
        return total_report

    def _set_nb_replicas_quantile(self, total_report, q=[0.5, 0.75, 0.95]):
        """
        Set data sets number of replicas by Intensity quantile values.

        :param pandas.DataFrame total_report: total_report

        :param list(float) q: list of Intensity quantile values. For example, q=[0.2, 0.5] means that 20% data sets on disk
         will have 1 replica, 30% - 2 replicas, 50% - 3 replicas.

        :return: pandas.DataFrame total_report. Total_report with 'NbReplicas' column.
        """
        marker = total_report['OnDisk'].values
        inten = total_report['Intensity'].values
        inten_cut = total_report['Intensity'][(marker==1)]
        total_report['NbReplicas'] = 1
        for i in q:
            total_report['NbReplicas'] = total_report['NbReplicas'] + (marker==1)*(inten>inten_cut.quantile(i))*1
        return total_report

    def _set_nb_replicas_values(self, total_report, q=[1, 100, 1000]):
        """
        Set data sets number of replicas by Intensity values.

        :param pandas.DataFrame total_report: total_report

        :param list(float) q: list of Intensity values. For example, q=[10, 100] means that data sets on disk which have Intensity values below or equal 10
         will have 1 replica, data sets with Intensity values below or equal 100 but above 10  - 2 replicas, data sets with Intensity values above 100 - 3 replicas.

        :return: pandas.DataFrame total_report. Total_report with 'NbReplicas' column.
        """
        marker = total_report['OnDisk'].values
        inten = total_report['Intensity'].values
        total_report['NbReplicas'] = 1
        for i in q:
            total_report['NbReplicas'] = total_report['NbReplicas'] + (marker==1)*(inten>i)*1
        return total_report

    def _set_nb_replicas_load(self, total_report, q=[0.1, 1, 10]):
        """
        Set data sets number of replicas by 'load' values (Intensity*LFNSize).

        :param pandas.DataFrame total_report: total_report

        :param list(float) q: list of 'load' values. For example, q=[10, 100] means that data sets on disk which have 'load' values below or equal 10
         will have 1 replica, data sets with 'load' values below or equal 100 but above 10  - 2 replicas, data sets with 'load' values above 100 - 3 replicas.

        :return: pandas.DataFrame total_report. Total_report with 'NbReplicas' column.
        """
        marker = total_report['OnDisk'].values
        inten = (total_report['Intensity'].values)*(total_report['LFNSize'].values)
        total_report['NbReplicas'] = 1
        for i in q:
            total_report['NbReplicas'] = total_report['NbReplicas'] + (marker==1)*(inten>i)*1
        return total_report

    def _set_nb_replicas_origin(self, total_report):
        """
        Add to total_report origin number od replicas for data sets.

        :param pandas.DataFrame total_report: total_report

        :return: pandas.DataFrame total_report. Total_report with 'NbReplicas' column.
        """
        total_report['NbReplicas'] = self.data['Nb_Replicas'].values + (self.data['Nb_Replicas'].values==0)*1
        return total_report

    def _set_nb_replicas_auto(self, total_report, alpha=10, min_replicas=1, max_replicas=4):
        """
        Set data sets number of replicas by optimized values from loss function.

        :param pandas.DataFrame total_report: total_report

        :param float alpha: fine for low number of replicas. Higher alpha value, higher number of data sets get >1 replicas.

        :param int max_replicas: max number of replicas.

        :return: pandas.DataFrame total_report. Total_report with 'NbReplicas' column.
        """
        marker = total_report['OnDisk'].values
        inten = total_report['Intensity'].values*marker
        values = np.array(np.float(min_replicas) + float(alpha)*inten/np.float(min_replicas), ndmin=2).T
        replicas = np.ones(values.shape)*np.float(min_replicas)
        for i in range(min_replicas+1, max_replicas+1):
            val_for_i = np.array(float(i) + float(alpha)*inten/float(i), ndmin=2).T
            rep_for_i = np.ones(val_for_i.shape)*i
            values = np.concatenate((values, val_for_i), axis=1)
            replicas = np.concatenate((replicas, rep_for_i), axis=1)
        min_values = values.min(axis=1).reshape(values.shape[0],1)
        mask_values = (values==min_values)
        nb_replicas = []
        for i in mask_values:
            nb_rep = np.nonzero(i)[0][0]+min_replicas
            nb_replicas.append(float(nb_rep))
        nb_replicas = np.array(nb_replicas)
        total_report['NbReplicas'] = nb_replicas
        #total_report['NbReplicas'] = list(1 + replicas[values==values.min(axis=1).reshape(values.shape[0],1)])
        return total_report

    def _set_missing(self, total_report):
        """
        Modify total_report.

        :param pandas.DataFrame total_report: total_report

        :return: pandas.DataFrame total_report. Total_report with 'Missing' column.
        """
        marker = total_report['OnDisk']
        label = total_report['Label']
        total_report['Missing'] = (label==0)*(marker==0)*1
        return total_report

    def _report_upgrade(self, total_report, pop_cut=0.5, q=None, alpha=10, min_replicas=1, max_replicas=4, set_replicas='auto'):
        """
        Return upgraded total_report.

        :param pandas.DataFrame total_report: total_report

        :param float(0,1] pop_cut: popularity cut value.

        :param str set_replicas: label of data sets number of replicas determination algorithm.
        'auto' (preferable): set data sets number of replicas by optimized values from loss function.
        'quantile': set data sets number of replicas by Intensity quantile values.
        'values': set data sets number of replicas by Intensity values.
        'load': set data sets number of replicas by 'load' values (Intensity*LFNSize).
        'origin': use origin number od replicas for data sets.

        :param list(float) q:For set_replicas='auto' (preferable): None (ignored).
         For set_replicas='quantile': list of Intensity quantile values. For example, q=[0.2, 0.5] means that 20% data sets on disk
         will have 1 replica, 30% - 2 replicas, 50% - 3 replicas.
         For set_replicas='values': list of Intensity values. For example, q=[10, 100] means that data sets on disk which have Intensity values below or equal 10
         will have 1 replica, data sets with Intensity values below or equal 100 but above 10  - 2 replicas, data sets with Intensity values above 100 - 3 replicas.
         For set_replicas='load': list of 'load' values. For example, q=[10, 100] means that data sets on disk which have 'load' values below or equal 10
         will have 1 replica, data sets with 'load' values below or equal 100 but above 10  - 2 replicas, data sets with 'load' values above 100 - 3 replicas.
         For set_replicas='origin': None (ignored).

        :param float alpha: fine for low number of replicas. Higher alpha value, higher number of data sets get >1 replicas.

        :param int max_replicas: max number of replicas.

        :return: pandas.DataFrame total_report. Upgraded total_report.
        """
        total_report = self._set_disk_tape_marker(total_report, pop_cut)
        if set_replicas=='quantile':
            total_report = self._set_nb_replicas_quantile(total_report, q)
        elif set_replicas=='values':
            total_report = self._set_nb_replicas_values(total_report, q)
        elif set_replicas=='load':
            total_report = self._set_nb_replicas_load(total_report, q)
        elif set_replicas=='origin':
            total_report = self._set_nb_replicas_origin(total_report)
        elif set_replicas=='auto':
            total_report = self._set_nb_replicas_auto(total_report, alpha, min_replicas, max_replicas)
        else:
            assert 0, "Value of the parameter 'set_replicas' is incorrect. Check your 'set_replicas' value."
        total_report = self._set_missing(total_report)
        return total_report

    def _loss_function(self, total_report, c_disk=100, c_tape=1, c_miss=10000, alpha=10):
        """
        Calculate loss function's value.

        :param pandas.DataFrame total_report: total_report.

        :param float c_disk: cost of storage of 1Gb on disk.

        :param float c_tape: cost of storage of 1Gb on tape.

        :param float c_miss: cost of restoring 1Gb from tape to disk.

        :param float alpha: fine for low number of replicas. Higher alpha value, higher number of data sets get >1 replicas.

        :return:float loss. Loss function value.
        """
        lfn_size = total_report['LFNSize'].values
        nb_replicas = total_report['NbReplicas'].values
        marker = total_report['OnDisk'].values
        miss = total_report['Missing'].values
        inten = total_report['Intensity'].values

        nc_disk = (marker*lfn_size*nb_replicas*c_disk).sum()
        nc_tape = ((1-marker)*lfn_size*nb_replicas*c_tape).sum()
        nc_miss = (miss*lfn_size*nb_replicas*c_miss).sum()
        nc_replicas = (marker*lfn_size*inten*float(alpha)/nb_replicas).sum()

        loss = (nc_disk + nc_tape + nc_miss + nc_replicas)
        return loss

    def _get_loss_curve(self, total_report, q=None, set_replicas='auto', c_disk=100, c_tape=1, c_miss=10000, alpha=10, min_replicas=1, max_replicas=4):
        """

        :param pandas.DataFrame total_report: total_report.

        :param str set_replicas: label of data sets number of replicas determination algorithm.
        'auto' (preferable): set data sets number of replicas by optimized values from loss function.
        'quantile': set data sets number of replicas by Intensity quantile values.
        'values': set data sets number of replicas by Intensity values.
        'load': set data sets number of replicas by 'load' values (Intensity*LFNSize).
        'origin': use origin number od replicas for data sets.

        :param list(float) q:For set_replicas='auto' (preferable): None (ignored).
         For set_replicas='quantile': list of Intensity quantile values. For example, q=[0.2, 0.5] means that 20% data sets on disk
         will have 1 replica, 30% - 2 replicas, 50% - 3 replicas.
         For set_replicas='values': list of Intensity values. For example, q=[10, 100] means that data sets on disk which have Intensity values below or equal 10
         will have 1 replica, data sets with Intensity values below or equal 100 but above 10  - 2 replicas, data sets with Intensity values above 100 - 3 replicas.
         For set_replicas='load': list of 'load' values. For example, q=[10, 100] means that data sets on disk which have 'load' values below or equal 10
         will have 1 replica, data sets with 'load' values below or equal 100 but above 10  - 2 replicas, data sets with 'load' values above 100 - 3 replicas.
         For set_replicas='origin': None (ignored).

        :param float c_disk: cost of storage of 1Gb on disk.

        :param float c_tape: cost of storage of 1Gb on tape.

        :param float c_miss: cost of restoring 1Gb from tape to disk.

        :param float alpha: fine for low number of replicas. Higher alpha value, higher number of data sets get >1 replicas.

        :param int max_replicas: max number of replicas.

        :return: list of cut values, list of loss function values for the cuts, min cut value for the min loss, min loss value.
        """
        cuts = np.array(np.arange(0.01, 1, 0.01))
        loss_curve = []
        for i in cuts:
            report = self._report_upgrade(total_report=total_report, pop_cut=i,  q=q, alpha=alpha, min_replicas=min_replicas, max_replicas=max_replicas, set_replicas=set_replicas)
            loss = self._loss_function(report, c_disk, c_tape, c_miss, alpha)
            loss_curve.append(loss)
        loss_curve = np.array(loss_curve)
        min_loss = loss_curve.min()
        min_cut = cuts[loss_curve==min_loss]
        return cuts, loss_curve, min_cut, min_loss

    def get_report(self, pop_cut=0.5, q=None, set_replicas='auto', alpha=10, min_replicas=1, max_replicas=4):
        """
        Get data sets storage placement report.

        :param float(0,1] pop_cut: popularity cut value.

        :param str set_replicas: label of data sets number of replicas determination algorithm.
        'auto' (preferable): set data sets number of replicas by optimized values from loss function.
        'quantile': set data sets number of replicas by Intensity quantile values.
        'values': set data sets number of replicas by Intensity values.
        'load': set data sets number of replicas by 'load' values (Intensity*LFNSize).
        'origin': use origin number od replicas for data sets.

        :param list(float) q:For set_replicas='auto' (preferable): None (ignored).
         For set_replicas='quantile': list of Intensity quantile values. For example, q=[0.2, 0.5] means that 20% data sets on disk
         will have 1 replica, 30% - 2 replicas, 50% - 3 replicas.
         For set_replicas='values': list of Intensity values. For example, q=[10, 100] means that data sets on disk which have Intensity values below or equal 10
         will have 1 replica, data sets with Intensity values below or equal 100 but above 10  - 2 replicas, data sets with Intensity values above 100 - 3 replicas.
         For set_replicas='load': list of 'load' values. For example, q=[10, 100] means that data sets on disk which have 'load' values below or equal 10
         will have 1 replica, data sets with 'load' values below or equal 100 but above 10  - 2 replicas, data sets with 'load' values above 100 - 3 replicas.
         For set_replicas='origin': None (ignored).

        :param float alpha: fine for low number of replicas. Higher alpha value, higher number of data sets get >1 replicas.

        :param int max_replicas: max number of replicas.

        :return: pandas.DataFrame report. Return data sets storage placement report.
        """
        report = self.total_report.copy()
        report = self._report_upgrade(report, pop_cut=pop_cut, q=q, alpha=alpha, min_replicas=1, max_replicas=max_replicas, set_replicas=set_replicas)
        return report

    def opti_placement(self, q=None, set_replicas='auto', c_disk=100, c_tape=1, c_miss=10000, alpha=10, min_replicas=1, max_replicas=4):
        """
        Get optimized data sets storage placement report (popularity cut value optimization).

        :param str set_replicas: label of data sets number of replicas determination algorithm.
        'auto' (preferable): set data sets number of replicas by optimized values from loss function.
        'quantile': set data sets number of replicas by Intensity quantile values.
        'values': set data sets number of replicas by Intensity values.
        'load': set data sets number of replicas by 'load' values (Intensity*LFNSize).
        'origin': use origin number od replicas for data sets.

        :param list(float) q:For set_replicas='auto' (preferable): None (ignored).
         For set_replicas='quantile': list of Intensity quantile values. For example, q=[0.2, 0.5] means that 20% data sets on disk
         will have 1 replica, 30% - 2 replicas, 50% - 3 replicas.
         For set_replicas='values': list of Intensity values. For example, q=[10, 100] means that data sets on disk which have Intensity values below or equal 10
         will have 1 replica, data sets with Intensity values below or equal 100 but above 10  - 2 replicas, data sets with Intensity values above 100 - 3 replicas.
         For set_replicas='load': list of 'load' values. For example, q=[10, 100] means that data sets on disk which have 'load' values below or equal 10
         will have 1 replica, data sets with 'load' values below or equal 100 but above 10  - 2 replicas, data sets with 'load' values above 100 - 3 replicas.
         For set_replicas='origin': None (ignored).

        :param float c_disk: cost of storage of 1Gb on disk.

        :param float c_tape: cost of storage of 1Gb on tape.

        :param float c_miss: cost of restoring 1Gb from tape to disk.

        :param float alpha: fine for low number of replicas. Higher alpha value, higher number of data sets get >1 replicas.

        :param int max_replicas: max number of replicas.

        :return: pandas.DataFrame opti_report. Return optimized data sets storage placement report.
        """
        cuts, loss, min_cut, min_loss = self._get_loss_curve(self.total_report, q=q, set_replicas=set_replicas, c_disk=c_disk, c_tape=c_tape, c_miss=c_miss, alpha=alpha, min_replicas=min_replicas, max_replicas=max_replicas)
        self.total_report = self._report_upgrade(total_report=self.total_report, pop_cut=min_cut,  q=q, set_replicas=set_replicas, alpha=alpha, min_replicas=min_replicas, max_replicas=max_replicas)
        opti_report = pd.DataFrame()
        opti_report = self.total_report[['Name', 'OnDisk', 'NbReplicas']]
        return opti_report

    def plot_loss_curve(self, q=None, set_replicas='auto', c_disk=100, c_tape=1, c_miss=10000, alpha=10, min_replicas=1, max_replicas=4):
        """
        Plot loss curve.

        :param str set_replicas: label of data sets number of replicas determination algorithm.
        'auto' (preferable): set data sets number of replicas by optimized values from loss function.
        'quantile': set data sets number of replicas by Intensity quantile values.
        'values': set data sets number of replicas by Intensity values.
        'load': set data sets number of replicas by 'load' values (Intensity*LFNSize).
        'origin': use origin number od replicas for data sets.

        :param list(float) q:For set_replicas='auto' (preferable): None (ignored).
         For set_replicas='quantile': list of Intensity quantile values. For example, q=[0.2, 0.5] means that 20% data sets on disk
         will have 1 replica, 30% - 2 replicas, 50% - 3 replicas.
         For set_replicas='values': list of Intensity values. For example, q=[10, 100] means that data sets on disk which have Intensity values below or equal 10
         will have 1 replica, data sets with Intensity values below or equal 100 but above 10  - 2 replicas, data sets with Intensity values above 100 - 3 replicas.
         For set_replicas='load': list of 'load' values. For example, q=[10, 100] means that data sets on disk which have 'load' values below or equal 10
         will have 1 replica, data sets with 'load' values below or equal 100 but above 10  - 2 replicas, data sets with 'load' values above 100 - 3 replicas.
         For set_replicas='origin': None (ignored).

        :param float c_disk: cost of storage of 1Gb on disk.

        :param float c_tape: cost of storage of 1Gb on tape.

        :param float c_miss: cost of restoring 1Gb from tape to disk.

        :param float alpha: fine for low number of replicas. Higher alpha value, higher number of data sets get >1 replicas.

        :param int max_replicas: max number of replicas.

        :return: int 1.
        """
        report = self.total_report.copy()
        cuts, loss, min_cut, min_loss = self._get_loss_curve(self.total_report, q=q, set_replicas=set_replicas, c_disk=c_disk, c_tape=c_tape, c_miss=c_miss, alpha=alpha, min_replicas=min_replicas, max_replicas=max_replicas)
        print ('Min point is ', (min_cut[0], min_loss))
        plt.plot(cuts, np.log(loss))
        plt.xlabel('Popularity cuts')
        plt.ylabel('log(Loss)')
        plt.title('Loss function')
        plt.show()
        return 1

    def opti_total_size(self, opti_report):
        """
        Calculate total size of data sets on disk.

        :param pandas.DataFrame opti_report: opti_report from opti_placement method.

        :return: float total_size.
        """
        total_size = (self.data['LFNSize'].values*opti_report['NbReplicas'].values*opti_report['OnDisk'].values).sum()
        return total_size