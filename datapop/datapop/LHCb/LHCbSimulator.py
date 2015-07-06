from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

######################
###Version 1.0.0   ###
######################

import numpy as np
import pandas as pd
import re

from datapop.core import Simulator, Performance
from datapop.LHCb import DataPreprocessor

class LHCbSimulator(Simulator):

    def __init__(self, data=None, begin=None, step=None, recommendation_system=None):

        preprocessed_data = DataPreprocessor(data=data).get_preprocessed_data()
        self.data = self._generate_data(preprocessed_data, preprocessed_data.shape[0])
        self.begin = int(begin)
        self.step = int(step)
        self.recommendation_system = recommendation_system
        self.forecast_horizont = recommendation_system.forecast_horizont
        self.class_abs_thresholds = recommendation_system.class_abs_thresholds

        columns = data.columns
        number_col_filter = re.compile("^\d+$")
        number_columns = [col for col in columns if number_col_filter.search(col)]

        self.max_period = int(number_columns[-1])

    def _generate_data(self, data=None, size=None):
        return super(LHCbSimulator, self).generate_data(data=data, size=size)

    def _data_at_period(self, data=None, period=None):
        return super(LHCbSimulator, self).data_at_period(data=data, period=period)

    def _true_lables_at_period(self, data=None, period=None, forecast_horizont=None, class_abs_thresholds=None):
        columns = data.columns
        number_col_filter = re.compile("^\d+$")
        number_columns = [col for col in columns if number_col_filter.search(col)]
        forecast_cols = number_columns[number_columns.index(str(period))+1:number_columns.index(str(period))+1+forecast_horizont]
        forecast_data = data[forecast_cols].sum(axis=1).values
        type_label = 0 #Storage type 0
        for type_num in range(0, len(self.class_abs_thresholds)):
            type_label = type_label + (forecast_data >= class_abs_thresholds[type_num])*(1) #Storage type type_num + 1 TODO

        report = pd.DataFrame(columns=['ID', 'Type'])
        report['ID'] = data['ID'].values
        report['Type'] = type_label
        return report


    def _get_reports(self):
        reports = []
        true_labels = []
        periods = []
        roc_aucs = []
        mistakes = []
        pr = Performance()
        current_period=self.begin
        while current_period <= self.max_period - self.forecast_horizont:
            current_period_data = self._data_at_period(data=self.data.copy(),\
                                                       period=current_period)
            current_period_report = self.recommendation_system._simulation_report(current_period_data)
            true_label = self._true_lables_at_period(self.data.copy().irow(current_period_data.index),\
                                                                     current_period,\
                                                     self.forecast_horizont, self.class_abs_thresholds)
            #roc_auc = pr.get_roc_auc(report=current_period_report, true_labels=true_label)
            mistake = pr.get_mistakes_matrix(report=current_period_report, true_labels=true_label)
            reports.append(current_period_report)
            true_labels.append(true_label)
            periods.append(current_period)
            #roc_aucs.append(roc_auc)
            mistakes.append(mistake)
            current_period = current_period + self.step
        return reports, true_labels, roc_aucs, mistakes, periods
