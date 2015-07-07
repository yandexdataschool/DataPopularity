from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

######################
###Version 1.0.0   ###
######################

import pandas as pd
import re

from datapop.core import Simulator
from datapop.LHCb import DataPreprocessor
from datapop.LHCb import Performance


class LHCbSimulator(Simulator):

    def __init__(self, data=None, begin=None, step=None, recommendation_system=None):

        preprocessed_data = DataPreprocessor(data=data).get_preprocessed_data()
        self.data = self._generate_data(preprocessed_data, preprocessed_data.shape[0])
        self.begin = int(begin)
        self.step = int(step)
        self.recommendation_system = recommendation_system

        columns = data.columns
        number_col_filter = re.compile("^\d+$")
        number_columns = [col for col in columns if number_col_filter.search(col)]

        self.max_period = int(number_columns[-1])

    def _generate_data(self, data=None, size=None):
        return super(LHCbSimulator, self).generate_data(data=data, size=size)

    def _get_period(self, data=None, period=None, forecast_horizont=None, class_abs_thresholds=None):
        return super(LHCbSimulator, self).get_period(data=data, period=period, forecast_horizont=forecast_horizont,\
                                                     class_abs_thresholds=class_abs_thresholds)

    def simulation_report(self):
        periods = []
        roc_aucs = []
        mistakes = []
        recommended_total_disk_sizes = []
        true_total_disk_sizes = []

        performance = Performance.Performance()

        current_period=self.begin
        while current_period <= self.max_period - self.recommendation_system.forecast_horizont:

            period_data = self._get_period(data=self.data.copy(),\
                                                       period=current_period,\
                                                       forecast_horizont=self.recommendation_system.forecast_horizont,\
                                                       class_abs_thresholds=self.recommendation_system.class_abs_thresholds)
            period_report = self.recommendation_system._report2(period_data)

            roc_auc = performance.get_roc_auc(report=period_report, period_data=period_data)
            roc_aucs.append(roc_auc)

            mistake = performance.get_mistakes_matrix(report=period_report, period_data=period_data)
            mistakes.append(mistake)

            recommended_total_disk_size, true_total_disk_size = performance.get_total_size(report=period_report, period_data=period_data)
            recommended_total_disk_sizes.append(recommended_total_disk_size)
            true_total_disk_sizes.append(true_total_disk_size)

            periods.append(current_period)

            current_period = current_period + self.step

        simulation_report = pd.DataFrame(columns=['Periods', 'ROC_AUC', 'Mistakes', 'Recommended_total_disk_size', 'True_total_disk_size'])
        simulation_report['Periods'] = periods
        simulation_report['ROC_AUC'] = roc_aucs
        simulation_report['Mistakes'] = mistakes
        simulation_report['Recommended_total_disk_size'] = recommended_total_disk_sizes
        simulation_report['True_total_disk_size'] = true_total_disk_sizes

        return simulation_report
