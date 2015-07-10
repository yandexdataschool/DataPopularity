from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

######################
###Version 1.0.0   ###
######################

import numpy as np
import pandas as pd
import re

from datapop.LHCb import DataPreprocessor

class LRU(object):
    """
    This is Recommendation System for LHCb.
    :param int forecast_horizont: number of last time periods of dataset time series for which the probabilities will predict.

    :param int or list[int] class_abs_thresholds: dataset usage threshold values which is used to estimate data storage type for each dataset.
    For example, class_abs_thresholds=[a,b,c] means that all datasets which have number of usages for last forecast_horizont time periods in range [0, a)
     is stored on type 1 storage; [a, b) - type 2 storage; [b, inf) - type 3 storage.
    """

    def __init__(self, forecast_horizont=None, class_abs_thresholds=None):
        assert forecast_horizont is not None, "Forecast_horizont is missing."
        assert class_abs_thresholds is not None, "Pos_class_abs_threshold is missing."

        self.forecast_horizont = forecast_horizont
        self.class_abs_thresholds = np.array(class_abs_thresholds, ndmin=1)

    def _predict(self, data=None, forecast_horizont=None, class_abs_thresholds=None):

        columns = data.columns
        number_col_filter = re.compile("^\d+$")
        number_columns = [col for col in columns if number_col_filter.search(col)]
        fh_data_values = data[number_columns[-forecast_horizont:]].sum(axis=1).values
        type_label = 0
        for type_num in range(0, len(self.class_abs_thresholds)):
            type_label = type_label*(fh_data_values < class_abs_thresholds[type_num]) +\
                         (fh_data_values >= class_abs_thresholds[type_num])*(type_num+1)
        return type_label


    def report(self, data=None):
        """
        This method generates a recommendation report.
        :param pandas.DataFrame data: data to predict the probabilities. Raw data.
        :return: pandas.DataFrame recommendation report.
        """
        preprocessed_data = DataPreprocessor(data=data).get_preprocessed_data()
        labels = self._predict(data=preprocessed_data,\
                                                   forecast_horizont=self.forecast_horizont,\
                                                   class_abs_thresholds=self.class_abs_thresholds)
        proba_cols = ['Proba_Type_%d' % i for i in range(0,len(self.class_abs_thresholds)+1)]
        report = pd.DataFrame()
        report['ID'] = preprocessed_data['ID'].values
        for i in range(0,len(self.class_abs_thresholds)+1):
            report['Proba_Type_%d' % i] = (labels==i)*1.
        report['Recommended_Type'] = labels
        return report

    def _get_preprocessed_data(self, data=None):
        """
        Method returns the preprocessed data.
        :param pandas.DataFrame data: data to predict the probabilities. The data should contain following columns:
        'ID' - unique name for each dataset; '1', '2', '3', ... - columns for dataset usage history time series.
        :return: pandas.DataFrame preprocessed data.
        """
        return DataPreprocessor(data=data).get_preprocessed_data()

    def _report2(self, preprocessed_data=None):
        """
        This method generates a recommendation report. Preprocessed data is required.
        :param pandas.DataFrame data: data to predict the probabilities. The data should contain following columns:
        'ID' - unique name for each dataset; '1', '2', '3', ... - columns for dataset usage history time series.
        :return: pandas.DataFrame recommendation report.
        """
        labels = self._predict(data=preprocessed_data,\
                                                   forecast_horizont=self.forecast_horizont,\
                                                   class_abs_thresholds=self.class_abs_thresholds)
        proba_cols = ['Proba_Type_%d' % i for i in range(0,len(self.class_abs_thresholds)+1)]
        report = pd.DataFrame()
        report['ID'] = preprocessed_data['ID'].values
        for i in range(0,len(self.class_abs_thresholds)+1):
            report['Proba_Type_%d' % i] = (labels==i)*1.
        report['Recommended_Type'] = labels
        return report
