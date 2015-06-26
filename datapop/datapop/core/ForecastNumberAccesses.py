from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

######################
###Version 1.0.0     ###
######################
###Using REP v0.6  ###
######################

import numpy as np
import pandas as pd
import re

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

try:
    from kernel_regression import KernelRegression
except ImportError as e:
    raise ImportError("Install kernel_regression from www or from 'Packages/' folder. ")

class ForecastNumberAccesses(object):
    """
    This class forecasts future number of accesses to aech dataset.
    Parameters:
    -----------
    :param pandas.DataFrame data: data to compute the probabilities. The data should contain following columns:
    'ID' - unique name for each dataset; '1', '2', '3', ... - columns for datasets usage history time series.

    :param int forecast_horizont: number of lastr time periods of datasets time series for which the forecast will compute.

    :param int or list[int] class_abs_thresholds: datasets usage threshold values which is used to estimate data storage type for each dataset.
    For example, class_abs_thresholds=[a,b,c] means that all datasets which have number of usages for last forecast_horizont time periods in range [0, a)
     is stored on type 1 storage; [a, b) - type 2 storage; [b, inf) - type 3 storage.
    """

    def __init__(self, data=None, forecast_horizont=None, class_abs_thresholds=None):

        self._parameters_checker(data, forecast_horizont, class_abs_thresholds)

        self.data = data
        self.forecast_horizont = int(forecast_horizont)
        self.class_abs_thresholds = np.array(class_abs_thresholds, ndmin=1)

    def _parameters_checker(self, data=None, forecast_horizont=None, class_abs_thresholds=None):
        """
        This method is used to be sure that parameters are correct.
        Parameters:
        Parameters:
        -----------
        :param pandas.DataFrame data: data to compute the probabilities. The data should contain following columns:
        'ID' - unique name for each dataset; '1', '2', '3', ... - columns for datasets usage history time series.

        :param int forecast_horizont: number of lastr time periods of datasets time series for which the forecast will compute.

        :param int or list[int] class_abs_thresholds: datasets usage threshold values which is used to estimate data storage type for each dataset.
        For example, class_abs_thresholds=[a,b,c] means that all datasets which have number of usages for last forecast_horizont time periods in range [0, a)
         is stored on type 1 storage; [a, b) - type 2 storage; [b, inf) - type 3 storage.
        """
        assert data is not None, "Data is missing."
        assert forecast_horizont is not None, "Forecast_horizont is missing."
        #Get needed columns
        columns = data.columns
        needed_col_filter = re.compile("^\d+$|(^ID$)")
        needed_columns = [col for col in columns if needed_col_filter.search(col)]

        assert len(needed_columns)>=3, "Data should contain following columns: 'ID', '1', '2', '3', ..."

    def _data_preprocessing(self, data=None):
        """
        This method is used to prepare the data for classifier.
        Parameters:
        -----------
        :param pandas.DataFrame data: data to forecast. The data should contain following columns:
        'ID' - unique name for each dataset; '1', '2', '3', ... - columns for datasets usage history time series.

        :return pandas.DataFrame preprocessed_data: the prepared data.
        numpy.array number_columns: array of column names for the datasets time series.
        """
        #Find time series columns
        columns = data.columns
        number_col_filter = re.compile("^\d+$")
        number_columns = [col for col in columns if number_col_filter.search(col)]

        preprocessed_data = pd.DataFrame(columns=['ID']+number_columns)
        preprocessed_data['ID'] = data['ID'].values
        preprocessed_data[number_columns] = data[number_columns].values
        return preprocessed_data, number_columns

    def _get_window(self, y):
        """
        The method is used to estimate window width for rolling mean.
        :param numpy.array(floats) y: time series values for one dataset.
        :return: int: window width.
        """
        nonzero_inds = y.nonzero()[0]
        if len(nonzero_inds)==0:
            return 1
        else:
            return (nonzero_inds[-1] - nonzero_inds[0])//(len(nonzero_inds) - 1) + 1

    def _one_time_series_forecast(self, y, forecast_horizont):
        """
        The method is used to make forecast for one dataset.
        :param numpy.array(floats) y: time series values for one dataset.
        :param int or list[int] class_abs_thresholds: datasets usage threshold values which is used to estimate data storage type for each dataset.
        For example, class_abs_thresholds=[a,b,c] means that all datasets which have number of usages for last forecast_horizont time periods in range [0, a)
         is stored on type 1 storage; [a, b) - type 2 storage; [b, inf) - type 3 storage.
        :return: numpy.array(float): time series which is fitted by forecast model.
        """

        x = np.array(range(0,len(y)))
        kr = KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
        y_kr = kr.fit(x, y).predict(x)

        window = self._get_window(y)
        y_rm = pd.rolling_mean(y_kr, window=window, axis=1)

        y_pred = np.array(forecast_horizont*[y_rm[-1]])
        return np.concatenate((y_rm, y_pred), axis=0)

    def get_forecast_report(self):
        """
        Get forecast report for all datasets.
        :return: pandas.DataFrame: forecast report.
        """

        preprocessed_data, number_columns = self._data_preprocessing(self.data)
        preprocessed_values = preprocessed_data[number_columns].values
        forecast_values = np.zeros((preprocessed_values.shape[0], preprocessed_values.shape[1] + self.forecast_horizont))
        for row in preprocessed_values.shape[0]:
            y = preprocessed_values[row, :]
            forecast_values[row, :] = self._one_time_series_forecast(y, self.forecast_horizont)

        forecast_cols = [str(int(number_columns[-1]) + i) for i in range(1, self.forecast_horizont+1)]
        forecast_data = pd.DataFrame(columns=['ID']+number_columns+forecast_cols)
        forecast_data['ID'] = preprocessed_data['ID'].values
        forecast_data[number_columns+forecast_cols] = forecast_values

        if self.class_abs_thresholds != None:
            #Class labels (types of data storages) for multiclasses
            fh_data_values = forecast_data[forecast_cols].sum(axis=1).values
            type_label = 0 #Storage type 0
            for type_num in range(0, len(self.class_abs_thresholds)):
                type_label = type_label + (fh_data_values >= self.class_abs_thresholds[type_num])*(type_num + 1) #Storage type type_num + 1
            forecast_data['Predict_Type'] = type_label
            self.forecast_data = forecast_data
            return forecast_data[['ID']+forecast_cols+['Predict_Type']]
        else:
            self.forecast_data = forecast_data
            return forecast_data[['ID']+forecast_cols]


