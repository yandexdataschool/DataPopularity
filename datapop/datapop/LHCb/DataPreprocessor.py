from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

######################
###Version 1.0.0   ###
######################

import numpy as np
import pandas as pd
import re

class DataPreprocessor(object):
    """
    This class is used to preprocess LHCb data, give it a standart form.
    Parameters:
    :param pandas.DataFrame data: LHCb data.
    """

    def __init__(self, data=None):
        self.data=data

    def _rename(self, x):
        """
        :param str or int x: column's name.
        :return: str renamed column's name.
        """
        return re.sub('\W', '_', str(x))

    def _fix_columns(self, data=None):
        """
        Rename data columns
        :param pandas.DataFrame data: LHCb data.
        :return: pandas.DataFrame data: LHCb data with fixed columns.
        """
        data.columns = map(self._rename, data.columns)
        return data

    def _get_ids(self, data=None):
        """
        This method is used to get datasets ids from LHCb data.
        :param pandas.DataFrame data: LHCb data.
        :return: pandas.DataFrame: LHCb datasets names.
        """
        return data['Name']

    def _get_time_series(self, data=None):
        """
        This method is used to get dataset usage history time series from LHCb data.
        :param pandas.DataFrame data: LHCb data.
        :return: pandas.DataFrame: time series.
        numpy.array([str]): time series periods.
        """
        data = self._fix_columns(data)
        columns = data.columns
        number_col_filter = re.compile("^\d+$")
        number_columns = [col for col in columns if number_col_filter.search(col)]
        periods = number_columns

        data1 = data[['Name'] + periods].copy()
        data2 = data[['Name'] + periods].copy()
        for i in range(0, len(periods)):
            if i!=0:
                data1[periods[i]] = data[periods[i]] - data[periods[i-1]]

        for i in range(0, len(periods)):
            k = len(periods)-1-i
            data2[periods[i]] = data1[periods[k]]

        return data2, periods

    def _get_time(self, data=None):
        """
        This method is used to get 'Creation-week', 'FirstUsage' and 'Now' columns from LHCb data.
        :param pandas.DataFrame data: LHCb data.
        :return: pandas.DataFrame: time columns.
        numpy.array([str]): names of the time columns.
        """
        data = self._fix_columns(data)
        columns = data.columns
        number_col_filter = re.compile("^\d+$")
        number_columns = [col for col in columns if number_col_filter.search(col)]
        time_cols = ['Creation', 'First', 'StartTime', 'Now']
        time_data = pd.DataFrame(columns=['Name']+time_cols)
        time_data['Name'] = data['Name'].values
        time_data[['Creation', 'First', 'Now']] = data[['Creation_week', 'FirstUsage', 'Now']].values
        time_data['StartTime'] = time_data['Now'] - len(number_columns)
        return time_data, time_cols

    def _get_size(self, data=None):
        """
        This method is used to get datasets disk size from LHCb data.
        :param pandas.DataFrame data: LHCb data.
        :return: pandas.DataFrame: LHCb datasets disk size.
        """
        return data['DiskSize']

    def get_preprocessed_data(self):
        """
        This method generate preprocessed LHCb data in standart form.
        :return: pandas.DataFrame: preprocessed data.
        """
        time_series, periods = self._get_time_series(self.data)
        time_data, time_cols = self._get_time(self.data)
        preprocessed_data = pd.DataFrame(columns=['ID', 'DiskSize']+time_cols+periods)
        preprocessed_data['ID'] = self._get_ids(self.data).values
        preprocessed_data['DiskSize'] = self._get_size(self.data).values
        preprocessed_data[time_cols] = time_data[time_cols].values
        preprocessed_data[periods] = time_series[periods].values
        return preprocessed_data