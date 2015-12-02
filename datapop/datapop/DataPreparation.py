from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

#version 3.0

import numpy
import pandas
import re

class DataPreparation(object):
    """
    This class is used to prepare LHCb data for further analysis.
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

    def _get_access_history(self, data=None):
        """
        This method is used to get dataset access history from LHCb data.
        :param pandas.DataFrame data: LHCb data.
        :return: pandas.DataFrame: access history.
        """
        data = self._fix_columns(data)
        columns = data.columns
        number_col_filter = re.compile("^\d+$")
        number_columns = [col for col in columns if number_col_filter.search(col)]
        periods = number_columns

        original_access_history = data[['Name'] + periods].copy()
        access_history = data[['Name'] + periods].copy()
        for i in range(0, len(periods)):
            if i!=0:
                original_access_history[periods[i]] = data[periods[i]] - data[periods[i-1]]

        for i in range(0, len(periods)):
            k = len(periods)-1-i
            access_history[periods[i]] = original_access_history[periods[k]]

        return access_history

    def _get_metadata(self, data=None):
        """
        This method is used to get dataset metadata from LHCb data.
        :param pandas.DataFrame data: LHCb data.
        :return: pandas.DataFrame: metadata.
        """

        data = self._fix_columns(data)
        columns = data.columns
        number_col_filter = re.compile("^\d+$")
        metadata_columns = [col for col in columns if not number_col_filter.search(col)]

        metadata = data[metadata_columns]
        return metadata



    def preparation(self):
        """
        Get datasets metadata and access history from LHCv data.
        :return: pandas.DataFrame: metadata;
        pandas.DataFrame: access_history.
        """
        metadata = self._get_metadata(data=self.data)
        access_history = self._get_access_history(data=self.data)
        return metadata, access_history