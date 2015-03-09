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

class DataBase(object):
    """
    Parameters:
    -----------
    :param str source_path: a train data file path.
    The data file must have .xls, .xlsx or .csv formats.

    :param pandas.DataFrame data: train data.

    :param int nb_of_weeks: number of weeks in data sets access history.
    """

    def __init__(self, source_path=None, data=None, nb_of_weeks=104):
        if source_path != None:
            ext = source_path.split('.')[-1]
            if ext=='csv':
                try:
                    self.data_origin = pd.read_csv(source_path)
                except:
                    print ("Can not open file.")
            elif ext=='xls' or ext=='xlsx':
                try:
                    self.data_origin = pd.read_excel(source_path)
                except:
                    print ("Can not open file.")
        else:
            self.data_origin = data

        self.periods = [str(i) for i in range(1,nb_of_weeks+1)]
        self._fix_columns()
        self.data = self._data_transform()


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

    def _data_transform(self):
        """
        Transform data
        :return: pandas.DataFrame transformed data
        """
        data1 = self.data_origin.copy()
        data2 = self.data_origin.copy()
        for i in range(0, len(self.periods)):
            if i!=0:
                data1[self.periods[i]] = self.data_origin[self.periods[i]] - self.data_origin[self.periods[i-1]]

        for i in range(0, len(self.periods)):
            k = len(self.periods)-1-i
            data2[self.periods[i]] = data1[self.periods[k]]

        for i in range(0, len(self.periods)):
            if i!=0:
                data2[self.periods[i]] = data2[self.periods[i]] + data2[self.periods[i-1]]
        self.data = data2
        return self.data

    def _check_columns(self):
        """
        Check whether all needed data columns are presence
        :return: 1 if all needed columns are presence in the train data. Otherwise, rise assertion.
        """
        cols_needed = pd.core.index.Index([u'Name', u'Configuration', u'ProcessingPass', u'FileType',
                                           u'Type', u'Creation_week', u'NbLFN', u'LFNSize', u'NbDisk', u'DiskSize', u'NbTape',
                                           u'TapeSize', u'NbArchived', u'ArchivedSize', u'Nb_Replicas', u'Nb_ArchReps',
                                           u'Storage', u'FirstUsage', u'LastUsage', u'Now'])
        cols = self.data.columns
        intersect = cols.intersection(cols_needed)
        diff_cols = cols_needed.diff(intersect)

        assert len(diff_cols)==0, str(["Please, add following columns to the data: ", list(diff_cols)])[1:-1]
        return 1
