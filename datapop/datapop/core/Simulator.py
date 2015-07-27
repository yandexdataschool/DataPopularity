from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

######################
###Version 1.0.0   ###
######################

import numpy as np
import pandas as pd
import re

class Simulator(object):
    """
    This class simulates work of data storage system.
    """

    def generate_data(self, data=None, size=None):
        """
        This method generates datasets with their usage history.
        :param pandas.DataFrame data: dataset usage history.
        :param int size: number of datasets in generated data.
        :return: new history of dataset usage.
        """
        index = data.index
        choice = np.random.choice(index, size)
        generated_data = data.irow(choice)
        return generated_data

    def get_period(self, data=None, period=None, forecast_horizont=None, class_abs_thresholds=None):
        """
        This method generates dataset usage history for the defined period and their true label for the forecast horizont.
        :param pandas.DataFrame data:full dataset usage history.
        :param int or srt period: period of time of a dataset's usage history.
        :param int forecast_horizont: forecast horizont that is used in simulation. If None, true label will not be calculated.
        :param int or list[int] class_abs_thresholds: dataset usage threshold values which is used to estimate data storage type for each dataset.
        For example, class_abs_thresholds=[a,b,c] means that all datasets which have number of usages for last forecast_horizont time periods in range [0, a)
        is stored on type 1 storage; [a, b) - type 2 storage; [b, inf) - type 3 storage. If None, true label will not be calculated.
        :return: pandas.DataFrame of the usage history at the period;
        """

        columns = data.columns
        number_col_filter = re.compile("^\d+$")
        number_columns = [col for col in columns if number_col_filter.search(col)]
        drop_cols = number_columns[number_columns.index(str(period))+1:]

        period_data = data.drop(drop_cols, axis=1)
        period_data['Now'] = data['Now'] - len(drop_cols)
        first = period_data['First']
        creation = period_data['Creation']
        now = period_data['Now']
        avg_delay = (first - creation)[creation <= first].mean() + 2*(first - creation)[creation <= first].std()#!!!
        selection = (creation <= first)*(first + 3 < now)# + (creation+avg_delay < now)*(creation + 3 < now)# + (first + 10 < now)#!!!!!
        #selection = period_data[number_columns[:number_columns.index(str(period))+1]].sum(axis=1)!=0
        period_data = period_data[selection]

        if forecast_horizont!=None and class_abs_thresholds!=None:
            class_abs_thresholds = np.array(class_abs_thresholds, ndmin=1)
            forecast_cols = number_columns[number_columns.index(str(period))+1:number_columns.index(str(period))+1+forecast_horizont]
            cols_sum = (data[forecast_cols])[selection].sum(axis=1).values
            labels = 0
            for type_num in range(0, len(class_abs_thresholds)):
                labels = labels*(cols_sum < class_abs_thresholds[type_num]) + (cols_sum >= class_abs_thresholds[type_num])*(type_num + 1)
            period_data['True_Type'] = labels
        return period_data
