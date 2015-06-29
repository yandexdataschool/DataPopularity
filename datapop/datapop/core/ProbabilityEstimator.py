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
from rep.metaml import FoldingClassifier
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

class ProbabilityEstimator(object):
    """
    This class computes datasets probabilities to be stored for given type of data storages.
    Parameters:
    -----------
    :param pandas.DataFrame data: data to compute the probabilities. The data should contain following columns:
    'ID' - unique name for each dataset; '1', '2', '3', ... - columns for datasets usage history time series.

    :param int forecast_horizont: number of lastr time periods of datasets time series for which the probabilities will compute.

    :param int or list[int] class_abs_thresholds: datasets usage threshold values which is used to estimate data storage type for each dataset.
    For example, class_abs_thresholds=[a,b,c] means that all datasets which have number of usages for last forecast_horizont time periods in range [0, a)
     is stored on type 1 storage; [a, b) - type 2 storage; [b, inf) - type 3 storage.
    """

    def __init__(self, data=None, forecast_horizont=None, class_abs_thresholds=None):

        self._parameters_checker(data, forecast_horizont, class_abs_thresholds)

        self.data = data
        self.forecast_horizont = forecast_horizont
        self.class_abs_thresholds = np.array(class_abs_thresholds, ndmin=1)

        self.preprocessed_data = None #Used just for output
        self.classifier = None #Used just for output

    def _parameters_checker(self, data=None, forecast_horizont=None, class_abs_thresholds=None):
        """
        This method is used to be sure that parameters are correct.
        Parameters:
        -----------
        :param pandas.DataFrame data: data to compute the probabilities. The data should contain following columns:
        'ID' - unique name for each dataset; '1', '2', '3', ... - columns for datasets usage history time series.

        :param int forecast_horizont: number of lastr time periods of datasets time series for which the probabilities will compute.

        :param int or list[int] class_abs_thresholds: datasets usage threshold values which is used to estimate data storage type for each dataset.
        For example, class_abs_thresholds=[a,b,c] means that all datasets which have number of usages for last forecast_horizont time periods in range [0, a)
         is stored on type 1 storage; [a, b) - type 2 storage; [b, inf) - type 3 storage.
        """
        assert data is not None, "Data is missing."
        assert forecast_horizont is not None, "Forecast_horizont is missing."
        assert class_abs_thresholds is not None, "Pos_class_abs_threshold is missing."
        #Get needed columns
        columns = data.columns
        needed_col_filter = re.compile("^\d+$|(^ID$)")
        needed_columns = [col for col in columns if needed_col_filter.search(col)]

        assert len(needed_columns)>=3, "Data should contain following columns: 'ID', '1', '2', '3', ..."
        assert len(needed_columns)-1 > class_abs_thresholds, "Pos_class_abs_threshold is larger than number of points in time series."

    def _data_preprocessing(self, data=None, forecast_horizont=None, class_abs_thresholds=None):
        """
        This method is used to prepare the data for classifier.
        Parameters:
        -----------
        :param pandas.DataFrame data: data to compute the probabilities. The data should contain following columns:
        'ID' - unique name for each dataset; '1', '2', '3', ... - columns for datasets usage history time series.

        :param int forecast_horizont: number of lastr time periods of datasets time series for which the probabilities will compute.

        :param int or list[int] class_abs_thresholds: datasets usage threshold values which is used to estimate data storage type for each dataset.
        For example, class_abs_thresholds=[a,b,c] means that all datasets which have number of usages for last forecast_horizont time periods in range [0, a)
         is stored on type 1 storage; [a, b) - type 2 storage; [b, inf) - type 3 storage.

        :return pandas.DataFrame preprocessed_data: the data prepared for classifier.
        numpy.array number_columns: array of column names for the datasets time series.
        """
        #Find time series columns
        columns = data.columns
        number_col_filter = re.compile("^\d+$")
        number_columns = [col for col in columns if number_col_filter.search(col)]
        #Class labels (types of data storages) for multiclasses
        fh_data_values = data[number_columns[-forecast_horizont:]].sum(axis=1).values
        type_label = 0 #Storage type 0
        for type_num in range(0, len(self.class_abs_thresholds)):
            type_label = type_label + (fh_data_values >= class_abs_thresholds[type_num])*(type_num + 1) #Storage type type_num + 1

        train_num_cols = number_columns[:-forecast_horizont]
        preprocessed_data = pd.DataFrame(columns=['ID']+train_num_cols+['FirstUsage', 'Type'])
        preprocessed_data['ID'] = data['ID'].values
        preprocessed_data[train_num_cols] = data[train_num_cols].values
        preprocessed_data['FirstUsage'] = (data[number_columns].values.cumsum(axis=1)!=0).sum(axis=1)
        preprocessed_data['Type'] = type_label
        train_cols = train_num_cols + ['FirstUsage']
        return preprocessed_data, train_cols

    def get_probabilities(self):
        """
        This method is used to compute and get probabilities to be stored on each data storage type for each dataset.
        :return: pandas.DataFrame report: the raport contain following columns: 'ID' - unique name for each dataset;
        'Current_Type' - current data storage type for each dataset estimated on the last forecast_horizont time periods;
        'Proba_Type_i' - datasets probabilities to be stored on storage type i.
        """
        preprocessed_data, train_columns = self._data_preprocessing(self.data, self.forecast_horizont, self.class_abs_thresholds)
        self.preprocessed_data = preprocessed_data.copy() #Used just for output
        X = preprocessed_data[train_columns].astype(np.float)
        Y = preprocessed_data['Type'].values

        try:
            n_folds = 3
            folder = FoldingClassifier(GradientBoostingClassifier(learning_rate=0.02, n_estimators=2500, max_depth=6, subsample=0.8),\
                                       n_folds=n_folds, features=train_columns, random_state=42)
            folder.fit(X, Y)
            self.classifier = folder #Used just for output
        except:
            print ("Can not train classifier. Please, check data.")

        probabilities = folder.predict_proba(X)
        classes = folder.classes_

        report = pd.DataFrame()
        report['ID'] = preprocessed_data['ID'].values
        report['Current_Type'] = preprocessed_data['Type']
        #In case of multiclasses
        for i in range(0, len(classes)):
            class_i = classes[i]
            report['Proba_Type_%d' % class_i] = probabilities[:,i]
        return report




