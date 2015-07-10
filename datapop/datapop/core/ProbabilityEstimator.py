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
    This class predicts future dataset probabilities to be stored on given types of data storages.
    Parameters:
    -----------
    :param pandas.DataFrame data: data to predict the probabilities. The data should contain following columns:
    'ID' - unique name for each dataset; '1', '2', '3', ... - columns for dataset usage history time series.

    :param int forecast_horizont: number of last time periods of dataset time series for which the probabilities will predict.

    :param int or list[int] class_abs_thresholds: dataset usage threshold values which is used to estimate data storage type for each dataset.
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
        'ID' - unique name for each dataset; '1', '2', '3', ... - columns for dataset usage history time series.

        :param int forecast_horizont: number of last time periods of dataset time series for which the probabilities will compute.

        :param int or list[int] class_abs_thresholds: dataset usage threshold values which is used to estimate data storage type for each dataset.
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
        'ID' - unique name for each dataset; '1', '2', '3', ... - columns for dataset usage history time series.

        :param int forecast_horizont: number of last time periods of dataset time series for which the probabilities will compute.

        :param int or list[int] class_abs_thresholds: datasets usage threshold values which is used to estimate data storage type for each dataset.
        For example, class_abs_thresholds=[a,b,c] means that all datasets which have number of usages for last forecast_horizont time periods in range [0, a)
         is stored on type 1 storage; [a, b) - type 2 storage; [b, inf) - type 3 storage.

        :return pandas.DataFrame preprocessed_data: the data prepared for classifier.
        numpy.array number_columns: array of column names for the dataset time series.
        """
        #Find time series columns
        columns = data.columns
        number_col_filter = re.compile("^\d+$")
        number_columns = [col for col in columns if number_col_filter.search(col)]
        #Class labels (types of data storages) for multiclasses
        fh_data_values = data[number_columns[-forecast_horizont:]].sum(axis=1).values
        type_label = 0 #Storage type 0
        for type_num in range(0, len(self.class_abs_thresholds)):
            type_label = type_label*(fh_data_values < class_abs_thresholds[type_num]) +\
                         (fh_data_values >= class_abs_thresholds[type_num])*(type_num+1) #Storage type type_num + 1 TODO

        train_num_cols = number_columns[:-forecast_horizont]
        train_data = pd.DataFrame(columns=['ID']+train_num_cols+['FirstUsage', 'Type'])
        train_data['ID'] = data['ID'].values
        train_data[train_num_cols] = data[train_num_cols].values
        train_data['FirstUsage'] = (data[number_columns].values.cumsum(axis=1)!=0).sum(axis=1)
        train_data['Type'] = type_label
        train_cols = train_num_cols + ['FirstUsage']

        test_num_cols = number_columns[forecast_horizont:]
        test_data = pd.DataFrame(columns=['ID']+test_num_cols+['FirstUsage'])
        test_data['ID'] = data['ID'].values
        test_data[test_num_cols] = data[test_num_cols].values
        test_data['FirstUsage'] = (data[number_columns].values.cumsum(axis=1)!=0).sum(axis=1)#+\
                                  #self.forecast_horizont*((data[number_columns].values.cumsum(axis=1)!=0).sum(axis=1)!=0)#!!!!!!!
        test_cols = test_num_cols + ['FirstUsage']

        return train_data, train_cols, test_data, test_cols

    def _get_current_probabilities(self):
        """
        This method is used to estimate current probabilities to be stored on each data storage type for each dataset.
        :return: pandas.DataFrame report: the report contains following columns: 'ID' - unique name for each dataset;
        'Current_Type' - current data storage type for each dataset estimated on the last forecast_horizont time periods;
        'Proba_Type_i' - datasets probabilities to be stored on storage type i.
        """
        train_data, train_columns, _, _ = self._data_preprocessing(self.data, self.forecast_horizont, self.class_abs_thresholds)
        X = train_data[train_columns].astype(np.float).values
        Y = train_data['Type'].values

        try:
            n_folds = 3
            folder = FoldingClassifier(GradientBoostingClassifier(learning_rate=0.02, n_estimators=2500, max_depth=6, subsample=0.8),\
                                       n_folds=n_folds, features=None, random_state=42)
            folder.fit(X, Y)
        except:
            print ("Can not train classifier. Please, check data.")

        probabilities = folder.predict_proba(X)
        classes = folder.classes_

        report_proba_cols = ['Proba_Type_%d' % classes[i] for i in range(0, len(classes))]
        report = pd.DataFrame(columns=['ID', 'Current_Type']+report_proba_cols)
        report['ID'] = train_data['ID'].values
        report['Current_Type'] = train_data['Type']
        report[report_proba_cols] = probabilities
        return report


    def _test_future_proba(self):

        test_data, test_columns, _, _ = self._data_preprocessing(self.data, self.forecast_horizont, self.class_abs_thresholds)
        X_test = test_data[test_columns[self.forecast_horizont:]].astype(np.float).values
        Y_test = test_data['Type'].values
        self.test = test_data

        train_data, train_columns, _, _ = self._data_preprocessing(test_data[['ID']+test_columns], self.forecast_horizont, self.class_abs_thresholds)
        X_train = train_data[train_columns].astype(np.float).values
        Y_train = train_data['Type'].values
        self.train = train_data

        n_folds = 2
        folder = FoldingClassifier(GradientBoostingClassifier(learning_rate=0.02, n_estimators=2500, max_depth=6, subsample=0.8),\
                                       n_folds=n_folds, features=None, random_state=42)
        folder.fit(X_train, Y_train)

        train_probabilities = folder.predict_proba(X_train)
        fpr_train, tpr_train, _ = roc_curve(Y_train, train_probabilities[:,1], pos_label=None, sample_weight=None)
        roc_auc_train = auc(fpr_train, tpr_train)

        test_probabilities = folder.predict_proba(X_test)
        fpr_test, tpr_test, _ = roc_curve(Y_test, test_probabilities[:,1], pos_label=None, sample_weight=None)
        roc_auc_test = auc(fpr_test, tpr_test)

        classes = folder.classes_

        train_report_proba_cols = ['Proba_Type_%d' % classes[i] for i in range(0, len(classes))]
        train_report = pd.DataFrame(columns=['ID', 'Type']+train_report_proba_cols)
        train_report['ID'] = train_data['ID'].values
        train_report['Type'] = train_data['Type'].values
        train_report[train_report_proba_cols] = train_probabilities

        test_report_proba_cols = ['Proba_Type_%d' % classes[i] for i in range(0, len(classes))]
        test_report = pd.DataFrame(columns=['ID', 'Type']+test_report_proba_cols)
        test_report['ID'] = test_data['ID'].values
        test_report['Type'] = test_data['Type'].values
        test_report[test_report_proba_cols] = test_probabilities

        return roc_auc_train, roc_auc_test, train_report, test_report

    def get_probabilities(self):
        """
        This method is used to predict probabilities to be stored on each data storage type for each dataset.
        :return: pandas.DataFrame report: the report contains following columns: 'ID' - unique name for each dataset;
        'Current_Type' - current data storage type for each dataset estimated on the last forecast_horizont time periods;
        'Proba_Type_i' - datasets probabilities to be stored on storage type i.
        """

        train_data, train_columns, test_data, test_columns = self._data_preprocessing(self.data, self.forecast_horizont, self.class_abs_thresholds)
        X_train = train_data[train_columns].astype(np.float).values
        Y_train = train_data['Type'].values
        X_test = test_data[test_columns].astype(np.float).values
        self.train2= train_data
        self.test2= test_data

        n_folds = 3
        folder = FoldingClassifier(GradientBoostingClassifier(learning_rate=0.02, n_estimators=2500, max_depth=6, subsample=0.8),\
                                       n_folds=n_folds, features=None, random_state=42)
        folder.fit(X_train, Y_train)
        train_probabilities = folder.predict_proba(X_train)
        test_probabilities = folder.predict_proba(X_test)
        classes = folder.classes_

        train_report_proba_cols = ['Proba_Type_%d' % classes[i] for i in range(0, len(classes))]
        train_report = pd.DataFrame(columns=['ID', 'Current_Type']+train_report_proba_cols)
        train_report['ID'] = train_data['ID'].values
        train_report['Current_Type'] = train_data['Type'].values
        train_report[train_report_proba_cols] = train_probabilities
        self.train_report = train_report

        test_report_proba_cols = ['Proba_Type_%d' % classes[i] for i in range(0, len(classes))]
        test_report = pd.DataFrame(columns=['ID']+test_report_proba_cols)
        test_report['ID'] = test_data['ID'].values
        test_report[test_report_proba_cols] = test_probabilities
        self.test_report = test_report

        return train_report



