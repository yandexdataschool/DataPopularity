from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

######################
###Version 1.0.0   ###
######################

import numpy as np
import pandas as pd
import re

from sklearn.metrics import roc_curve, auc

class Performance(object):
    """
    This class contains methods to evaluate LHCb Recommendation System's quality.
    """

    def get_roc_auc(self, report=None, period_data=None):
        """
        This method returns square under the ROC curve.
        :param pandas.dataFrame report: The Recommendation System's report with probabilities.
        :param pandas.DataFrame period_data: data for which the report was generated.
        :return: float square under the ROC curve.
        """
        fpr, tpr, _ = roc_curve(period_data['True_Type'].values.astype(np.float), report['Proba_Type_1'].values.astype(np.float))
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def get_mistakes_matrix(self, report=None, period_data=None):
        """
        This method returns number of mistakes, when a dataset was removed from disk but then was used.
        :param pandas.dataFrame report: The Recommendation System's report with probabilities.
        :param pandas.DataFrame period_data: data for which the report was generated.
        :return: int number of mistakes.
        """
        recommended_type = report['Recommended_Type'].values
        true_type = period_data['True_Type'].values
        return true_type[recommended_type==0].sum()

    def get_total_size(self, report=None, period_data=None):
        """
        This method returns total disk size of datasets.
        :param pandas.dataFrame report: The Recommendation System's report with probabilities.
        :param pandas.DataFrame period_data: data for which the report was generated.
        :return: (float, float) recommended and true(with mistakes) total disk sizes.
        """
        total_size = period_data['DiskSize'].values.sum(axis=0)
        recommended_total_disk_size = (period_data['DiskSize'])[report['Recommended_Type'].values==1].sum(axis=0)/total_size
        selection = (report['Recommended_Type'].values==1) + (period_data['True_Type'].values==1)
        true_total_disk_size = (period_data['DiskSize'])[selection].sum(axis=0)/total_size
        return recommended_total_disk_size, true_total_disk_size
