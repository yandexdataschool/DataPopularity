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

    def get_roc_auc(self, report=None, true_labels=None):
        fpr, tpr, _ = roc_curve(true_labels['Type'].values, report['Proba_Type_1'].values)
        roc_auc = auc(fpr, tpr)
        return roc_auc

    def get_mistakes_matrix(self, report=None, true_labels=None):
        recommended_type = report['Recommended_Type'].values
        true_type = true_labels['Type'].values
        return true_type[recommended_type==0].sum()

    def get_total_size(self, report=None, true_labels=None):
        return 0
