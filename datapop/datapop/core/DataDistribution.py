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

class DataDistribution(object):
    """
    This class contains methods for data distribution over different data storage types.
    """

    def __init__(self):
        pass

    def probability_threshold(self, probability_report, proba_thresholds):
        """
        This method estimates the datasets data storage type based on dataset popularities to be on defined data storage in future.
        :param pandas.DataFrame probability_report: probability report from ProbabilityEstimator.
        :param list[floats] proba_thresholds: probability threshold values for every data storage type.
        For example, [0.95, 0.05] means that all datasets with Proba_Type_0>=0.95 will be stored on storage type 0,
        and all datasets with Proba_Type_1>=0.05 will be stored on storage type 1.
        :return: pandas.DataFrame probability report with estimated data storage types.
        """

        class_abs_thresholds = np.array(proba_thresholds, ndmin=1)

        assert probability_report is not None, "Probabilities report is missing."
        assert proba_thresholds is not None, "Proba_thresholds are missing."
        columns = probability_report.columns
        needed_col_filter = re.compile("^Proba_Type_\d+$|(^ID$)")
        needed_columns = [col for col in columns if needed_col_filter.search(col)]
        assert len(needed_columns)>=2, "Data should contain following columns: 'ID', 'Proba_Type_0', 'Proba_Type_1', 'Proba_Type_2', ..."

        columns = probability_report.columns
        probas_col_filter = re.compile("^Proba_Type_\d+$")
        probas_columns = [col for col in columns if probas_col_filter.search(col)]
        labels = 0
        for i in range(0,len(probas_columns)):
            proba = probability_report[probas_columns[i]].values
            threshold = proba_thresholds[i]
            labels = labels*(proba < threshold)*i + (proba >= threshold)*i

        probability_report['Recommended_Type'] = labels
        return probability_report

    def risk_minimization(self, probability_report, risk_matrix):
        """
        This method estimates the datasets data storage type based on minimal risk value.
        :param pandas.DataFrame probability_report: probability report from ProbabilityEstimator.
        :param numpy.ndarray(floats or np.array) or numpy.matrix risk_matrix: payoff matrix [num_types, num_types].
        Payoff matrix means how much we should pay for a decision. For example, in [[0, 100], [1, 0]] 100 means
        that we will pay 100 when we decide to store a dataset on type 0, but it should be stored on type 1.
        :return: pandas.DataFrame with probabilities, risk values for all decisions and estimated data storage types.
        """

        assert probability_report is not None, "Probabilities report is missing."
        assert risk_matrix is not None, "Risk_matrix are missing."
        columns = probability_report.columns
        needed_col_filter = re.compile("^Proba_Type_\d+$|(^ID$)")
        needed_columns = [col for col in columns if needed_col_filter.search(col)]
        assert len(needed_columns)>=2, "Data should contain following columns: 'ID', 'Proba_Type_0', 'Proba_Type_1', 'Proba_Type_2', ..."

        columns = probability_report.columns
        probas_col_filter = re.compile("^Proba_Type_\d+$")
        probas_columns = [col for col in columns if probas_col_filter.search(col)]

        risk_cols = ['Risk_Type_%d' % i for i in range(0, len(probas_columns))]
        risk_table = pd.DataFrame(columns=[['ID']+probas_columns+risk_cols])
        risk_table['ID'] = probability_report['ID'].values
        risk_table[probas_columns] = probability_report[probas_columns].values
        risk_values = np.array(np.matrix(probability_report[probas_columns].values)*np.matrix(risk_matrix))
        risk_table[risk_cols] = risk_values

        labels_table = np.array(probability_report.shape[0]*[[i for i in range(0, len(probas_columns))]])
        risk_table['Recommended_Type'] = risk_values.argmin(axis=1)

        return risk_table

