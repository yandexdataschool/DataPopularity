from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

######################
###Version 1.0.0   ###
######################

import numpy as np
import pandas as pd
import re

from datapop.core import ProbabilityEstimator, DataDistribution
from datapop.LHCb import DataPreprocessor

class RecommendationSystem(object):
    """
    This is Recommendation System for LHCb.
    :param int forecast_horizont: number of last time periods of dataset time series for which the probabilities will predict.

    :param int or list[int] class_abs_thresholds: dataset usage threshold values which is used to estimate data storage type for each dataset.
    For example, class_abs_thresholds=[a,b,c] means that all datasets which have number of usages for last forecast_horizont time periods in range [0, a)
     is stored on type 1 storage; [a, b) - type 2 storage; [b, inf) - type 3 storage.

    :param numpy.ndarray(floats or np.array) or numpy.matrix risk_matrix: payoff matrix [num_types, num_types].
    Payoff matrix means how much we should pay for a decision. For example, in [[0, 100], [1, 0]] 100 means
    that we will pay 100 when we decide to store a dataset on type 0, but it should be stored on type 1.
    """

    def __init__(self, forecast_horizont=None, class_abs_thresholds=None, risk_matrix=None):
        assert forecast_horizont is not None, "Forecast_horizont is missing."
        assert class_abs_thresholds is not None, "Pos_class_abs_threshold is missing."
        assert risk_matrix is not None, "Risk_matrix are missing."

        self.forecast_horizont = forecast_horizont
        self.class_abs_thresholds = np.array(class_abs_thresholds, ndmin=1)
        self.risk_matrix = risk_matrix

    def report(self, data=None):
        """
        This method generates a recommendation report.
        :param pandas.DataFrame data: data to predict the probabilities. Raw data.
        :return: pandas.DataFrame recommendation report.
        """
        preprocessed_data = DataPreprocessor(data=data).get_preprocessed_data()
        probability_report = ProbabilityEstimator(data=preprocessed_data,\
                                           forecast_horizont=self.forecast_horizont,\
                                           class_abs_thresholds=self.class_abs_thresholds).get_probabilities()
        self.probability_report = probability_report
        risk_report = DataDistribution().risk_minimization(probability_report=probability_report,\
                                                      risk_matrix=self.risk_matrix)
        self.risk_report = risk_report
        report = DataDistribution().lhcb_conservative(probability_report, risk_report)
        return report

    def _get_preprocessed_data(self, data=None):
        """
        Method returns the preprocessed data.
        :param pandas.DataFrame data: data to predict the probabilities. The data should contain following columns:
        'ID' - unique name for each dataset; '1', '2', '3', ... - columns for dataset usage history time series.
        :return: pandas.DataFrame preprocessed data.
        """
        return DataPreprocessor(data=data).get_preprocessed_data()

    def _report2(self, preprocessed_data=None):
        """
        This method generates a recommendation report. Preprocessed data is required.
        :param pandas.DataFrame data: data to predict the probabilities. The data should contain following columns:
        'ID' - unique name for each dataset; '1', '2', '3', ... - columns for dataset usage history time series.
        :return: pandas.DataFrame recommendation report.
        """
        probability_report = ProbabilityEstimator(data=preprocessed_data,\
                                           forecast_horizont=self.forecast_horizont,\
                                           class_abs_thresholds=self.class_abs_thresholds).get_probabilities()
        risk_report = DataDistribution().risk_minimization(probability_report=probability_report,\
                                                      risk_matrix=self.risk_matrix)
        report = DataDistribution().lhcb_conservative(probability_report, risk_report)
        return report
