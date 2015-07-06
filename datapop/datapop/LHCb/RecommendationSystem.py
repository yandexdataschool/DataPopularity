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

    def __init__(self, forecast_horizont=None, class_abs_thresholds=None, risk_matrix=None):
        assert forecast_horizont is not None, "Forecast_horizont is missing."
        assert class_abs_thresholds is not None, "Pos_class_abs_threshold is missing."
        assert risk_matrix is not None, "Risk_matrix are missing."

        self.forecast_horizont = forecast_horizont
        self.class_abs_thresholds = np.array(class_abs_thresholds, ndmin=1)
        self.risk_matrix = risk_matrix

    def report(self, data=None):
        preprocessed_data = DataPreprocessor(data=data).get_preprocessed_data()
        probability_report = ProbabilityEstimator(data=preprocessed_data,\
                                           forecast_horizont=self.forecast_horizont,\
                                           class_abs_thresholds=self.class_abs_thresholds).get_probabilities()
        report = DataDistribution().risk_minimization(probability_report=probability_report,\
                                                      risk_matrix=self.risk_matrix)
        return report

    def _get_preprocessed_data(self, data=None):
        return DataPreprocessor(data=data).get_preprocessed_data()

    def _simulation_report(self, preprocessed_data=None):
        probability_report = ProbabilityEstimator(data=preprocessed_data,\
                                           forecast_horizont=self.forecast_horizont,\
                                           class_abs_thresholds=self.class_abs_thresholds).get_probabilities()
        report = DataDistribution().risk_minimization(probability_report=probability_report,\
                                                      risk_matrix=self.risk_matrix)
        return report
