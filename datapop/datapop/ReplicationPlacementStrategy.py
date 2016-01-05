from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

# datapop 3.0.0

import numpy
import pandas
from datapop import AccessProbabilityPrediction, NumberAccessPrediction, DataPreparation

# TODO: what about Tiers? what about number of replicas?

class ReplicationPlacementStrategy(object):

    def __init__(self, data=None, min_replicas=1, max_replicas=7):
        """
        For data replication and placement
        :param pandas.DataFrame data: full data
        :param int min_replicas: minimum number of datasets replicas
        :param int max_replicas: maximum number of the datasets replicas
        :return:
        """

        self.data = data
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas

        self.flag1 = 0
        self.flag2 = 0
        self.flag3 = 0

    def get_combine_report(self, data):
        """
        Combine probability and prediction reports
        :param pandas.DataFrame data: full data
        :return: pandas.DataFrame: combine report
        """

        metadata, access_history = \
            DataPreparation(data=data).preparation()

        proba_report = AccessProbabilityPrediction(metadata,
                                                   access_history,
                                                   forecast_horizont=26).predict()

        predict_report = NumberAccessPrediction(metadata,
                                                access_history,
                                                forecast_horizont=13).predict()

        report = pandas.merge(proba_report, predict_report, how='inner', on='Name')

        report = pandas.merge(report, metadata[['Nb_Replicas', 'LFNSize', 'Name']], how='inner', on='Name')

        return report

    def get_full_combine_report(self, data):
        """
        Combine probability and prediction reports
        :param pandas.DataFrame data: full data
        :return: pandas.DataFrame: combine report
        """

        metadata, access_history = \
            DataPreparation(data=data).preparation()

        app = AccessProbabilityPrediction(metadata,
                                          access_history,
                                          forecast_horizont=26)

        proba_report = app.predict()

        _, _, features = app.data_preprocessing(app.metadata,
                                          app.access_history,
                                          app.forecast_horizont)


        predict_report = NumberAccessPrediction(metadata,
                                                access_history,
                                                forecast_horizont=13).predict()

        report = pandas.merge(proba_report, predict_report, how='inner', on='Name')

        report = pandas.merge(report, metadata[['Nb_Replicas', 'LFNSize', 'Name']], how='inner', on='Name')

        report = pandas.merge(report, features, how='inner', on='Name')

        return report








    def _full_save_report(self, combine_report):
        """
        Get full save report
        :param pandas.DataFrame combine_report: combination of the probability and the prediction reports
        :return: pandas.DataFrame: full save report
        """

        report = pandas.DataFrame()

        values = combine_report.values
        columns = combine_report.columns

        for row_num in range(0, combine_report.shape[0]):

            row_values = values[row_num, :]
            row = pandas.DataFrame(data=[row_values], columns=columns)

            while row.Nb_Replicas.values[0] >= self.min_replicas + 1:

                row['Metric'] = row.Prediction / row.Nb_Replicas
                row['DecreaseReplicas'] = 1
                report = pandas.concat([report, row])
                row.Nb_Replicas -= 1

        return report

    def save_n_tb(self, n_tb=None):
        """
        Get recommendations to save n Tb of disk space
        :param int n_tb: volume of disk space wanted to be saved
        :return: pandas.DataFrame: recommendations report
        """

        if self.flag1 == 0:
            combine_report = self.get_combine_report(self.data)
            full_decrease_report = self._full_save_report(combine_report)

            full_decrease_report = full_decrease_report.sort(['Metric', 'Probability'])
            self.full_save_report = full_decrease_report

            self.saved_space = self.full_save_report.LFNSize.values.cumsum()

        self.flag1 = 1

        if n_tb == None:
            report = self.full_save_report
        else:
            report = self.full_save_report[self.saved_space <= n_tb]

        report = report.sort(['Metric', 'Probability'])

        return report








    def _full_fill_report(self, combine_report):
        """
        Get full fill report
        :param pandas.DataFrame combine_report: combination of the probability and the prediction reports
        :return: pandas.DataFrame: full fill report
        """

        report = pandas.DataFrame()

        values = combine_report.values
        columns = combine_report.columns

        for row_num in range(0, combine_report.shape[0]):

            row_values = values[row_num, :]
            row = pandas.DataFrame(data=[row_values], columns=columns)

            while row.Nb_Replicas.values[0] <= self.max_replicas - 1:

                row['Metric'] = row.Prediction / row.Nb_Replicas
                row['IncreaseReplicas'] = 1
                report = pandas.concat([report, row])
                row.Nb_Replicas += 1

        return report

    def fill_n_tb(self, n_tb=None):
        """
        Get recommendations to fill n Tb of disk free space
        :param int n_tb: volume of disk space wanted to be filled
        :return: pandas.DataFrame: recommendations report
        """

        if self.flag2 == 0:
            combine_report = self.get_combine_report(self.data)
            full_fill_report = self._full_fill_report(combine_report)

            full_fill_report = full_fill_report.sort(['Metric', 'Probability'], ascending=False)
            self.full_fill_report = full_fill_report

            self.fill_space = self.full_fill_report.LFNSize.values.cumsum()

        self.flag2 = 1


        if n_tb == None:
            report = self.full_fill_report
        else:
            report = self.full_fill_report[self.fill_space <= n_tb]

        report = report.sort(['Metric', 'Probability'], ascending=False)

        return report






    def clean_n_tb(self, n_tb):
        """
        Get recommendations to remove n Tb from disk space
        :param int n_tb: volume of disk space wanted to be cleaned
        :return: pandas.DataFrame: recommendations report
        """

        if self.flag3 == 0:
            self.combine_report = self.get_combine_report(self.data)
            self.combine_report = self.combine_report.sort('Probability')

            self.clean_space = (self.combine_report.LFNSize * self.combine_report.Nb_Replicas).values.cumsum()

        self.flag3 = 1

        if n_tb == None:
            report = self.combine_report
        else:
            report = self.combine_report[self.clean_space <= n_tb]
            
        report = report.sort('Probability')

        return report


