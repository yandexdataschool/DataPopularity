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
from hmmlearn import hmm

class HMMPredict(object):

    def __init__(self, data=None, forecast_horizont=None, last_periods=None, class_abs_thresholds=None):
        self.data = data
        self.forecast_horizont = forecast_horizont
        self.last_periods = last_periods
        self.class_abs_thresholds = np.array(class_abs_thresholds, ndmin=1)

    def get_sequences(self, data=None, forecast_horizont=None, last_periods=None):
        columns = data.columns
        number_col_filter = re.compile("^\d+$")
        number_columns = [col for col in columns if number_col_filter.search(col)]

        ids = []
        train_sequences = []
        test_sequences = []
        for i in range(0, data.shape[0]):
            dataset = data.irow([i])
            ids.append(dataset['ID'].values[0])

            creation = dataset['Creation'].values[0]
            first = dataset['First'].values[0]
            start = dataset['StartTime'].values[0]

            if first - start > 0:
                start_index = first-start-1
            elif first>0:
                start_index=0
            elif creation - start >0:
                start_index = creation-start-1
            else:
                start_index=0

            if last_periods==None:
                x_train = np.array(dataset[number_columns[start_index:]].values, ndmin=2).T
            elif start_index <= len(number_columns)-last_periods:
                x_train = np.array(dataset[number_columns[-last_periods:]].values, ndmin=2).T
            else:
                x_train = np.array(dataset[number_columns[start_index:]].values, ndmin=2).T

            train_sequences.append(x_train)

            x_tets = np.zeros(shape=(forecast_horizont,1))
            test_sequences.append(x_tets)

        ids = np.array(ids)
        train_sequences = np.array(train_sequences)
        test_sequences = np.array(test_sequences)
        return (ids, train_sequences, test_sequences)

    def _one_predict(self, x_train=None, forecast_horizont=None, q=None):

        model = hmm.GaussianHMM(3, 'full')
        model.fit([x_train])

        last_state_probas = model.predict_proba(x_train)[-1,:]
        model.startprob_ = last_state_probas

        predicted_means = []
        for i in range(100):
            predicted_mean = model.sample(forecast_horizont)[0][:,0].mean()
            predicted_means.append(predicted_mean)
        predicted_means = np.array(predicted_means)

        low_percentile = np.percentile(predicted_means, (100.-q)/2.)
        hight_percentile = np.percentile(predicted_means, (100.+q)/2.)
        mean = predicted_means.mean()
        return low_percentile, mean, hight_percentile


    def get_forecast_report(self):
        (ids, train_sequences, test_sequences) = self.get_sequences(self.data.copy(), self.forecast_horizont, self.last_periods)

        low_percentiles = []
        hight_percentiles = []
        means = []

        for i in range(0, len(ids)):
            x_train = train_sequences[i]

            low_percentile, mean, hight_percentile = self._one_predict(x_train, self.forecast_horizont, 95.)

            low_percentiles.append(low_percentile)
            means.append(mean)
            hight_percentiles.append(hight_percentile)

        fh_data_values = np.array(hight_percentiles)
        type_label = 0
        for type_num in range(0, len(self.class_abs_thresholds)):
            type_label = type_label*(fh_data_values < self.class_abs_thresholds[type_num]) +\
                         (fh_data_values >= self.class_abs_thresholds[type_num])*(type_num+1)

        report = pd.DataFrame()
        report['ID'] = ids
        report['LowPercentile95'] = low_percentiles
        report['Mean'] = means
        report['HightPercentile95'] = hight_percentiles
        report['Recommended_Type'] = type_label
        return report

