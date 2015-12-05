from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

# version 3.0

import numpy
import pandas

try:
    from kernel_regression import KernelRegression
except ImportError as e:
    raise ImportError("Install kernel_regression from www or from 'Packages/' folder. ")

# TODO: Better predictor

class ReplicationPlacementStrategy(object):

    def __init__(self, metadata=None, access_history=None, forecast_horizont=None):
        self.metadata = metadata
        self.access_history = access_history
        self.forecast_horizont = forecast_horizont

    def data_preprocessing(self, metadata, access_history, forecast_horizont):
        """
        Data preprocessing for classification.
        :param pandas.DataFrame metadata: metadata of LHCb datasets.
        :param pandas.DataFrame access_history: access history of LHCb datasets.
        :param pandas.DataFrame train_data, test_data, to_predict_data:
        data for train, test and for the probability prediction.
        """
        # TODO: Why FirstUsage but not Creation_week?
        # train data
        train_selection = (metadata['Now'].values - \
                          metadata['FirstUsage'].values > \
                          2*forecast_horizont + 26) * (metadata['Storage'] == 'Disk')
        train_access_history = access_history[train_selection]
        train_metadata = metadata[train_selection]

        train_first_used = train_metadata['Now'].values - \
                           train_metadata['FirstUsage'].values - \
                           2 * forecast_horizont

        train_time_columns = \
            train_access_history.drop('Name', 1).columns[:-2 * forecast_horizont]
        train_data = train_access_history[['Name'] + list(train_time_columns)].copy()
        train_data['first_used'] = train_first_used
        train_data['Y'] = train_access_history.drop('Name', 1).values[:,
                      -2*forecast_horizont:-forecast_horizont].mean(axis=1)

        # test data
        test_selection = (metadata['Now'].values - \
                         metadata['FirstUsage'].values > \
                         1*forecast_horizont + 26) * (metadata['Storage'] == 'Disk')

        test_access_history = access_history[test_selection]
        test_metadata = metadata[test_selection]

        test_first_used = test_metadata['Now'].values - \
                           test_metadata['FirstUsage'].values - \
                           1 * forecast_horizont

        test_time_columns = \
            test_access_history.drop('Name', 1).columns[:-1 * forecast_horizont]
        test_data = test_access_history[['Name'] + list(test_time_columns)].copy()
        test_data['first_used'] = test_first_used
        test_data['Y'] = test_access_history.drop('Name', 1).values[:,
                     -forecast_horizont:].mean(axis=1)

        # to predict data
        to_predict_selection = (metadata['Now'].values - \
                         metadata['FirstUsage'].values > 26) * \
                               metadata['Storage'] == 'Disk'
        to_predict_access_history = access_history[to_predict_selection]
        to_predict_metadata = metadata[to_predict_selection]

        to_predict_first_used = to_predict_metadata['Now'].values - \
                           to_predict_metadata['FirstUsage'].values

        to_predict_data = to_predict_access_history.copy()
        to_predict_data['first_used'] = to_predict_first_used
        return train_data, test_data, to_predict_data

    def _static_predictior(self, access_time_series, first_used_list, window_to_past):

        static_predictions_list = []
        for dataset, first_used in zip(access_time_series, first_used_list):
            index = min(window_to_past, first_used)
            static_predicition = dataset[-index:].mean()
            static_predictions_list.append(static_predicition)
        return numpy.array(static_predictions_list)


    def static_predict(self, window_to_past):

        train_data, test_data, to_predict_data = \
            self.data_preprocessing(self.metadata, self.access_history, self.forecast_horizont)

        # test
        test_access_time_series = test_data.drop(['Name', 'Y', 'first_used'], 1).values
        test_first_used_list = test_data['first_used'].values
        Y_test = test_data['Y'].values
        Y_pred_test = self._static_predictior(test_access_time_series,
                                              test_first_used_list,
                                              window_to_past)
        # TODO: something wrong with errors.
        rel_error_test = 100. * (Y_pred_test - Y_test)/(1. + Y_test)
        mpe = rel_error_test.mean()

        # to predict
        to_predict_access_time_series = \
            to_predict_data.drop(['Name', 'first_used'], 1).values
        to_predict_first_used_list = \
            to_predict_data['first_used'].values
        Y_pred_to_predict = self._static_predictior(to_predict_access_time_series,
                                                    to_predict_first_used_list,
                                                    window_to_past)

        # report
        report = pandas.DataFrame()
        report['Name'] = to_predict_data['Name']
        report['Prediction'] = Y_pred_to_predict
        report['mpe'] = mpe
        return report, Y_pred_test, Y_test

    def _max_reuse_distance(self, access_time_series):
        max_reuse_distances_list = []
        for dataset in access_time_series:
            max_reuse_distance = 1
            reuse_distance = 0
            start_flag = 0
            for access in dataset:
                if access > 0:
                    start_flag = 1
                if start_flag == 1:
                    reuse_distance += 1
                if start_flag == 1 and access > 0:
                    max_reuse_distance = max(max_reuse_distance, reuse_distance)
                    reuse_distance = 1
            max_reuse_distances_list.append(max_reuse_distance)
        return numpy.array(max_reuse_distances_list)

    def _total_fraquency_week(self, access_time_series):
        return (1. * (access_time_series > 0)).sum(axis=1) + 1

    def _kernel_smoothing_predictor(self, access_time_series, first_used_list, window_to_past):

        kernel_predictions_list = []
        max_reuse_distance_list = self._max_reuse_distance(access_time_series[:, -window_to_past:])
        total_frequency_week_list = self._total_fraquency_week(access_time_series[:, -window_to_past:])
        lists = zip(access_time_series, first_used_list, max_reuse_distance_list)
        for dataset, first_used, max_reuse_distance in lists:
            # index = min(window_to_past, first_used)
            index = window_to_past
            x_coords = numpy.array([[x_val] for x_val in range(0, index)])
            y_values = dataset[-index:]
            #selection = total_frequency_week_list == total_frequency_week
            #windows = max_reuse_distance_list[selection]
            #window = int(numpy.percentile(windows, 80)) + 1
            window = min(max_reuse_distance, index)
            kernel_regressor = KernelRegression(kernel="rbf", gamma=numpy.logspace(-2, 2, 10))
            y_kernel = kernel_regressor.fit(x_coords, y_values).predict(x_coords)
            #y_kernel = y_values
            y_rolling_mean = pandas.rolling_mean(y_kernel, window=window, axis=1)
            kernel_prediction = y_rolling_mean[-1]
            kernel_predictions_list.append(kernel_prediction)
        return numpy.array(kernel_predictions_list)

    def kernel_smoothing_predict(self, window_to_past):

        train_data, test_data, to_predict_data = \
            self.data_preprocessing(self.metadata, self.access_history, self.forecast_horizont)

        # test
        test_access_time_series = test_data.drop(['Name', 'Y', 'first_used'], 1).values
        test_first_used_list = test_data['first_used'].values
        Y_test = test_data['Y'].values
        Y_pred_test = self._kernel_smoothing_predictor(test_access_time_series,
                                                       test_first_used_list,
                                                       window_to_past)
        # TODO: something wrong with errors.
        rel_error_test = 100. * (Y_pred_test - Y_test)/(1. + Y_test)
        mpe = rel_error_test.mean()

        # to predict
        to_predict_access_time_series = \
            to_predict_data.drop(['Name', 'first_used'], 1).values
        to_predict_first_used_list = \
            to_predict_data['first_used'].values
        Y_pred_to_predict = self._kernel_smoothing_predictor(to_predict_access_time_series,
                                                             to_predict_first_used_list,
                                                             window_to_past)

        # report
        report = pandas.DataFrame()
        report['Name'] = to_predict_data['Name']
        report['Prediction'] = Y_pred_to_predict
        report['mpe'] = mpe
        return report, Y_pred_test, Y_test
