from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

# version 3.0

import numpy
import pandas

def tsa_brown(Y, alpha):
    """
    The Brown model for time series forecasting.
    :param numpy.array Y: values for a time series
    :param float alpha: value of the parameter of the model in [0, 1]
    :return: numpy.array the time series forecast,
    numpy.array the forecast error.
    """
    Y_predict = [Y[0]]
    predict_errors = []
    for y_curr in Y:
        y_pred_prev = Y_predict[-1]
        error = y_pred_prev - y_curr
        y_pred = alpha * y_curr + (1. - alpha) * y_pred_prev
        Y_predict.append(y_pred)
        predict_errors.append(error)
    return numpy.array(Y_predict), numpy.array(predict_errors)

def tsa_brown_grid(Y, alphas):
    """
    Grid search for The Brown model.
    :param numpy.array Y: values for a time series
    :param numpy.array alphas: values of the parameter of the model in [0, 1]
    :return: numpy.array the time series forecast,
    numpy.array the forecast error.
    """
    mse_all = []
    alphas = numpy.array(alphas)
    for alpha in alphas:
        Y_pred, errors = tsa_brown(Y, alpha)
        mse = (errors**2).mean()
        mse_all.append(mse)
    mse_all = numpy.array(mse_all)
    alpha_opti = alphas[mse_all == mse_all.min()][0]
    Y_pred, errors = tsa_brown(Y, alpha_opti)
    return Y_pred, errors

def bins(Y, size=5):
    """
    Group a time series values into bins.
    :param numpy.array Y: values for a time series
    :param int size: the bins size
    :return: numpy.array new time series
    """
    Y_bins = []
    index = len(Y)
    while index >= size:
        Y_bins.append(Y[index-size:index].mean())
        index -= size
    Y_bins.reverse()
    return numpy.array(Y_bins)

# TODO: Better predictor

class NumberAccessPrediction(object):

    def __init__(self, metadata=None, access_history=None, forecast_horizont=None):
        self.metadata = metadata
        self.access_history = access_history
        self.forecast_horizont = forecast_horizont
        self.ts_predictions = None

    def data_preprocessing(self, metadata, access_history, forecast_horizont):
        """
        Data preprocessing for classification.
        :param pandas.DataFrame metadata: metadata of LHCb datasets.
        :param pandas.DataFrame access_history: access history of LHCb datasets.
        :param pandas.DataFrame train_data, test_data, to_predict_data:
        data for the probability prediction.
        """
        # TODO: Why FirstUsage but not Creation_week?
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
        return to_predict_data

    def _brown_predictior(self, access_time_series, first_used_list, forecast_horizont):
        """
        Use Brown model for the time series forecast.
        :param numpy.ndarray access_time_series: time series of the access history
        :param numpy.array first_used_list: array with times of the dataset first usages
        :param int forecast_horizont: forecast horizont
        :return: numpy.array predictions,
        numpy.array root mean square errors
        """
        all_predictions = []
        all_rmse = []
        self.ts_predictions = []
        for dataset, first_used in zip(access_time_series, first_used_list):

            Y = dataset[-first_used:]
            Y_bins = bins(Y, forecast_horizont)

            alphas = [0.1*i for i in range(0, 11)]

            Y_pred, errors = tsa_brown_grid(Y_bins, alphas)
            self.ts_predictions.append([Y_bins, Y_pred, errors])

            rmse = numpy.sqrt((errors**2).mean())
            prediction = Y_pred[-1]

            all_predictions.append(prediction)
            all_rmse.append(rmse)

        return numpy.array(all_predictions), numpy.array(all_rmse)


    def predict(self):
        """
        Get prediction report
        :return: pandas.DataFrame prediction report
        """
        to_predict_data = \
            self.data_preprocessing(self.metadata, self.access_history, self.forecast_horizont)

        # to predict
        to_predict_access_time_series = \
            to_predict_data.drop(['Name', 'first_used'], 1).values
        to_predict_first_used_list = \
            to_predict_data['first_used'].values
        prediction, rmse = self._brown_predictior(to_predict_access_time_series,
                                                   to_predict_first_used_list,
                                                   self.forecast_horizont)

        # report
        report = pandas.DataFrame()
        report['Name'] = to_predict_data['Name']
        report['Prediction'] = prediction
        report['rmse'] = rmse
        return report
