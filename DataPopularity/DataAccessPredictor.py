from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

######################
###Version 0.2     ###
######################
###Using REP v0.5  ###
######################

import numpy as np
import pandas as pd
import re

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

try:
    from kernel_regression import KernelRegression
except ImportError as e:
    raise ImportError("Install kernel_regression from www or from 'Packages/' folder. ")

class DataAccessPredictor(object):
    """
    Data sets access predictor.
    KernelRegression is used.
    Nadaraya-Watson regression with leave-one-out window-control. Then, rolling mean with adaptive window.
    Parameters:
    -----------
    :param str source_path: a train data file path.
    The data file must have .xls, .xlsx or .csv formats.

    :param pandas.DataFrame data: train data.

    :param int nb_of_weeks: number of weeks in data sets access history.
    """

    def __init__(self, source_path=None, data=None, nb_of_weeks=104):
        if source_path != None:
            ext = source_path.split('.')[-1]
            if ext=='csv':
                try:
                    self.data = pd.read_csv(source_path)
                except:
                    print ("Can not open file.")
            elif ext=='xls' or ext=='xlsx':
                try:
                    self.data = pd.read_excel(source_path)
                except:
                    print ("Can not open file.")
        else:
            self.data = data

        self.periods = [str(i) for i in range(1,nb_of_weeks+1)]
        self._fix_columns()
        self._data_transform()

        self.predicted_curves = None
        self.predict_report = None
        self.df = None

    def _rename(self, x):
        """
        :param str or int x: column's name
        :return: str renamed column's name
        """
        return re.sub('\W', '_', str(x))

    def _fix_columns(self):
        """
        Rename data columns
        :return: list[str] renamed column names
        """
        self.data.columns = map(self._rename, self.data.columns)
        return self.data.columns

    def _data_transform(self):
        """
        Transform data
        :return: pandas.DataFrame transformed data
        """
        data1 = self.data.copy()
        data2 = self.data.copy()
        for i in range(0, len(self.periods)):
            if i!=0:
                data1[self.periods[i]] = self.data[self.periods[i]] - self.data[self.periods[i-1]]

        for i in range(0, len(self.periods)):
            k = len(self.periods)-1-i
            data2[self.periods[i]] = data1[self.periods[k]]

        for i in range(0, len(self.periods)):
            if i!=0:
                data2[self.periods[i]] = data2[self.periods[i]] + data2[self.periods[i-1]]
        self.data = data2
        return self.data

    def _check_columns(self):
        """
        Check whether all needed data columns are presence
        :return: 1 if all needed columns are presence in the train data. Otherwise, rise assertion.
        """
        cols_needed = pd.core.index.Index([u'Name', u'Configuration', u'ProcessingPass', u'FileType',
                                           u'Type', u'Creation_week', u'NbLFN', u'LFNSize', u'NbDisk', u'DiskSize', u'NbTape',
                                           u'TapeSize', u'NbArchived', u'ArchivedSize', u'Nb_Replicas', u'Nb_ArchReps',
                                           u'Storage', u'FirstUsage', u'LastUsage', u'Now'])
        cols = self.data.columns
        intersect = cols.intersection(cols_needed)
        diff_cols = cols_needed.diff(intersect)

        assert len(diff_cols)==0, str(["Please, add following columns to the data: ", list(diff_cols)])[1:-1]
        return 1

    def _features_intervals(self, data, periods):
        """
        Calculate new features
        :param pandas.DataFrame data: train data.
        :param list[str] periods: periods of data sets history access that are used to calculate new features
        :return: numpy.array new features
        """
        data_bv = data.copy()
        for i in range(0, len(periods)):
            if i!=0:
                data_bv[periods[i]] = data[periods[i]] - data[periods[i-1]]
        data_bv[periods] = (data_bv[periods] != 0)*1

        inter_max = []
        last_zeros = []
        nb_peaks = []
        inter_mean = []
        inter_std = []
        inter_rel = []

        for i in range(0,data_bv.shape[0]):
            ds = data_bv[periods].irow(i)
            nz = ds.nonzero()[0]
            inter = []

            nb_peaks.append(len(nz))
            if len(nz)==0:
                nz = [0]
            if len(nz)<2:
                inter = [0]
                #nz = [0]
            else:
                for k in range(0, len(nz)-1):
                    val = nz[k+1]-nz[k]
                    inter.append(val)

            inter = np.array(inter)
            inter_mean.append(inter.mean())
            inter_std.append(inter.std())
            if inter.mean()!=0:
                inter_rel.append(inter.std()/inter.mean())
            else:
                inter_rel.append(0)

            last_zeros.append(int(periods[-1]) - nz[-1] + 1)
            inter_max.append(max(inter))

        return np.array(inter_max), np.array(last_zeros), np.array(nb_peaks), np.array(inter_mean), np.array(inter_std), np.array(inter_rel)

    def _data_preparation(self):
        """
        Preparing data to make prediction
        :return:pandas.DataFrame for prediction
        """
        self._check_columns()
        periods = self.periods
        df = pd.DataFrame()
        df['Name'] = self.data['Name']
        inter_max, last_zeros, nb_peaks, inter_mean, inter_std, inter_rel = self._features_intervals(self.data, periods)
        df['last_zeros'] = last_zeros
        df['inter_max'] = inter_max
        df['nb_peaks'] = nb_peaks
        df['inter_mean'] = inter_mean
        df['inter_std'] = inter_std
        df['inter_rel'] = inter_rel

        for i in range(0, len(periods)):
            if i!=0:
                df[periods[i]] = self.data[periods[i]] - self.data[periods[i-1]]
            else :
                df[periods[i]] = self.data[periods[i]]
        self.df = df
        return df

    def _kernel_regression_and_rolling_mean(self, X, y, window):
        """
        Implementation of Nadaraya-Watson regression and rolling_mean to the time series.
        :param X: array-like of shape = [n_samples, n_features]
        The training input samples.
        :param y: array-like, shape = [n_samples]
        The target values
        :param window: int
        Size of the moving window. This is the number of observations used for
        calculating the statistic.
        :return: tuple of 3 array-like, shape = [n_samples]
        Predicted target values, rolling_mean target values and rolling_std of the target values
        """
        kr = KernelRegression(kernel="rbf", gamma=np.logspace(-2, 2, 10))
        y_kr = kr.fit(X, y).predict(X)
        y_rm = pd.rolling_mean(y_kr, window=window,axis=1)
        y_std = pd.rolling_std(y, window=window,axis=1)
        return y_kr, y_rm, y_std

    def _prediction_for_time_series(self, rows_range, df, zero_one_scale=False):
        """
        Method for prediction
        :param list[int] rows_range: list of data rows for prediction
        :param pandas.DataFrame df: data for prediction
        :param boolean zero_one_scale: if True transform the time serie values to [0,1] range
        :return list[dict]: list of the prediction curves
        """
        predicted_curves = []
        for row in rows_range:
            ts_train = df.irow([row])
            name = ts_train['Name'].values[0]
            X = np.array([[i] for i in range(0, len(self.periods))])
            y = ts_train[self.periods].values[0]
            if zero_one_scale==True:
                max_value = y.max()+1
                y = y/(1.0*max_value)

            p = df['nb_peaks'].irow([row]).values[0]
            w = df['inter_max'][df['nb_peaks']==p]
            window = int(w.quantile(0.90))+1
            y_kr, y_rm, y_std = self._kernel_regression_and_rolling_mean(X, y, window)

            res = {'Name':name, 'y':y,
                   'y_out':y_rm, 'y_kr':y_kr,
                   'y_std':y_std}
            predicted_curves.append(res)
            self.predicted_curves = predicted_curves
        return predicted_curves

    def predict(self, zero_one_scale=False):
        """
        Predict data accesses by the last point of the smoothing regression curve.
        :param boolean zero_one_scale: if True transform the time serie values to [0,1] range
        :return: pandas.DataFrame of the predicted data accesses.
        """
        prepared_data = self._data_preparation()
        rows_range = range(prepared_data.shape[0])
        predicted_curves = self._prediction_for_time_series(rows_range, prepared_data, zero_one_scale)

        names = []
        accesses = []
        std_errors = []
        for curve in predicted_curves:
            names.append(curve['Name'])
            accesses.append(curve['y_out'][-1])
            std_errors.append(curve['y_std'][-1])

        predict_report = pd.DataFrame()
        predict_report['Name'] = names
        predict_report['Access'] = accesses
        predict_report['Std_error'] = std_errors
        self.predict_report = predict_report
        return predict_report

    def show_examples(self, start_row=0, end_row=50):
        """
        Show images of the original, regression and predicted curves.
        :param int start_row: the first row number for plotting.
        :param end_row: the last row number for plotting.
        :return: 1 and curves plots.
        """
        assert self.predict_report is not None, "Use 'predict()' method first."
        N = end_row - start_row
        plt.figure(figsize=(15, 5*(N//3+1)))
        for row in range(start_row,end_row):
            plt.subplot(N//3+1,3,row-start_row+1)
            res = self.predicted_curves[row]
            plt.plot(res['y_kr'], color='b', label='Nadaraya-Watson')
            plt.plot(res['y'], color='g', label='original')
            plt.plot(res['y_out'], color='r', label='rolling_mean')
            plt.fill_between(x=range(1,len(self.periods)+1) ,y1=res['y_out']-res['y_std'], y2=res['y_out']+res['y_std'], alpha=0.1, color='r', label='std_error')
            y1 = [res['y_out'][-1]-res['y_std'][-1]]*20
            y2 = [res['y_out'][-1]+res['y_std'][-1]]*20
            x=range(len(self.periods)+1,len(self.periods)+21)
            plt.fill_between(x=x ,y1=y1, y2=y2, alpha=0.1, color='r', label='std_error')
            plt.plot(x, [res['y_out'][-1]]*20, color='r')
            plt.title('Row number is '+str(row))
            plt.legend(loc='best')
        return 1

