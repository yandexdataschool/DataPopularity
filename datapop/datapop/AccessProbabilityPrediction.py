from __future__ import division, print_function, absolute_import

__author__ = 'mikhail91'

# datapop 3.0.0

import numpy
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score

class AccessProbabilityPrediction (object):
    """
    This class is for prediction of probability that a dataset will be requested in near future.
    :param pandas.DataFrame metadata: metadata of LHCb datasets.
    :param pandas.DataFrame access_history: access history of LHCb datasets.
    :param int forecas_horizont: forecast horizont for the prediction.
    """
    def __init__(self, metadata=None, access_history=None, forecast_horizont=None):
        self.metadata = metadata
        self.access_history = access_history
        self.forecast_horizont = forecast_horizont

    def recency(self, access_time_series):
        """
        This method computes recency of LHCb datasets.
        Recency - is when a dataset was accesses last time.
        :param numpy.ndarray access_time_series: time series of the datasets access history.
        :return: numpy.array: the datasets recency.
        """
        dataset_recencies = []
        for dataset in access_time_series:
            recency = 0
            index = dataset.shape[0] - 1
            while dataset[index] == 0 and index >= 0:
                recency += 1
                index -= 1
            dataset_recencies.append(recency)
        return numpy.array(dataset_recencies)

    def reuse_distance(self, access_time_series):
        """
        This method calculates the datasets reuse distances.
        Reuse distance - distance between a dataset last access and the second last access.
        :param numpy.ndarray access_time_series: time series of the datasets access history.
        :return: numpy.array: the datasets reuse distances.
        """
        dataset_reuse_distances = []
        for dataset in access_time_series:
            reuse_distance = 0
            index = dataset.shape[0] - 1
            last_request = index
            second_last_request = -1
            while dataset[index] == 0 and index >= 0:
                index -= 1
                last_request = index
            if last_request == -1:
                reuse_distance = 999
                dataset_reuse_distances.append(reuse_distance)
                continue
            index = index - 1
            second_last_request = index
            while dataset[index] == 0 and index >=0:
                index -= 1
                second_last_request = index
            if second_last_request == -1:
                reuse_distance = 999
                dataset_reuse_distances.append(reuse_distance)
                continue
            reuse_distance = last_request - second_last_request
            dataset_reuse_distances.append(reuse_distance)
        return numpy.array(dataset_reuse_distances)

    def frequency(self, access_time_series):
        """
        This method computes LHCb datasets frequencies in number of accesses.
        :param numpy.ndarray access_time_series: time series of the datasets access history.
        :return: numpy.array: the datasets frequencies.
        """
        window_to_past = 52
        return access_time_series[:, -window_to_past:].sum(axis=1)

    def frequency_week(self, access_time_series):
        """
        This method computes LHCb datasets frequencies in number of weeks.
        :param numpy.ndarray access_time_series: time series of the datasets access history.
        :return: numpy.array: the datasets frequencies.
        """
        window_to_past = 52
        return ((access_time_series > 0) * 1)[:, -window_to_past:].sum(axis=1)

    def file_extentions_dic(self, metadata):
        """
        This method constructs the datasets extentions dictionary.
        :param pandas.DataFrame metadata: metadata of LHCb datasets.
        :return: dict file_extentions_dict: dictionary of the datasets extentions.
        """
        file_extentions = \
            [name.split('.')[-1] for name in metadata['FileType'].values]
        unique_file_extentions = numpy.unique(numpy.array(file_extentions))
        file_extentions_dict = dict((unique_file_extentions[i], i) \
                                    for i in range(0, unique_file_extentions.shape[0]))
        return file_extentions_dict

    def extentions(self, metadata, file_extentions_dict):
        """
        This method links the datasets extentions with integers.
        :param pandas.DataFrame metadata: metadata of LHCb datasets.
        :param dict file_extentions_dict: dictionary of the datasets extentions.
        :return: numpy.array file_extention_code_list: the datasets coded extentions.
        """
        file_extention_code_list = []
        for name in metadata['FileType'].values:
            file_extention = name.split('.')[-1]
            file_extention_code = file_extentions_dict[file_extention]
            file_extention_code_list.append(file_extention_code)
        return numpy.array(file_extention_code_list)

    def data_preprocessing(self, metadata, access_history, forecast_horizont):
        """
        Data preprocessing for classification.
        :param pandas.DataFrame metadata: metadata of LHCb datasets.
        :param pandas.DataFrame access_history: access history of LHCb datasets.
        :param pandas.DataFrame train_data, test_data, to_predict_data:
        data for train, test and for the probability prediction.
        """
        file_extentions_dict = self.file_extentions_dic(metadata)

        # train data
        train_selection = (metadata['Now'].values - \
                          metadata['FirstUsage'].values > \
                          2*forecast_horizont) * (metadata['Storage'] == 'Disk')
        train_access_history = access_history[train_selection]
        train_metadata = metadata[train_selection]

        train_access_time_series = \
            train_access_history.drop('Name', 1).values[:, :-2*forecast_horizont]
        train_recency = self.recency(train_access_time_series)
        train_reuse_distance = self.reuse_distance(train_access_time_series)
        train_frequency = self.frequency(train_access_time_series)
        train_frequency_week = self.frequency_week(train_access_time_series)
        train_size = train_metadata['LFNSize'].values
        train_nblfn = train_metadata['NbLFN'].values
        train_first_used = train_metadata['Now'].values - \
                           train_metadata['FirstUsage'].values - \
                           2 * forecast_horizont
        train_creation = train_metadata['Now'].values - \
                           train_metadata['Creation_week'].values - \
                           2 * forecast_horizont
        train_type = train_metadata['Type'].values
        train_extentions = self.extentions(train_metadata, file_extentions_dict)

        train_data = pandas.DataFrame()
        train_data['Name'] = train_metadata['Name'].values
        train_data['recency'] = train_recency
        train_data['reuse_distance'] = train_reuse_distance
        train_data['first_used'] = train_first_used
        train_data['creation'] = train_creation
        train_data['frequency'] = train_frequency
        train_data['frequency_week'] = train_frequency_week
        train_data['type'] = train_type
        train_data['extentions'] = train_extentions
        train_data['size'] = train_size
        train_data['nblfn'] = train_nblfn
        train_data['Y'] = 1* (train_access_history.drop('Name', 1).values[:,
                      -2*forecast_horizont:-forecast_horizont].sum(axis=1) > 0)

        # test data
        test_selection = (metadata['Now'].values - \
                         metadata['FirstUsage'].values > \
                         1*forecast_horizont) * (metadata['Storage'] == 'Disk')

        test_access_history = access_history[test_selection]
        test_metadata = metadata[test_selection]

        test_access_time_series = \
            test_access_history.drop('Name', 1).values[:, :-1*forecast_horizont]
        test_recency = self.recency(test_access_time_series)
        test_reuse_distance = self.reuse_distance(test_access_time_series)
        test_frequency = self.frequency(test_access_time_series)
        test_frequency_week = self.frequency_week(test_access_time_series)
        test_size = test_metadata['LFNSize'].values
        test_nblfn = test_metadata['NbLFN'].values
        test_first_used = test_metadata['Now'].values - \
                          test_metadata['FirstUsage'].values - \
                          forecast_horizont
        test_creation = test_metadata['Now'].values - \
                          test_metadata['Creation_week'].values - \
                          forecast_horizont
        test_type = test_metadata['Type'].values
        test_extentions = self.extentions(test_metadata, file_extentions_dict)

        test_data = pandas.DataFrame()
        test_data['Name'] = test_metadata['Name'].values
        test_data['recency'] = test_recency
        test_data['reuse_distance'] = test_reuse_distance
        test_data['first_used'] = test_first_used
        test_data['creation'] = test_creation
        test_data['frequency'] = test_frequency
        test_data['frequency_week'] = test_frequency_week
        test_data['type'] = test_type
        test_data['extentions'] = test_extentions
        test_data['size'] = test_size
        test_data['nblfn'] = test_nblfn
        test_data['Y'] = 1* (test_access_history.drop('Name', 1).values[:,
                     -forecast_horizont:].sum(axis=1) > 0)

        # to predict data
        to_predict_selection = metadata['Storage'] == 'Disk'
        to_predict_access_history = access_history[to_predict_selection]
        to_predict_metadata = metadata[to_predict_selection]

        to_predict_access_time_series = \
            to_predict_access_history.drop('Name', 1).values
        to_predict_recency = self.recency(to_predict_access_time_series)
        to_predict_reuse_distance = self.reuse_distance(to_predict_access_time_series)
        to_predict_frequency = self.frequency(to_predict_access_time_series)
        to_predict_frequency_week = self.frequency_week(to_predict_access_time_series)
        to_predict_size = to_predict_metadata['LFNSize'].values
        to_predict_nblfn = to_predict_metadata['NbLFN'].values
        to_predict_first_used = to_predict_metadata['Now'].values - \
                                to_predict_metadata['FirstUsage'].values
        to_predict_creation = to_predict_metadata['Now'].values - \
                                to_predict_metadata['Creation_week'].values
        to_predict_type = to_predict_metadata['Type'].values
        to_predict_extentions = self.extentions(to_predict_metadata, file_extentions_dict)

        to_predict_data = pandas.DataFrame()
        to_predict_data['Name'] = to_predict_metadata['Name'].values
        to_predict_data['recency'] = to_predict_recency
        to_predict_data['reuse_distance'] = to_predict_reuse_distance
        to_predict_data['first_used'] = to_predict_first_used
        to_predict_data['creation'] = to_predict_creation
        to_predict_data['frequency'] = to_predict_frequency
        to_predict_data['frequency_week'] = to_predict_frequency_week
        to_predict_data['type'] = to_predict_type
        to_predict_data['extentions'] = to_predict_extentions
        to_predict_data['size'] = to_predict_size
        to_predict_data['nblfn'] = to_predict_nblfn
        return train_data, test_data, to_predict_data

    def predict(self):
        """
        Prediction of the probability that a dataset will be requested in near future.
        :return: pandas.DataFrame report: report with the probabilities and
        the classification performance.
        """
        # TODO: classifier optimization
        train_data, test_data, to_predict_data = \
            self.data_preprocessing(self.metadata,
                                    self.access_history,
                                    self.forecast_horizont)
        X_train = train_data.drop(['Name', 'Y'], 1).values
        Y_train = train_data['Y'].values
        X_test = test_data.drop(['Name', 'Y'], 1).values
        Y_test = test_data['Y'].values
        X_to_predict = to_predict_data.drop('Name', 1).values

        rfc = RandomForestClassifier(n_estimators=200,
                                     max_features='auto',
                                     max_depth=5,
                                     class_weight='balanced')

        # train - test
        rfc.fit(X_train, Y_train)
        Y_proba_test = rfc.predict_proba(X_test)[:, 1]
        Y_pred_test = rfc.predict(X_test)
        roc_auc_test = roc_auc_score(Y_test, Y_proba_test)
        precision0_test = precision_score(1 - Y_test, 1 - Y_pred_test)

        # test - to predict
        rfc.fit(X_test, Y_test)
        Y_proba_to_predict = rfc.predict_proba(X_to_predict)[:, 1]

        #report
        report = pandas.DataFrame()
        report['Name'] = to_predict_data['Name'].values
        report['Probability'] = Y_proba_to_predict
        report['roc_auc'] = roc_auc_test
        report['precision0'] = precision0_test
        return report




