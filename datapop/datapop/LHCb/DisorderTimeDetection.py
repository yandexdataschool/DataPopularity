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

class DisorderTimeDetection(object):

    def __init__(self, data=None):
        self.data = data

    def _cumsum_stat(self, array):
        new_array = [0]
        cumsum = 0
        for i in range(1, len(array)):
            new_array.append(np.max([0, new_array[i-1]+array[i]]))
        return np.array(new_array)

    def _one_detection(self, x):

        if x.nonzero()[0].shape[0]==0:
            return 0, 999

        start_ind = np.max([x.nonzero()[0][0] - 1, 0])
        finish_ind = np.min([(x).nonzero()[0][-1] + 2, len(x)])
        x_train = x[start_ind:finish_ind]

        if np.unique(x_train).shape[0] > 2:
            n_states = 3
            unique = np.unique(x_train)
            unique.sort()
            states_train = 0 + (x_train > 0)*1 + (x_train > unique[unique.shape[0]//2])*1
            states = 0 + (x > 0)*1 + (x > unique[unique.shape[0]//2])*1
            zeros = np.array([0]*1000+[1]+[0]*1000+[2]+[0]*1000)
        else:
            n_states = 2
            states_train = 0 + (x_train > 0)*1
            states = 0 + (x > 0)*1
            zeros = np.array([0]*1000+[1]+[0]*1000)

        model = hmm.MultinomialHMM(n_states, n_iter=10)
        model.fit([states_train])

        model_zero = hmm.MultinomialHMM(n_states, n_iter=10)
        model_zero.fit([zeros])

        likelihood_diffs = []
        for i in range(2, x.shape[0]-start_ind+1):
            likelihood1 = model.score(states[start_ind:start_ind+i]) - model.score(states[start_ind:start_ind+i-1])
            likelihood2 = model_zero.score(states[start_ind:start_ind+i]) - model_zero.score(states[start_ind:start_ind+i-1])
            likelihood_diff = likelihood2 - likelihood1
            likelihood_diffs.append(likelihood_diff)
        likelihood_diffs = np.array(likelihood_diffs)

        cumsum = self._cumsum_stat(likelihood_diffs)
        #self.cumsum = cumsum


        threshold = cumsum[:finish_ind-start_ind-2].mean() + 5*cumsum[:finish_ind-start_ind-2].std()

        if threshold==0:
            ratio = 999
        else:
            ratio = cumsum[-1]/float(threshold)

        if cumsum[-1]>threshold:
            return 0, ratio
        else:
            return 1, ratio

    def get_report(self):

        columns = self.data.columns
        number_col_filter = re.compile("^\d+$")
        number_columns = [col for col in columns if number_col_filter.search(col)]

        recommendation_type = []
        ratios = []
        for i in range(0, self.data.shape[0]):
            x = self.data[number_columns].irow([i]).values[0]
            type, ratio = self._one_detection(x)
            recommendation_type.append(type)
            ratios.append(ratio)

        report = pd.DataFrame()
        report['ID'] = self.data['ID'].values
        report['Recommended_Type'] = recommendation_type
        report['Ratio'] = ratios
        return report





