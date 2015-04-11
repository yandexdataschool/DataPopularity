__author__ = 'mikhail91'

import numpy as np
import pandas as pd
import re
from StringIO import StringIO
from requests import get, post, put

class DataPopularityApiWrapper(object):
    def __init__(self, service_url='http://localhost:5000'):
        self.service_url = service_url
        message = post(service_url).json()
        self.session_id = re.compile('\d+').findall(message)[0]

    def upload(self, data_path):
        method_url = self.service_url + '/' + self.session_id + '/Upload'
        post(method_url, files={'file':open(data_path)})
        return 1

    def run_algorithm(self, nb_of_weeks=104):
        params={'nb_of_weeks':nb_of_weeks, 'q':None, 'set_replicas':'auto', 'c_disk':100, 'c_tape':1,
                'c_miss':2000, 'alpha':1, 'max_replicas':4, 'method':'opti', 'pop_cut':-1}
        method_url = self.service_url + '/' + self.session_id + '/DataPopularityApi'
        post(method_url, data={'params':str(params)})

    def get_data_popularity(self):
        method_url = self.service_url + '/' + self.session_id + '/Download/popularity.csv'
        output = get(method_url)
        return pd.read_csv(StringIO(output.content))

    def get_data_intensity_prediction(self):
        method_url = self.service_url + '/' + self.session_id + '/Download/prediction.csv'
        output = get(method_url)
        return pd.read_csv(StringIO(output.content))



    def get_opti_report(self, q=None, set_replicas='auto', c_disk=100, c_tape=1, c_miss=2000,\
                alpha=1, max_replicas=4):
        params={'q':q, 'set_replicas':set_replicas, 'c_disk':c_disk, 'c_tape':c_tape,\
                'c_miss':c_miss, 'alpha':alpha, 'max_replicas':max_replicas, 'method':'opti'}
        method_url1 = self.service_url + '/' + self.session_id + '/DataPopularityApi'
        put(method_url1, data={'params':str(params)})
        method_url2 = self.service_url + '/' + self.session_id + '/Download/opti_report.csv'
        output = get(method_url2)
        return pd.read_csv(StringIO(output.content))

    def get_report(self, q=None, set_replicas='auto', c_disk=100, c_tape=1, c_miss=2000,\
                alpha=1, max_replicas=4, pop_cut=0.5):
        params={'q':q, 'set_replicas':set_replicas, 'c_disk':c_disk, 'c_tape':c_tape,\
                'c_miss':c_miss, 'alpha':alpha, 'max_replicas':max_replicas, 'method':'report', 'pop_cut':pop_cut}
        method_url1 = self.service_url + '/' + self.session_id + '/DataPopularityApi'
        put(method_url1, data={'params':str(params)})
        method_url2 = self.service_url + '/' + self.session_id + '/Download/report.csv'
        output = get(method_url2)
        return pd.read_csv(StringIO(output.content))

