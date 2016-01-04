__author__ = 'mikhail91'

# datapopserv 3.0.0

from flask import Flask, request, send_from_directory
from flask.ext.restful import reqparse, abort, Api, Resource

from datapop import ReplicationPlacementStrategy
import numpy as np
import pandas as pd
import os
import ast

ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
api = Api(app)

working_dir = os.environ.get('WORKING_DIR')
data_popularity_data = working_dir + '/'



#DataPopularity
class DataPopularityApi(Resource):
    """
    This class is for datapop library
    """

    def get_session_id(self):
        """
        Generate a session id and crate directory for the session.
        :return: int: session id
        """

        self.session_id = str(np.random.randint(low=100, high=10000000000))

        while os.path.exists(data_popularity_data + self.session_id):
            self.session_id = str(np.random.randint(low=100, high=10000000000))

        os.makedirs(data_popularity_data + self.session_id)

        return self.session_id

    def allowed_file(self, filename):
        """
        Check a file extention
        :param string filename: name fo the file
        :return: boolean
        """
        return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

    def data_upload(self, session_id):
        """
        Method for data upload
        :param int session_id: a session id
        :return: int
        """

        file = request.files['file']

        if file and self.allowed_file(file.filename):

            data_path = data_popularity_data + session_id + '/' + 'data.csv'
            file.save(data_path)

        else:
            return 0

        return 1

    def take_params(self, req):
        """
        Take request parameters
        :param req: request
        :return: dict: parameters
        """

        get_params = ast.literal_eval(req)

        params = {}
        params['mode'] = get_params['mode'] if get_params.has_key('mode') else 'save'
        params['n_tb'] = get_params['n_tb'] if get_params.has_key('n_tb') else None
        params['proba_threshold'] = get_params['proba_threshold'] if get_params.has_key('proba_threshold') else 0.01
        params['min_replicas'] = get_params['min_replicas'] if get_params.has_key('min_replicas') else 1
        params['max_replicas'] = get_params['max_replicas'] if get_params.has_key('max_replicas') else 7

        return params






    def post(self):
        """
        Method for a POST request
        :return:
        """

        # generate a session id
        session_id = self.get_session_id()

        # upload data
        upload_msg = self.data_upload(session_id)



        # datapop
        params = self.take_params(request.form['params'])

        data_folder = data_popularity_data + session_id

        data_path = data_folder + '/data.csv'
        data = pd.read_csv(data_path)

        rps = ReplicationPlacementStrategy(data, params['min_replicas'], params['max_replicas'])

        if params['mode'] == 'save':
            report = rps.save_n_tb(params['n_tb'])

        elif params['mode'] == 'fill':
            report = rps.fill_n_tb(params['n_tb'])

        elif params['mode'] == 'clean':
            report = rps.clean_n_tb(params['n_tb'], params['proba_threshold'])

        elif params['mode'] == 'combine':
            report = rps.get_combine_report(data)

        report.to_csv(data_folder + '/report.csv')


        # return .csv report or a text message
        if upload_msg == 1:
            return send_from_directory(data_folder, filename='report.csv', as_attachment=True)

        else:
            return 'Upload data in .csv format.'


    def put(self):
        """
        Method for a PUT request
        :return:
        """
        self.post()

    def get(self):
        """
        Method for a GET request
        :return: text message
        """
        return "Hello, user! Use POST request."






#Add API
api.add_resource(DataPopularityApi, '/')


if __name__ == '__main__':
    #app.run(debug=False, host='0.0.0.0')
    app.run()