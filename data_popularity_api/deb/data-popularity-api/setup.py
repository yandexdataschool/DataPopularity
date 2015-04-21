__author__ = 'mikhail91'
from setuptools import setup, find_packages

setup(
    name='data-popularity-api-serv',
    version='0.1',
    url='',
    author='Mikhail Hushchyn',
    author_email='mikhail91@yandex-team.ru',
    install_requires=[
        'data-popularity-api==0.1',
        'flask',
        'flask-restful',
        'Werkzeug',
        'gunicorn',
    ],
)

