__author__ = 'mikhail91'

from setuptools import setup, find_packages

setup(
    name='datapopclient',
    version='0.1.0',
    url='',
    author='Mikhail Hushchyn',
    author_email='mikhail91@yandex-team.ru',
    packages=find_packages(),
    description='',
    include_package_data=True,
    py_modules = ['datapopclient'],
    install_requires=[
        'numpy == 1.9.2',
        'pandas == 0.14.0',
        'requests == 2.5.3',
    ],
)
