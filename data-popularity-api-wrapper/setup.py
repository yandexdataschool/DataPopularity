__author__ = 'mikhail91'

from setuptools import setup, find_packages

setup(
    name='data-popularity-api-wrapper',
    version='0.1.0',
    url='',
    author='Mikhail Hushchyn',
    author_email='mikhail91@yandex-team.ru',
    packages=find_packages(),
    description='',
    include_package_data=True,
    py_modules = ['data-popularity-api-wrapper'],
    install_requires=[
        'numpy >= 1.8.1',
        'pandas == 0.14.0',
        'requests',
    ],
)
